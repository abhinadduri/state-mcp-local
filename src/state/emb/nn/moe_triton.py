"""Triton-fused MoE kernels for Expert Parallel forward.

Replaces the ~25 small PyTorch ops in the EP pack phase with a single fused kernel,
and the ~8 ops in the unpack phase with another fused kernel.
This reduces kernel launches from ~46 to ~8 per MoE layer forward pass.

Key kernels:
- moe_pack_kernel: sort tokens by expert, weight, and pack into padded [E, C, D] buffer
- moe_unpack_kernel: unpack results from padded buffer, apply routing weights, unsort
"""

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _moe_pack_kernel(
        # Token data
        x_ptr,           # [N, D] input tokens (bf16)
        weights_ptr,     # [M] routing weights (bf16)
        expert_ids_ptr,  # [M] expert assignments (int64)
        token_ids_ptr,   # [M] token indices into x (int64)
        # Output
        packed_ptr,      # [E, C, D] output padded buffer (bf16)
        expert_counts_ptr,  # [E] number of tokens per expert (int64)
        # Mapping output (for unpack)
        pack_expert_ptr,    # [M] expert id for each assignment (int64)
        pack_position_ptr,  # [M] position within expert (int64)
        pack_sort_ptr,      # [M] sort index for unsort in unpack (int64)
        # Dimensions
        N: tl.constexpr, D: tl.constexpr, M: tl.constexpr,
        E: tl.constexpr, C: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Pack tokens into expert-sorted padded buffer using atomic counters.

        Each program instance handles one token-expert assignment.
        Uses atomic_add on per-expert counters to determine pack position.
        Dropless: C is guaranteed >= max tokens per expert, so all tokens fit.
        """
        pid = tl.program_id(0)
        if pid >= M:
            return

        expert = tl.load(expert_ids_ptr + pid)
        token_idx = tl.load(token_ids_ptr + pid)
        weight = tl.load(weights_ptr + pid)

        # Atomically claim a position for this token in the expert's buffer
        position = tl.atomic_add(expert_counts_ptr + expert, 1)

        # Save mapping for unpack phase
        tl.store(pack_expert_ptr + pid, expert)
        tl.store(pack_position_ptr + pid, position)
        tl.store(pack_sort_ptr + pid, pid)  # identity (no sort needed with atomics)

        # Copy weighted token into packed buffer
        d_offsets = tl.arange(0, BLOCK_D)
        for d_start in range(0, D, BLOCK_D):
            d_idx = d_start + d_offsets
            mask = d_idx < D
            token_val = tl.load(x_ptr + token_idx * D + d_idx, mask=mask)
            packed_val = token_val * weight
            tl.store(packed_ptr + expert * C * D + position * D + d_idx, packed_val, mask=mask)

    @triton.jit
    def _moe_unpack_kernel(
        # Combine buffer
        recv_ptr,           # [E, C, D] combined results (bf16)
        # Mapping from pack phase
        expert_ids_ptr,     # [M] expert id per assignment
        positions_ptr,      # [M] position within expert
        # Token mapping
        token_ids_ptr,      # [M] original token indices
        # Output
        output_ptr,         # [N, D] output (bf16)
        # Dimensions
        N: tl.constexpr, D: tl.constexpr, M: tl.constexpr,
        E: tl.constexpr, C: tl.constexpr, K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Unpack results from padded buffer back to token order.

        Each program handles one token-expert assignment, atomically accumulating
        into the output (handles top-K > 1 by accumulation).
        Dropless: all tokens were packed, so all are unpacked unconditionally.
        """
        pid = tl.program_id(0)
        if pid >= M:
            return

        expert = tl.load(expert_ids_ptr + pid)
        position = tl.load(positions_ptr + pid)
        token_idx = tl.load(token_ids_ptr + pid)

        d_offsets = tl.arange(0, BLOCK_D)
        for d_start in range(0, D, BLOCK_D):
            d_idx = d_start + d_offsets
            mask = d_idx < D
            result = tl.load(recv_ptr + expert * C * D + position * D + d_idx, mask=mask)
            # Atomic add for top-K accumulation
            tl.atomic_add(output_ptr + token_idx * D + d_idx, result, mask=mask)


def triton_ep_pack(x_flat, top_k_weights, top_k_indices, E, C):
    """Fused pack: route tokens + weight + pack into padded [E, C, D] buffer.

    Replaces ~20 PyTorch ops with 1 Triton kernel.
    Dropless: C is guaranteed >= max tokens per expert.
    Returns: (packed_buffer, token_ids, expert_ids, positions)
    """
    N, D = x_flat.shape
    K = top_k_indices.shape[1]
    M = N * K

    # Flatten top-K assignments
    flat_experts = top_k_indices.reshape(-1)  # [M]
    flat_weights = top_k_weights.reshape(-1)  # [M]
    flat_token_idx = torch.arange(N, device=x_flat.device).unsqueeze(1).expand(-1, K).reshape(-1)  # [M]

    # Output buffers
    packed = torch.zeros(E, C, D, device=x_flat.device, dtype=x_flat.dtype)
    expert_counts = torch.zeros(E, device=x_flat.device, dtype=torch.int64)

    # Mapping buffers (for unpack)
    pack_expert = torch.empty(M, device=x_flat.device, dtype=torch.int64)
    pack_position = torch.empty(M, device=x_flat.device, dtype=torch.int64)
    pack_sort = torch.empty(M, device=x_flat.device, dtype=torch.int64)

    BLOCK_D = min(triton.next_power_of_2(D), 1024)

    _moe_pack_kernel[(M,)](
        x_flat, flat_weights, flat_experts, flat_token_idx,
        packed, expert_counts,
        pack_expert, pack_position, pack_sort,
        N, D, M, E, C, BLOCK_D,
    )

    return packed, flat_token_idx, pack_expert, pack_position


def triton_ep_unpack(recv_flat, token_ids, expert_ids, positions, N, E, C, K):
    """Fused unpack: extract results from padded buffer back to token order.

    Replaces ~8 PyTorch ops with 1 Triton kernel.
    Dropless: all tokens were packed, so all are unconditionally unpacked.
    """
    D = recv_flat.shape[1]
    M = expert_ids.shape[0]

    output = torch.zeros(N, D, device=recv_flat.device, dtype=recv_flat.dtype)

    BLOCK_D = min(triton.next_power_of_2(D), 1024)

    _moe_unpack_kernel[(M,)](
        recv_flat.reshape(E, C, D),
        expert_ids, positions, token_ids,
        output,
        N, D, M, E, C, K, BLOCK_D,
    )

    return output
