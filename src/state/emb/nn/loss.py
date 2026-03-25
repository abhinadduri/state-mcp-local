import torch
import torch.nn as nn
import torch.nn.functional as F
from geomloss import SamplesLoss


class WassersteinLoss(nn.Module):
    """
    Implements Wasserstein distance loss for distributions represented by logits.
    This implementation supports both 1D and 2D Wasserstein distance calculations.
    """

    def __init__(self, p=1, reduction="mean"):
        """
        Args:
            p (int): Order of Wasserstein distance (1 or 2)
            reduction (str): 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.p = p
        self.reduction = reduction

    def forward(self, p, q):
        """
        Compute Wasserstein distance between predicted and target distributions.

        Args:
            logits (torch.Tensor): Predicted logits of shape (batch_size, num_classes)
            target (torch.Tensor): Target probabilities of shape (batch_size, num_classes)
                                 or class indices of shape (batch_size,)

        Returns:
            torch.Tensor: Computed Wasserstein distance
        """

        q = torch.nan_to_num(q, nan=0.0)
        # Convert logits to probabilities
        pred_probs = F.softmax(p, dim=-1)
        q = F.softmax(q, dim=-1)

        # Compute cumulative distribution functions (CDFs)
        pred_cdf = torch.cumsum(pred_probs, dim=-1)
        target_cdf = torch.cumsum(q, dim=-1)

        max_len = max(pred_cdf.size(1), target_cdf.size(1))
        if pred_cdf.size(1) < max_len:
            pred_cdf = F.pad(pred_cdf, (0, max_len - pred_cdf.size(1)), "constant", 0)
        if target_cdf.size(1) < max_len:
            target_cdf = F.pad(target_cdf, (0, max_len - target_cdf.size(1)), "constant", 0)

        # Compute Wasserstein distance
        wasserstein_dist = torch.abs(pred_cdf - target_cdf).pow(self.p)
        wasserstein_dist = wasserstein_dist.sum(dim=-1)

        # Apply reduction if specified
        if self.reduction == "mean":
            return wasserstein_dist.mean()
        elif self.reduction == "sum":
            return wasserstein_dist.sum()
        return wasserstein_dist


class KLDivergenceLoss(nn.Module):
    def __init__(self, apply_normalization=False, epsilon=1e-10):
        super().__init__()
        self.apply_normalization = apply_normalization
        self.epsilon = epsilon

    def forward(self, p, q):
        q = torch.nan_to_num(q, nan=0.0)
        p = torch.nan_to_num(p, nan=0.0)

        max_len = max(p.size(1), q.size(1))
        if p.size(1) < max_len:
            p = F.pad(p, (0, max_len - p.size(1)), "constant", 0)
        if q.size(1) < max_len:
            q = F.pad(q, (0, max_len - q.size(1)), "constant", 0)

        if self.apply_normalization:
            p = F.softmax(p, dim=-1)
            q = F.softmax(q, dim=-1)

        return torch.sum(p * torch.log(p / q))


class MMDLoss(nn.Module):
    def __init__(self, kernel="energy", blur=0.05, scaling=0.5):
        super().__init__()
        self.mmd_loss = SamplesLoss(loss=kernel, blur=blur, scaling=scaling)

    def forward(self, input, target, downsample=1):
        input = input.reshape(-1, downsample, input.shape[-1])
        target = target.reshape(-1, downsample, target.shape[-1])

        # Pre-compute uniform weights on the correct device to avoid
        # geomloss creating CPU tensors then calling .type_as() which
        # triggers a cudaStreamSynchronize on every forward pass.
        B, N, _ = input.shape
        α = torch.ones(B, N, device=input.device, dtype=input.dtype) / N
        β = torch.ones(B, N, device=target.device, dtype=target.dtype) / N

        loss = self.mmd_loss(α, input, β, target)
        return loss.mean()


class TabularLoss(nn.Module):
    def __init__(self, shared=128):
        super().__init__()
        self.shared = shared

        self.gene_loss = SamplesLoss(loss="energy")
        self.cell_loss = SamplesLoss(loss="energy")

    def forward(self, input, target, downsample=1):
        """Dual energy-distance loss: gene-level MMD + cell-level MMD.

        Gene-level: treats each cell's decoder outputs as a point cloud over
        genes, then compares predicted vs target distributions per cell.

        Cell-level: for the S shared genes (the last ``self.shared`` features),
        transposes so each gene becomes a "sample set" of cells, then compares
        predicted vs target cell distributions per gene.  This encourages the
        model to match cross-cell structure, not just per-cell reconstruction.
        """
        # Group augmented copies together:
        # (B*D, P+N+S) -> (B, D, P+N+S)  where D = downsample count
        input = input.reshape(-1, downsample, input.shape[-1])
        target = target.reshape(-1, downsample, target.shape[-1])

        # Pre-compute uniform weights on GPU to avoid geomloss type_as syncs
        B, N, _ = input.shape
        α_gene = torch.ones(B, N, device=input.device, dtype=input.dtype) / N
        β_gene = torch.ones(B, N, device=target.device, dtype=target.dtype) / N
        gene_mmd = self.gene_loss(α_gene, input, β_gene, target).nanmean()

        # Extract the S shared genes (last S columns) for cross-cell comparison
        cell_inputs = input[:, :, -self.shared :]
        cell_targets = target[:, :, -self.shared :]

        # Transpose (B, D, S) -> (S, D, B): each shared gene becomes a "sample"
        # of (downsample, batch) points, so SamplesLoss compares per-gene
        # distributions across cells
        cell_inputs = cell_inputs.transpose(2, 0)
        cell_targets = cell_targets.transpose(2, 0)

        S, D, _ = cell_inputs.shape
        α_cell = torch.ones(S, D, device=cell_inputs.device, dtype=cell_inputs.dtype) / D
        β_cell = torch.ones(S, D, device=cell_targets.device, dtype=cell_targets.dtype) / D
        cell_mmd = self.cell_loss(α_cell, cell_inputs, β_cell, cell_targets).nanmean()

        # Combine losses, skipping any that are NaN (can happen with degenerate batches)
        final_loss = torch.tensor(0.0, device=cell_mmd.device, dtype=cell_mmd.dtype)
        if not gene_mmd.isnan():
            final_loss += gene_mmd
        if not cell_mmd.isnan():
            final_loss += cell_mmd

        return final_loss
