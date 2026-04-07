"""Cell-set data loading for tabular attention.

Groups cells by SRX (h5ad file) for inter-cellular attention.
Each batch item is a set of n_cells_per_set cells from the same experiment.
"""

import logging
from typing import Optional, NamedTuple

import h5py
import numpy as np
import torch
import torch.utils.data as data

from .. import utils
from .loader import H5adSentenceDataset

log = logging.getLogger(__name__)


class TabularLatentBatch(NamedTuple):
    """Output of TabularLatentCollator. Flat cells with cell-set metadata."""

    gene_indices: torch.Tensor       # [B_total, k_max]
    gene_counts: torch.Tensor        # [B_total, k_max]
    gene_mask: torch.Tensor          # [B_total, k_max] bool
    task_genes: torch.Tensor         # [B_total, P+N]
    task_counts: torch.Tensor        # [B_total, P+N]
    dataset_nums: Optional[torch.Tensor]  # [B_total]
    n_cells_per_set: int             # cells per set (constant)


class CellSetH5adDataset(H5adSentenceDataset):
    """Dataset that yields groups of cells from the same h5ad file.

    Each __getitem__ returns a list of n_cells_per_set (counts, idx, dataset, dataset_num)
    tuples, all from the same SRX/experiment.
    """

    def __init__(self, cfg, n_cells_per_set=32, test=False, **kwargs):
        super().__init__(cfg, test=test, **kwargs)
        self.n_cells_per_set = n_cells_per_set

        # Pre-compute cell set boundaries: (dataset_name, start_cell_idx)
        self.cell_sets = []
        for name in self.datasets:
            n_cells = self.num_cells[name]
            n_sets = n_cells // n_cells_per_set
            for i in range(n_sets):
                self.cell_sets.append((name, i * n_cells_per_set))

        log.info(
            f"CellSetH5adDataset: {len(self.cell_sets)} cell sets "
            f"({n_cells_per_set} cells each) from {len(self.datasets)} files"
        )

    def __len__(self):
        return len(self.cell_sets)

    def __getitem__(self, idx):
        dataset_name, start_idx = self.cell_sets[idx]
        dataset_num = self.datasets_to_num[dataset_name]
        h5f = self.dataset_file(dataset_name)
        attrs = dict(h5f["X"].attrs)
        n_genes = self.num_genes[dataset_name]

        cells = []
        if attrs.get("encoding-type") == "csr_matrix":
            # Batch-read indptrs for efficiency
            ptrs = np.array(
                h5f["/X/indptr"][start_idx : start_idx + self.n_cells_per_set + 1]
            )
            all_data = np.array(
                h5f["/X/data"][int(ptrs[0]) : int(ptrs[-1])]
            )
            all_indices = np.array(
                h5f["/X/indices"][int(ptrs[0]) : int(ptrs[-1])]
            )
            base_ptr = int(ptrs[0])

            for i in range(self.n_cells_per_set):
                sp = int(ptrs[i]) - base_ptr
                ep = int(ptrs[i + 1]) - base_ptr
                sub_data = torch.tensor(all_data[sp:ep], dtype=torch.float)
                sub_indices = torch.tensor(all_indices[sp:ep], dtype=torch.int32)
                counts = torch.sparse_csr_tensor(
                    torch.tensor([0, len(sub_data)]),
                    sub_indices,
                    sub_data,
                    (1, n_genes),
                )
                counts = counts.to_dense()
                cells.append((counts, start_idx + i, dataset_name, dataset_num))
        else:
            # Dense matrix — batch read
            chunk = torch.tensor(
                np.array(h5f["X"][start_idx : start_idx + self.n_cells_per_set])
            )
            for i in range(self.n_cells_per_set):
                cells.append(
                    (chunk[i].unsqueeze(0), start_idx + i, dataset_name, dataset_num)
                )

        return cells


class TabularLatentCollator:
    """Collator that flattens cell sets and produces TabularLatentBatch.

    Reuses LatentCollator's per-cell processing logic.
    """

    def __init__(self, cfg, ds_emb_mapping, n_genes, is_train=True, k_top=None, n_cells_per_set=32):
        # Import here to avoid circular imports
        from ..nn.tokenizer import LatentCollator

        self.inner = LatentCollator(
            cfg=cfg,
            ds_emb_mapping=ds_emb_mapping,
            n_genes=n_genes,
            is_train=is_train,
            k_top=k_top,
        )
        self.n_cells_per_set = n_cells_per_set
        self.cfg = cfg

    def __call__(self, batch):
        """
        Args:
            batch: list of cell sets, each cell set is a list of
                   (counts, idx, dataset, dataset_num) tuples.
        Returns:
            TabularLatentBatch with flattened cells and n_cells_per_set metadata.
        """
        # Flatten cell sets into a single list of cells
        flat_batch = []
        for cell_set in batch:
            flat_batch.extend(cell_set)

        # Use LatentCollator's collation logic
        latent_batch = self.inner(flat_batch)

        return TabularLatentBatch(
            gene_indices=latent_batch.gene_indices,
            gene_counts=latent_batch.gene_counts,
            gene_mask=latent_batch.gene_mask,
            task_genes=latent_batch.task_genes,
            task_counts=latent_batch.task_counts,
            dataset_nums=latent_batch.dataset_nums,
            n_cells_per_set=self.n_cells_per_set,
        )
