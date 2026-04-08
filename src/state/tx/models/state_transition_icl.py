"""
In-Context Learning State Transition Model.

Extends StateTransitionPerturbationModel with demonstration-based ICL:
- Prepends encoded demonstration (ctrl → pert) pairs to the query sequence
- Uses segment embeddings to distinguish demo vs query tokens
- Supports variable numbers of demonstrations (including zero)
- During training, randomly samples demos from same context with dropout

Architecture:
    Demo tokens:  basal_encoder(ctrl) + pert_encoder(pert_id) + effect_encoder(pert_effect) + seg(0)
    Query tokens: basal_encoder(ctrl) + pert_encoder(pert_id) + seg(1)
    Sequence:     [demo_tokens | query_tokens] → transformer → extract query outputs → project_out
"""

import logging
import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .state_transition import StateTransitionPerturbationModel
from .utils import build_mlp

logger = logging.getLogger(__name__)


class ICLStateTransitionPerturbationModel(StateTransitionPerturbationModel):
    """State Transition model with In-Context Learning via demonstration concatenation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        pert_dim: int,
        batch_dim: int = None,
        # ICL-specific parameters
        n_demo_perts: int = 4,
        n_demo_cells_per_pert: int = 8,
        demo_dropout: float = 0.3,
        use_segment_embeddings: bool = True,
        **kwargs,
    ):
        """
        Args:
            n_demo_perts: Number of different demonstration perturbations to sample.
            n_demo_cells_per_pert: Number of cells per demonstration perturbation.
            demo_dropout: Probability of providing zero demonstrations during training.
            use_segment_embeddings: Whether to use segment embeddings for demo/query.
        """
        self._icl_n_demo_perts = n_demo_perts
        self._icl_n_demo_cells_per_pert = n_demo_cells_per_pert
        self._icl_demo_dropout = demo_dropout
        self._icl_use_segment_embeddings = use_segment_embeddings
        self._icl_cross_context_prob = kwargs.get("cross_context_prob", 0.0)

        # Increase n_positions to accommodate demo tokens
        max_demo_tokens = n_demo_perts * n_demo_cells_per_pert
        extra_tokens = kwargs.get("extra_tokens", 0) + max_demo_tokens
        kwargs["extra_tokens"] = extra_tokens

        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            pert_dim=pert_dim,
            batch_dim=batch_dim,
            **kwargs,
        )

        # Demo cache: populated during on_fit_start
        # Structure: cell_type_str -> list of (pert_name, ctrl_mean [output_dim], pert_mean [output_dim], pert_onehot [pert_dim])
        self._demo_cache: Dict[str, List[Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]]] = {}
        self._demo_cache_ready = False

    def _build_networks(self, lora_cfg=None):
        """Build parent networks + ICL-specific modules."""
        super()._build_networks(lora_cfg=lora_cfg)

        # Effect encoder: encodes the observed perturbation effect (perturbed cell expression)
        self.effect_encoder = build_mlp(
            in_dim=self.output_dim,
            out_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_encoder_layers,
            dropout=self.dropout,
            activation=self.activation_class,
        )

        # Segment embeddings: 0=demonstration, 1=query
        if self._icl_use_segment_embeddings:
            self.segment_embedding = nn.Embedding(2, self.hidden_dim)
            nn.init.zeros_(self.segment_embedding.weight)
        else:
            self.segment_embedding = None

    def encode_demonstration(
        self,
        demo_ctrl: torch.Tensor,
        demo_pert: torch.Tensor,
        demo_effect: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode demonstration tokens.

        Args:
            demo_ctrl: Control cell expressions [B, D, input_dim]
            demo_pert: Perturbation identity (one-hot) [B, D, pert_dim]
            demo_effect: Observed perturbed cell expressions [B, D, output_dim]

        Returns:
            Demo tokens [B, D, hidden_dim]
        """
        ctrl_enc = self.encode_basal_expression(demo_ctrl)
        pert_enc = self.encode_perturbation(demo_pert)
        effect_enc = self.effect_encoder(demo_effect)

        demo_tokens = ctrl_enc + pert_enc + effect_enc

        if self.segment_embedding is not None:
            demo_seg = self.segment_embedding(
                torch.zeros(demo_tokens.shape[:2], dtype=torch.long, device=demo_tokens.device)
            )
            demo_tokens = demo_tokens + demo_seg

        return demo_tokens

    def forward(
        self,
        batch: dict,
        padded: bool = True,
        return_nb_params: bool = False,
        demo_batch: Optional[dict] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional demonstrations.

        Args:
            batch: Standard batch dict with pert_emb, ctrl_cell_emb, etc.
            padded: Whether batch is padded to cell_sentence_len.
            return_nb_params: Whether to return NB parameters.
            demo_batch: Optional dict with keys:
                - demo_ctrl: [B, D, input_dim] control cells for demos
                - demo_pert: [B, D, pert_dim] perturbation IDs for demos
                - demo_effect: [B, D, output_dim] observed perturbed cells for demos
                where D = total demo tokens (n_demo_perts * n_demo_cells_per_pert)
        """
        # Encode query tokens (same as parent)
        if padded:
            pert = batch["pert_emb"].reshape(-1, self.cell_sentence_len, self.pert_dim)
            basal = batch["ctrl_cell_emb"].reshape(-1, self.cell_sentence_len, self.input_dim)
        else:
            pert = batch["pert_emb"].reshape(1, -1, self.pert_dim)
            basal = batch["ctrl_cell_emb"].reshape(1, -1, self.input_dim)

        B = pert.shape[0]
        S = pert.shape[1]

        pert_embedding = self.encode_perturbation(pert)
        control_cells = self.encode_basal_expression(basal)
        query_tokens = pert_embedding + control_cells

        # Add segment embedding for query tokens
        if self.segment_embedding is not None:
            query_seg = self.segment_embedding(
                torch.ones(B, S, dtype=torch.long, device=query_tokens.device)
            )
            query_tokens = query_tokens + query_seg

        # Add batch embeddings if configured
        if self.batch_encoder is not None:
            batch_indices = batch["batch"]
            if batch_indices.dim() > 1 and batch_indices.size(-1) == self.batch_dim:
                batch_indices = batch_indices.argmax(-1)
            if padded:
                batch_indices = batch_indices.reshape(-1, self.cell_sentence_len)
            else:
                batch_indices = batch_indices.reshape(1, -1)
            batch_embeddings = self.batch_encoder(batch_indices.long())
            query_tokens = query_tokens + batch_embeddings

        # Encode and prepend demonstration tokens
        n_demo_tokens = 0
        if demo_batch is not None and demo_batch.get("demo_ctrl") is not None:
            demo_ctrl = demo_batch["demo_ctrl"]  # [B, D, input_dim]
            demo_pert = demo_batch["demo_pert"]  # [B, D, pert_dim]
            demo_effect = demo_batch["demo_effect"]  # [B, D, output_dim]

            demo_tokens = self.encode_demonstration(demo_ctrl, demo_pert, demo_effect)
            n_demo_tokens = demo_tokens.shape[1]

            # Concatenate: [demo_tokens | query_tokens]
            seq_input = torch.cat([demo_tokens, query_tokens], dim=1)
        else:
            seq_input = query_tokens

        # Forward pass through transformer
        if self.hparams.get("mask_attn", False):
            batch_size, seq_length, _ = seq_input.shape
            device = seq_input.device
            self.transformer_backbone._attn_implementation = "eager"
            base = torch.eye(seq_length, device=device, dtype=torch.bool).view(1, 1, seq_length, seq_length)
            num_heads = self.transformer_backbone.config.num_attention_heads
            attn_mask = base.repeat(batch_size, num_heads, 1, 1)
            outputs = self.transformer_backbone(inputs_embeds=seq_input, attention_mask=attn_mask)
            transformer_output = outputs.last_hidden_state
        else:
            outputs = self.transformer_backbone(inputs_embeds=seq_input)
            transformer_output = outputs.last_hidden_state

        # Extract only query token outputs (skip demo tokens)
        if n_demo_tokens > 0:
            res_pred = transformer_output[:, n_demo_tokens:, :]
        else:
            res_pred = transformer_output

        # Project out (same as parent)
        if self.predict_residual and self.output_space == "all":
            out_pred = self.project_out(res_pred) + basal
            out_pred = self.final_down_then_up(out_pred)
        elif self.predict_residual:
            out_pred = self.project_out(res_pred + control_cells)
        else:
            out_pred = self.project_out(res_pred)

        is_gene_space = self.hparams["embed_key"] == "X_hvg" or self.hparams["embed_key"] is None
        if is_gene_space or (self.gene_decoder is None and not self.nb_loss):
            out_pred = self.relu(out_pred)

        # NB loss handling (same as parent)
        nb_mean = None
        nb_dispersion = None
        if self.nb_loss:
            if self.nb_parameter_head is None:
                raise RuntimeError("nb_loss=True but nb_parameter_head was not initialized.")
            nb_params = self.nb_parameter_head(out_pred)
            px_scale_logits, nb_dispersion_logits = torch.chunk(nb_params, chunks=2, dim=-1)
            px_scale = self._apply_nb_scale_activation(px_scale_logits)
            if self.nb_library_size_mode == "predicted":
                if self.nb_library_head is None:
                    raise RuntimeError("nb_library_size_mode='predicted' requires nb_library_head.")
                library_sizes = F.softplus(self.nb_library_head(out_pred)) + 1.0
            else:
                ctrl_for_library = self._get_nb_control_tensor_for_library(batch, basal, padded)
                library_sizes = self._compute_library_sizes_from_control(ctrl_for_library, self.nb_library_size_mode)
            nb_mean = px_scale * library_sizes
            nb_dispersion = F.softplus(nb_dispersion_logits) + self.nb_eps

        output = out_pred.reshape(-1, self.output_dim)

        if not self.nb_loss or not return_nb_params:
            return output
        if nb_mean is None or nb_dispersion is None:
            raise RuntimeError("nb_loss=True but NB parameters were not produced.")
        return output, nb_mean.reshape(-1, self.nb_target_dim), nb_dispersion.reshape(-1, self.nb_target_dim)

    # ------------------------------------------------------------------
    # Demo cache management
    # ------------------------------------------------------------------

    def build_demo_cache(self, datamodule) -> None:
        """
        Build demonstration cache from training dataset info.

        Uses the data module's split information to compute per-(cell_type, perturbation)
        mean expressions directly from h5ad files, avoiding slow dataloader iteration.
        """
        logger.info("Building ICL demo cache from training data...")

        import h5py

        embed_key = self.hparams.get("embed_key", "X_hvg")
        cell_type_col = "cell_line"  # default for replogle
        pert_col = "gene"

        # Get perturbation one-hot map from the data module
        pert_onehot_map = getattr(datamodule, "pert_onehot_map", {})

        # Collect means per (cell_type, pert) from all training h5ad files
        ct_pert_data = defaultdict(lambda: defaultdict(lambda: {"pert_sum": None, "ctrl_sum": None, "count": 0}))

        # Deduplicate h5ad files across Subset datasets
        seen_paths = set()
        for ds in datamodule.train_datasets:
            dataset = ds.dataset if hasattr(ds, "dataset") else ds
            file_path = getattr(dataset, "h5_path", None)
            if file_path is None:
                continue
            file_path = str(file_path)
            if file_path in seen_paths:
                continue
            seen_paths.add(file_path)

            logger.info("Loading demo data from: %s", file_path)
            try:
                with h5py.File(file_path, "r") as f:
                    # Read cell type and perturbation labels
                    if cell_type_col in f["obs"]:
                        ct_data = f["obs"][cell_type_col]
                        if "categories" in ct_data:
                            cats = [c.decode() if isinstance(c, bytes) else c for c in ct_data["categories"][:]]
                            codes = ct_data["codes"][:]
                            cell_types = [cats[c] for c in codes]
                        else:
                            cell_types = [x.decode() if isinstance(x, bytes) else x for x in ct_data[:]]
                    else:
                        continue

                    if pert_col in f["obs"]:
                        pt_data = f["obs"][pert_col]
                        if "categories" in pt_data:
                            cats = [c.decode() if isinstance(c, bytes) else c for c in pt_data["categories"][:]]
                            codes = pt_data["codes"][:]
                            perts = [cats[c] for c in codes]
                        else:
                            perts = [x.decode() if isinstance(x, bytes) else x for x in pt_data[:]]
                    else:
                        continue

                    # Read HVG embeddings
                    if embed_key and embed_key in f["obsm"]:
                        emb = f["obsm"][embed_key]
                        n_cells = emb.shape[0]
                        dim = emb.shape[1]
                    else:
                        continue

                    # Process in chunks for memory efficiency
                    chunk_size = 10000
                    for start in range(0, n_cells, chunk_size):
                        end = min(start + chunk_size, n_cells)
                        chunk = torch.tensor(emb[start:end], dtype=torch.float32)

                        for i in range(end - start):
                            ct = cell_types[start + i]
                            pn = perts[start + i]
                            entry = ct_pert_data[ct][pn]

                            vec = chunk[i]
                            if entry["pert_sum"] is None:
                                entry["pert_sum"] = vec.clone()
                                entry["ctrl_sum"] = torch.zeros_like(vec)
                            else:
                                entry["pert_sum"] += vec
                            entry["count"] += 1

            except Exception as e:
                logger.warning("Failed to load demo data from %s: %s", file_path, e)
                continue

        # For control means: use the control perturbation entries
        control_means = {}
        for ct in ct_pert_data:
            ctrl_entry = ct_pert_data[ct].get(self.control_pert)
            if ctrl_entry is not None and ctrl_entry["count"] > 0:
                control_means[ct] = ctrl_entry["pert_sum"] / ctrl_entry["count"]

        # Build the demo cache
        self._demo_cache = {}
        for ct in ct_pert_data:
            entries = []
            ctrl_mean = control_means.get(ct)
            if ctrl_mean is None:
                continue

            for pn in ct_pert_data[ct]:
                if pn == self.control_pert:
                    continue
                entry = ct_pert_data[ct][pn]
                if entry["count"] < 5:
                    continue

                pert_mean = entry["pert_sum"] / entry["count"]

                # Get one-hot encoding from the data module's pert map
                if pn in pert_onehot_map:
                    onehot = pert_onehot_map[pn].cpu() if isinstance(pert_onehot_map[pn], torch.Tensor) else torch.tensor(pert_onehot_map[pn])
                else:
                    onehot = torch.zeros(self.pert_dim)

                entries.append((pn, ctrl_mean, pert_mean, onehot))

            if entries:
                self._demo_cache[ct] = entries

        total = sum(len(v) for v in self._demo_cache.values())
        logger.info(
            "ICL demo cache built: %d cell types, %d total (cell_type, pert) entries",
            len(self._demo_cache),
            total,
        )
        self._demo_cache_ready = True

    def _sample_demo_batch(
        self,
        cell_types: List[str],
        pert_names: List[str],
        batch_size: int,
        device: torch.device,
    ) -> Optional[dict]:
        """
        Sample demonstration data for a training batch.

        Args:
            cell_types: Cell type per cell in the batch.
            pert_names: Perturbation name per cell in the batch.
            batch_size: Number of sentences (B).
            device: Target device.

        Returns:
            Demo batch dict or None if no demos available.
        """
        if not self._demo_cache_ready:
            return None

        # Demo dropout: with probability demo_dropout, return no demos
        if self.training and torch.rand(1).item() < self._icl_demo_dropout:
            return None

        n_demo_perts = self._icl_n_demo_perts
        n_cells_per = self._icl_n_demo_cells_per_pert
        total_demo_cells = n_demo_perts * n_cells_per

        # Identify unique cell types in this batch (usually just one)
        # Group by cell_type at the sentence level
        # Each sentence has cell_sentence_len cells, all from same (ct, pert)
        ct_per_sentence = cell_types[:: self.cell_sentence_len][:batch_size]
        pn_per_sentence = pert_names[:: self.cell_sentence_len][:batch_size]

        demo_ctrl_list = []
        demo_pert_list = []
        demo_effect_list = []

        for b_idx in range(batch_size):
            ct = ct_per_sentence[b_idx] if b_idx < len(ct_per_sentence) else ct_per_sentence[-1]
            query_pn = pn_per_sentence[b_idx] if b_idx < len(pn_per_sentence) else pn_per_sentence[-1]

            # Cross-context demo sampling: with probability cross_context_prob,
            # sample demos from a DIFFERENT cell type. This teaches the model to
            # extract useful perturbation info from foreign contexts.
            use_cross_context = (
                self.training
                and self._icl_cross_context_prob > 0
                and torch.rand(1).item() < self._icl_cross_context_prob
            )

            if use_cross_context:
                other_cts = [c for c in self._demo_cache if c != ct]
                if other_cts:
                    demo_ct = other_cts[torch.randint(len(other_cts), (1,)).item()]
                else:
                    demo_ct = ct
            else:
                demo_ct = ct

            cache = self._demo_cache.get(demo_ct, [])
            # Filter out the query perturbation and controls
            candidates = [(pn, c, p, oh) for pn, c, p, oh in cache if pn != query_pn and pn != self.control_pert]

            if len(candidates) == 0:
                # No demos available for this cell type - return zeros
                demo_ctrl_list.append(torch.zeros(total_demo_cells, self.input_dim))
                demo_pert_list.append(torch.zeros(total_demo_cells, self.pert_dim))
                demo_effect_list.append(torch.zeros(total_demo_cells, self.output_dim))
                continue

            # Sample n_demo_perts perturbations (with replacement if needed)
            n_sample = min(n_demo_perts, len(candidates))
            indices = torch.randperm(len(candidates))[:n_sample].tolist()

            b_ctrl = []
            b_pert = []
            b_effect = []

            for idx in indices:
                pn, ctrl_mean, pert_mean, onehot = candidates[idx]
                # Replicate mean to n_cells_per (with small noise for diversity)
                noise_scale = 0.01
                ctrl_cells = ctrl_mean.unsqueeze(0).expand(n_cells_per, -1)
                ctrl_cells = ctrl_cells + noise_scale * torch.randn_like(ctrl_cells)
                pert_cells = pert_mean.unsqueeze(0).expand(n_cells_per, -1)
                pert_cells = pert_cells + noise_scale * torch.randn_like(pert_cells)
                pert_oh = onehot.unsqueeze(0).expand(n_cells_per, -1)

                b_ctrl.append(ctrl_cells)
                b_pert.append(pert_oh)
                b_effect.append(pert_cells)

            # Pad if we sampled fewer than n_demo_perts
            while len(b_ctrl) < n_demo_perts:
                b_ctrl.append(torch.zeros(n_cells_per, self.input_dim))
                b_pert.append(torch.zeros(n_cells_per, self.pert_dim))
                b_effect.append(torch.zeros(n_cells_per, self.output_dim))

            demo_ctrl_list.append(torch.cat(b_ctrl, dim=0))  # [total_demo_cells, input_dim]
            demo_pert_list.append(torch.cat(b_pert, dim=0))
            demo_effect_list.append(torch.cat(b_effect, dim=0))

        demo_batch = {
            "demo_ctrl": torch.stack(demo_ctrl_list).to(device),  # [B, D, input_dim]
            "demo_pert": torch.stack(demo_pert_list).to(device),  # [B, D, pert_dim]
            "demo_effect": torch.stack(demo_effect_list).to(device),  # [B, D, output_dim]
        }
        return demo_batch

    # ------------------------------------------------------------------
    # Training / validation overrides
    # ------------------------------------------------------------------

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, padded=True) -> torch.Tensor:
        """Training step with demonstration sampling."""
        B = batch["pert_emb"].shape[0] // self.cell_sentence_len if padded else 1

        # Sample demonstrations from cache
        demo_batch = self._sample_demo_batch(
            cell_types=batch.get("cell_type", []),
            pert_names=batch.get("pert_name", []),
            batch_size=B,
            device=batch["pert_emb"].device,
        )

        # Forward with demos
        if self.nb_loss:
            pred, nb_mean_flat, nb_dispersion_flat = self.forward(
                batch, padded=padded, return_nb_params=True, demo_batch=demo_batch
            )
        else:
            pred = self.forward(batch, padded=padded, demo_batch=demo_batch)
        self._log_residual_metrics_if_ready()

        target = batch["pert_cell_emb"]

        if padded:
            pred = pred.reshape(-1, self.cell_sentence_len, self.output_dim)
            target = target.reshape(-1, self.cell_sentence_len, self.output_dim)
        else:
            pred = pred.reshape(1, -1, self.output_dim)
            target = target.reshape(1, -1, self.output_dim)

        # Loss computation (same as parent)
        embedding_aux_loss = None
        nb_log1p_mse_aux_loss = None
        nb_library_mse_aux_loss = None
        if self.nb_loss:
            if padded:
                nb_mean = nb_mean_flat.reshape(-1, self.cell_sentence_len, self.nb_target_dim)
                nb_dispersion = nb_dispersion_flat.reshape(-1, self.cell_sentence_len, self.nb_target_dim)
            else:
                nb_mean = nb_mean_flat.reshape(1, -1, self.nb_target_dim)
                nb_dispersion = nb_dispersion_flat.reshape(1, -1, self.nb_target_dim)

            nb_target = self._get_nb_target_tensor(batch, target, padded)
            per_set_main_losses = self._compute_nb_nll_loss(nb_mean, nb_dispersion, nb_target)
            if self.nb_embed_loss_weight > 0.0:
                embedding_aux_losses = self._compute_distribution_loss(pred, target)
                embedding_aux_loss = torch.nanmean(embedding_aux_losses)
                self.log("train/embedding_loss", embedding_aux_loss)
            if self.nb_log1p_mse_weight > 0.0:
                nb_log1p_mse_per_set = self._compute_nb_log1p_mse_per_set(nb_mean, nb_target)
                nb_log1p_mse_aux_loss = torch.nanmean(nb_log1p_mse_per_set)
                self.log("train/nb_log1p_mse_loss", nb_log1p_mse_aux_loss)
            nb_library_mse_per_set = self._compute_nb_library_mse_per_set(nb_mean, nb_target)
            nb_library_mse_metric = torch.nanmean(nb_library_mse_per_set)
            self.log("train/nb_library_mse", nb_library_mse_metric)
            if self.nb_library_mse_weight > 0.0:
                nb_library_mse_aux_loss = nb_library_mse_metric
        else:
            per_set_main_losses = self._compute_distribution_loss(pred, target)
        main_loss = torch.nanmean(per_set_main_losses)
        self.log(self._train_main_loss_key(), main_loss)

        total_loss = main_loss
        if embedding_aux_loss is not None:
            total_loss = total_loss + self.nb_embed_loss_weight * embedding_aux_loss
        if nb_log1p_mse_aux_loss is not None:
            total_loss = total_loss + self.nb_log1p_mse_weight * nb_log1p_mse_aux_loss
        if nb_library_mse_aux_loss is not None:
            total_loss = total_loss + self.nb_library_mse_weight * nb_library_mse_aux_loss

        # Decoder loss
        if (not self.nb_loss) and self.gene_decoder is not None and "pert_cell_counts" in batch:
            gene_targets = batch["pert_cell_counts"]
            if self.detach_decoder:
                if np.random.rand() < 0.1:
                    latent_preds = target.reshape_as(pred).detach()
                else:
                    latent_preds = pred.detach()
            else:
                latent_preds = pred

            pert_cell_counts_preds = self.gene_decoder(latent_preds)
            if padded:
                gene_targets = gene_targets.reshape(-1, self.cell_sentence_len, self.gene_decoder.gene_dim())
            else:
                gene_targets = gene_targets.reshape(1, -1, self.gene_decoder.gene_dim())

            decoder_per_set = self._compute_distribution_loss(pert_cell_counts_preds, gene_targets)
            decoder_loss = decoder_per_set.mean()
            self.log(self._train_expression_loss_key(), decoder_loss)
            total_loss = total_loss + self.decoder_loss_weight * decoder_loss

        # Log whether demos were used
        self.log("train/demos_used", float(demo_batch is not None), on_step=True, on_epoch=False)

        return total_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation with no demonstrations (tests generalization without ICL)."""
        # Validate WITHOUT demos to measure base performance
        if self.nb_loss:
            pred, nb_mean_flat, nb_dispersion_flat = self.forward(batch, return_nb_params=True, demo_batch=None)
        else:
            pred = self.forward(batch, demo_batch=None)

        target = batch["pert_cell_emb"]
        pred = pred.reshape(-1, self.cell_sentence_len, self.output_dim)
        target = target.reshape(-1, self.cell_sentence_len, self.output_dim)

        if self.nb_loss:
            nb_mean = nb_mean_flat.reshape(-1, self.cell_sentence_len, self.nb_target_dim)
            nb_dispersion = nb_dispersion_flat.reshape(-1, self.cell_sentence_len, self.nb_target_dim)
            nb_target = self._get_nb_target_tensor(batch, target, True)
            per_set_losses = self._compute_nb_nll_loss(nb_mean, nb_dispersion, nb_target)
        else:
            per_set_losses = self._compute_distribution_loss(pred, target)

        loss = torch.nanmean(per_set_losses)
        self.log(self._val_main_loss_key(), loss, sync_dist=True)

        if self.gene_decoder is not None and not self.nb_loss and "pert_cell_counts" in batch:
            gene_targets = batch["pert_cell_counts"].reshape(-1, self.cell_sentence_len, self.gene_decoder.gene_dim())
            decoder_pred = self.gene_decoder(pred)
            decoder_per_set = self._compute_distribution_loss(decoder_pred, gene_targets)
            val_exp_loss = decoder_per_set.mean()
            self.log(self._val_expression_loss_key(), val_exp_loss, sync_dist=True)

    # ------------------------------------------------------------------
    # Inference support
    # ------------------------------------------------------------------

    def set_inference_demos(self, demo_data: Optional[dict] = None) -> None:
        """
        Set demonstration data for inference.

        Call this before running predict/evaluation to inject demos.
        Pass None to clear demos (predict without ICL).

        Args:
            demo_data: Dict with 'demo_ctrl', 'demo_pert', 'demo_effect' tensors.
                       Each has shape [1, D, dim] (will be broadcast to batch size).
        """
        self._inference_demo_data = demo_data
        if demo_data is not None:
            logger.info(
                "Inference demos set: %d demo tokens",
                demo_data["demo_ctrl"].shape[1],
            )
        else:
            logger.info("Inference demos cleared (no ICL).")

    def predict_step(self, batch, batch_idx, **kwargs):
        """
        Predict step with optional ICL demonstrations.

        If set_inference_demos() was called, demos are injected.
        Otherwise falls back to demo cache (training) or no demos.
        """
        padded = kwargs.get("padded", True)
        demo_batch = getattr(self, "_inference_demo_data", None)

        # If no explicit inference demos, try demo cache
        if demo_batch is None and self._demo_cache_ready:
            if padded:
                B = batch["pert_emb"].shape[0] // self.cell_sentence_len
            else:
                B = 1
            demo_batch = self._sample_demo_batch(
                cell_types=batch.get("cell_type", []),
                pert_names=batch.get("pert_name", []),
                batch_size=B,
                device=batch["pert_emb"].device,
            )

        # Expand demo_batch to match batch size if needed
        if demo_batch is not None:
            if padded:
                B = batch["pert_emb"].shape[0] // self.cell_sentence_len
            else:
                B = 1
            for key in ("demo_ctrl", "demo_pert", "demo_effect"):
                if demo_batch[key].shape[0] == 1 and B > 1:
                    demo_batch[key] = demo_batch[key].expand(B, -1, -1)

        latent_output = self.forward(batch, padded=padded, demo_batch=demo_batch)
        output_dict = {
            "preds": latent_output,
            "pert_cell_emb": batch.get("pert_cell_emb", None),
            "pert_cell_counts": batch.get("pert_cell_counts", None),
            "pert_name": batch.get("pert_name", None),
            "celltype_name": batch.get("cell_type", None),
            "batch": batch.get("batch", None),
            "ctrl_cell_emb": batch.get("ctrl_cell_emb", None),
        }

        if self.gene_decoder is not None:
            pert_cell_counts_preds = self.gene_decoder(latent_output)
            output_dict["pert_cell_counts_preds"] = pert_cell_counts_preds

        return output_dict
