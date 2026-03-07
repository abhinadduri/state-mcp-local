import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
import typing as tp

from .utils import get_loss_fn

logger = logging.getLogger(__name__)


class LatentToGeneDecoder(nn.Module):
    """
    A decoder module to transform latent embeddings back to gene expression space.

    This takes concat([cell embedding]) as the input, and predicts
    counts over all genes as output.

    This decoder is trained separately from the main perturbation model.

    Args:
        latent_dim: Dimension of latent space
        gene_dim: Dimension of gene space (number of HVGs)
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout rate
    """

    def __init__(
        self,
        latent_dim: int,
        gene_dim: int,
        hidden_dims: List[int] = [512, 1024],
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        input_dim = latent_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        # Final output layer
        layers.append(nn.Linear(input_dim, gene_dim))
        # Make sure outputs are non-negative
        layers.append(nn.ReLU())

        self.decoder = nn.Sequential(*layers)

    def gene_dim(self):
        # return the output dimension of the last layer
        for module in reversed(self.decoder):
            if isinstance(module, nn.Linear):
                return module.out_features
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder.

        Args:
            x: Latent embeddings of shape [batch_size, latent_dim]

        Returns:
            Gene expression predictions of shape [batch_size, gene_dim]
        """
        return self.decoder(x)


class PretrainedBinaryDecoder(nn.Module):
    """
    Decoder that uses the pretrained SE binary decoder to map cell embeddings to gene expression.

    The SE binary decoder was trained on 100M+ cells and knows rich gene-cell relationships.
    This decoder feeds TX model outputs (predicted cell embeddings) through the pretrained
    binary decoder to predict per-gene expression.

    Input: [B, latent_dim] where latent_dim = cell_emb_dim(2048) + ds_emb_dim(10) = 2058
    Output: [B, gene_dim] gene expression predictions

    Args:
        latent_dim: Input dimension (should be 2058 for X_state)
        gene_dim: Number of output genes
        gene_names: List of gene names matching the gene_dim
        se_checkpoint: Path to SE model checkpoint
        se_config: Path to SE model config
        cell_batch_size: Batch size for cell processing (to avoid OOM with large gene sets)
        freeze_binary_decoder: Whether to freeze the binary decoder weights
        read_depth: Initial read-depth scalar for RDA
    """

    def __init__(
        self,
        latent_dim: int,
        gene_dim: int,
        gene_names: Optional[List[str]] = None,
        se_checkpoint: str = "/home/aadduri/SE-600M/se600m_epoch16.ckpt",
        se_config: str = "/home/aadduri/SE-600M/config.yaml",
        cell_batch_size: int = 16,
        freeze_binary_decoder: bool = True,
        read_depth: float = 4.0,
    ):
        super().__init__()
        from omegaconf import OmegaConf
        from pathlib import Path

        self._gene_dim = gene_dim
        self._cell_batch_size = cell_batch_size
        self._gene_names = gene_names or []
        self._cell_emb_dim = 2048  # SE output_dim
        self._ds_emb_dim = latent_dim - self._cell_emb_dim  # typically 10

        # Learnable read-depth scalar
        self.read_depth = nn.Parameter(torch.tensor(read_depth), requires_grad=True)

        # Load SE model weights from safetensors (avoids pickle/vci import issues)
        cfg = OmegaConf.load(se_config)
        se_dir = Path(se_checkpoint).parent
        safetensors_path = se_dir / "model.safetensors"

        from ...emb.nn.model import StateEmbeddingModel

        se_model = StateEmbeddingModel(
            token_dim=cfg.tokenizer.token_dim,
            d_model=cfg.model.emsize,
            nhead=cfg.model.nhead,
            d_hid=cfg.model.d_hid,
            nlayers=cfg.model.nlayers,
            output_dim=cfg.model.output_dim,
            dropout=0.0,
            cfg=cfg,
        )

        if safetensors_path.exists():
            from safetensors.torch import load_file
            state_dict = load_file(str(safetensors_path))
            se_model.load_state_dict(state_dict, strict=False)
        else:
            # Fallback to .ckpt with legacy module shims
            import sys, types
            for m in ("vci", "vci.nn", "vci.nn.model", "vci.nn.loss",
                       "vci.nn.flash_transformer", "vci.train", "vci.train.trainer", "vci.utils"):
                if m not in sys.modules:
                    sys.modules[m] = types.ModuleType(m)
            ckpt = torch.load(se_checkpoint, map_location="cpu", weights_only=False)
            se_model.load_state_dict(ckpt["state_dict"], strict=False)

        # Extract and register binary decoder as a submodule
        self.binary_decoder = se_model.binary_decoder
        if freeze_binary_decoder:
            for p in self.binary_decoder.parameters():
                p.requires_grad = False

        # Extract gene_embedding_layer (the encoder: Linear(5120, 2048) + LayerNorm + SiLU)
        self.gene_embedding_layer = se_model.gene_embedding_layer
        for p in self.gene_embedding_layer.parameters():
            p.requires_grad = False

        # Load protein embeddings
        pe_path = se_dir / "protein_embeddings.pt"
        if not pe_path.exists():
            try:
                pe_path_cfg = cfg.embeddings[cfg.embeddings.current].all_embeddings
                if Path(pe_path_cfg).exists():
                    pe_path = Path(pe_path_cfg)
            except Exception:
                pass
        protein_embeds = torch.load(str(pe_path), weights_only=False)

        # Build gene embedding matrix for the requested genes
        self._build_gene_embeddings(protein_embeds, se_model)

        # Don't keep the full SE model in memory
        del se_model

        logger.info(
            "PretrainedBinaryDecoder: gene_dim=%d, cell_batch_size=%d, "
            "freeze=%s, matched_genes=%d/%d",
            gene_dim, cell_batch_size, freeze_binary_decoder,
            self._n_matched, len(self._gene_names),
        )

    def _build_gene_embeddings(self, protein_embeds, se_model):
        """Pre-compute gene embeddings for all requested genes."""
        embed_size = next(iter(protein_embeds.values())).shape[-1]  # 5120

        raw_embeds = []
        present = []
        for g in self._gene_names:
            if g in protein_embeds:
                raw_embeds.append(protein_embeds[g])
                present.append(True)
            else:
                raw_embeds.append(torch.zeros(embed_size))
                present.append(False)

        self._n_matched = sum(present)

        if len(raw_embeds) > 0:
            raw_tensor = torch.stack(raw_embeds)
            with torch.no_grad():
                gene_embeds = se_model.gene_embedding_layer(raw_tensor)  # [G, d_model]
        else:
            gene_embeds = torch.zeros(0, 2048)

        # Register as buffer (non-trainable, moves with device)
        self.register_buffer("_gene_embeds", gene_embeds)

    def gene_dim(self):
        return self._gene_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: cell embeddings -> gene expression via pretrained binary decoder.

        Args:
            x: [B, latent_dim] or [B, S, latent_dim]
        Returns:
            [B, gene_dim] or [B, S, gene_dim]
        """
        orig_shape = x.shape
        if x.dim() == 3:
            B, S, D = x.shape
            x = x.reshape(B * S, D)
        elif x.dim() == 2:
            pass
        else:
            raise ValueError(f"Expected 2D or 3D input, got {x.dim()}D")

        n_cells = x.shape[0]

        # Split cell embedding and dataset embedding
        cell_embs = x[:, :self._cell_emb_dim]
        ds_emb = x[:, self._cell_emb_dim:] if self._ds_emb_dim > 0 else None

        gene_embeds = self._gene_embeds  # [G, d_model]

        # Process in batches with no_grad (binary decoder is frozen, no backprop needed)
        outputs = []
        with torch.no_grad():
            for i in range(0, n_cells, self._cell_batch_size):
                end = min(i + self._cell_batch_size, n_cells)
                cell_batch = cell_embs[i:end]  # [b, 2048]
                bs = cell_batch.shape[0]
                n_genes = gene_embeds.shape[0]

                rda = self.read_depth.expand(bs)

                ds_batch = ds_emb[i:end] if ds_emb is not None else None

                # Build [b, G, d_model+output_dim+z_dim] pairwise features
                A = gene_embeds.unsqueeze(0).expand(bs, -1, -1)  # [b, G, d_model]
                B_cell = cell_batch.unsqueeze(1).expand(-1, n_genes, -1)  # [b, G, output_dim]

                # RDA
                rda_expanded = rda.unsqueeze(1).unsqueeze(2).expand(-1, n_genes, 1)  # [b, G, 1]
                combined = torch.cat([A, B_cell, rda_expanded], dim=2)

                # Dataset embedding
                if ds_batch is not None:
                    ds_expanded = ds_batch.unsqueeze(1).expand(-1, n_genes, -1)
                    combined = torch.cat([combined, ds_expanded], dim=2)

                # Run binary decoder in bf16
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=cell_batch.is_cuda):
                    logits = self.binary_decoder(combined).squeeze(-1)  # [b, G]

                outputs.append(logits)
                del combined  # free memory immediately

        result = torch.cat(outputs, dim=0)  # [n_cells, G]

        # Apply ReLU for non-negative outputs
        result = torch.relu(result)

        # Match input dtype (important for bf16-mixed precision compatibility)
        result = result.to(x.dtype)

        # Reshape back if input was 3D
        if len(orig_shape) == 3:
            result = result.reshape(orig_shape[0], orig_shape[1], -1)

        return result


class PerturbationModel(ABC, LightningModule):
    """
    Base class for perturbation models that can operate on either raw counts or embeddings.

    Args:
        input_dim: Dimension of input features (genes or embeddings)
        hidden_dim: Hidden dimension for neural network layers
        output_dim: Dimension of output (gene space or embedding space)
        pert_dim: Dimension of perturbation embeddings
        dropout: Dropout rate
        lr: Learning rate for optimizer
        loss_fn: Loss function ('mse' or custom nn.Module)
        output_space: 'gene', 'all', or 'embedding'
    """

    @staticmethod
    def _sanitize_decoder_cfg(decoder_cfg: dict | None) -> dict | None:
        if decoder_cfg is None:
            return None
        sanitized_cfg = dict(decoder_cfg)
        if "residual_decoder" in sanitized_cfg:
            sanitized_cfg.pop("residual_decoder")
            logger.warning("decoder_cfg.residual_decoder is deprecated and will be ignored.")
        return sanitized_cfg

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        pert_dim: int,
        batch_dim: int = None,
        dropout: float = 0.1,
        lr: float = 3e-4,
        loss_fn: nn.Module = nn.MSELoss(),
        control_pert: str = "non-targeting",
        embed_key: Optional[str] = None,
        output_space: str = "gene",
        gene_names: Optional[List[str]] = None,
        batch_size: int = 64,
        gene_dim: int = 5000,
        hvg_dim: int = 2001,
        decoder_cfg: dict | None = None,
        **kwargs,
    ):
        super().__init__()
        if "residual_decoder" in kwargs:
            kwargs = dict(kwargs)
            kwargs.pop("residual_decoder")
            logger.warning("model.kwargs.residual_decoder is deprecated and will be ignored.")
        self.decoder_cfg = self._sanitize_decoder_cfg(decoder_cfg)
        self.save_hyperparameters()
        self.gene_decoder_bool = kwargs.get("gene_decoder_bool", True)

        # Core architecture settings
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.pert_dim = pert_dim
        self.batch_dim = batch_dim
        self.gene_dim = gene_dim
        self.hvg_dim = hvg_dim

        if kwargs.get("batch_encoder", False):
            self.batch_dim = batch_dim
        else:
            self.batch_dim = None

        self.embed_key = embed_key
        self.output_space = output_space
        if self.output_space not in {"embedding", "gene", "all"}:
            raise ValueError(
                f"Unsupported output_space '{self.output_space}'. Expected one of 'embedding', 'gene', or 'all'."
            )
        self.batch_size = batch_size
        self.control_pert = control_pert

        # Training settings
        self.gene_names = gene_names  # store the gene names that this model output for gene expression space
        self.dropout = dropout
        self.lr = lr
        self.loss_fn = get_loss_fn(loss_fn)

        if self.output_space == "embedding":
            self.gene_decoder_bool = False
            self.decoder_cfg = None
            # keep hyperparameters metadata consistent with the actual model state
            try:
                if hasattr(self, "hparams"):
                    self.hparams["gene_decoder_bool"] = False  # type: ignore[index]
                    self.hparams["decoder_cfg"] = None  # type: ignore[index]
            except Exception:
                pass

        self._build_decoder()

    def transfer_batch_to_device(self, batch, device, dataloader_idx: int):
        return {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

    @abstractmethod
    def _build_networks(self):
        """Build the core neural network components."""
        pass

    def _build_decoder(self):
        """Create self.gene_decoder from self.decoder_cfg (or leave None)."""
        self.decoder_cfg = self._sanitize_decoder_cfg(self.decoder_cfg)
        if self.gene_decoder_bool == False:
            self.gene_decoder = None
            return
        if self.decoder_cfg is None:
            self.gene_decoder = None
            return

        cfg = dict(self.decoder_cfg)
        use_pretrained = cfg.pop("pretrained_binary_decoder", False)
        if use_pretrained:
            pretrained_kwargs = cfg.pop("pretrained_kwargs", {})
            self.gene_decoder = PretrainedBinaryDecoder(
                latent_dim=cfg["latent_dim"],
                gene_dim=cfg["gene_dim"],
                gene_names=self.gene_names,
                **pretrained_kwargs,
            )
        else:
            cfg.pop("pretrained_kwargs", None)
            self.gene_decoder = LatentToGeneDecoder(**cfg)

    def _main_loss_is_expression(self) -> bool:
        """
        Determine whether the primary train/val loss is in expression/count space.
        """
        if self.output_space == "embedding":
            return False
        return self.embed_key in {"X_hvg", None}

    def _train_main_loss_key(self) -> str:
        return "train/expression_loss" if self._main_loss_is_expression() else "train/embedding_loss"

    def _val_main_loss_key(self) -> str:
        return "val/expression_loss" if self._main_loss_is_expression() else "val/embedding_loss"

    @staticmethod
    def _train_expression_loss_key() -> str:
        return "train/expression_loss"

    @staticmethod
    def _val_expression_loss_key() -> str:
        return "val/expression_loss"

    def on_load_checkpoint(self, checkpoint: dict[str, tp.Any]) -> None:
        """
        Lightning calls this *before* the checkpoint's state_dict is loaded.
        Re-create the decoder using the exact hyper-parameters saved in the ckpt,
        so that parameter shapes match and load_state_dict succeeds.
        """
        # Check if decoder_cfg was already set externally (e.g., by training script for output_space mismatch)
        decoder_already_configured = (
            hasattr(self, "_decoder_externally_configured") and self._decoder_externally_configured
        )

        if self.gene_decoder_bool == False:
            self.gene_decoder = None
            return

        if decoder_already_configured:
            logger.info("Decoder was already configured externally, skipping checkpoint decoder configuration")
            return

        checkpoint_hparams = checkpoint.get("hyper_parameters", {})
        if "decoder_cfg" in checkpoint_hparams:
            self.decoder_cfg = self._sanitize_decoder_cfg(checkpoint_hparams["decoder_cfg"])
        elif self.decoder_cfg is None:
            raise ValueError(
                "Checkpoint is missing hyper_parameters.decoder_cfg and no decoder_cfg was provided at init. "
                "Decoder configuration is required."
            )

        self.decoder_cfg = self._sanitize_decoder_cfg(self.decoder_cfg)
        self.gene_decoder = LatentToGeneDecoder(**self.decoder_cfg)
        logger.info(f"Loaded decoder from decoder_cfg: {self.decoder_cfg}")

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step logic for both main model and decoder."""
        # Get model predictions (in latent space)
        pred = self(batch)

        # Compute main model loss
        main_loss = self.loss_fn(pred, batch["pert_cell_emb"])
        self.log(self._train_main_loss_key(), main_loss)

        # Process decoder if available
        decoder_loss = None
        if self.gene_decoder is not None and "pert_cell_counts" in batch:
            # Train decoder to map latent predictions to gene space
            with torch.no_grad():
                latent_preds = pred.detach()  # Detach to prevent gradient flow back to main model

            pert_cell_counts_preds = self.gene_decoder(latent_preds)
            gene_targets = batch["pert_cell_counts"]
            decoder_loss = self.loss_fn(pert_cell_counts_preds, gene_targets)

            # Log decoder loss
            self.log(self._train_expression_loss_key(), decoder_loss)

            total_loss = main_loss + decoder_loss
        else:
            total_loss = main_loss

        return total_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step logic."""
        pred = self(batch)
        loss = self.loss_fn(pred, batch["pert_cell_emb"])

        # TODO: remove unused
        # is_control = self.control_pert in batch["pert_name"]
        self.log(self._val_main_loss_key(), loss)

        return {"loss": loss, "predictions": pred}

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        latent_output = self(batch)

        output_dict = {
            "preds": latent_output,  # The distribution's sample
            "pert_cell_emb": batch.get("pert_cell_emb", None),  # The target gene expression or embedding
            "pert_cell_counts": batch.get("pert_cell_counts", None),  # the true, raw gene expression
            "pert_name": batch.get("pert_name", None),
            "celltype_name": batch.get("cell_type", None),
            "batch": batch.get("batch", None),
            "ctrl_cell_emb": batch.get("ctrl_cell_emb", None),
        }

        if self.gene_decoder is not None:
            pert_cell_counts_preds = self.gene_decoder(latent_output)
            output_dict["pert_cell_counts_preds"] = pert_cell_counts_preds

    def predict_step(self, batch, batch_idx, **kwargs):
        """
        Typically used for final inference. We'll replicate old logic:
         returning 'preds', 'X', 'pert_name', etc.
        """
        latent_output = self.forward(batch)
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

    def decode_to_gene_space(self, latent_embeds: torch.Tensor, basal_expr: None) -> torch.Tensor:
        """
        Decode latent embeddings to gene expression space.

        Args:
            latent_embeds: Embeddings in latent space

        Returns:
            Gene expression predictions or None if decoder is not available
        """
        if self.gene_decoder is not None:
            pert_cell_counts_preds = self.gene_decoder(latent_embeds)
            if basal_expr is not None:
                # Add basal expression if provided
                pert_cell_counts_preds += basal_expr
            return pert_cell_counts_preds
        return None

    def configure_optimizers(self):
        """
        Configure a single optimizer for both the main model and the gene decoder.
        """
        # Use a single optimizer for all parameters
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
