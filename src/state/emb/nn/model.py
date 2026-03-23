import warnings

warnings.filterwarnings("ignore")

import logging
import torch.nn.functional as F
import torch
from omegaconf import OmegaConf
import lightning as L

import sys

sys.path.append("../../")
sys.path.append("../")

from torch import nn, Tensor
from torch.nn import BCEWithLogitsLoss


from torch.optim.lr_scheduler import ChainedScheduler, LinearLR, CosineAnnealingLR

from ..utils import (
    get_embedding_cfg,
    get_dataset_cfg,
)
from .loss import WassersteinLoss, KLDivergenceLoss, MMDLoss, TabularLoss
from .tokenizer import Tokenizer, TokenizerOutput, SentenceTokenizer, SkipBlock


class StateEmbeddingModel(L.LightningModule):
    def __init__(
        self,
        token_dim: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        output_dim: int,
        dropout: float = 0.0,
        warmup_steps: int = 0,
        compiled: bool = False,
        max_lr=4e-4,
        emb_cnt=145469,
        emb_size=5120,
        cfg=None,
        collater=None,
        tokenizer: Tokenizer = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        self.compiled = compiled
        self.model_type = "Transformer"
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.dropout = dropout
        self.max_lr = max_lr
        self.collater = collater

        # --- Tokenizer (owns encoder, transformer, decoder, CLS token, count encoding) ---
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            # Backward compat: create SentenceTokenizer from args
            self.tokenizer = SentenceTokenizer(
                token_dim=token_dim,
                d_model=d_model,
                nhead=nhead,
                d_hid=d_hid,
                nlayers=nlayers,
                output_dim=output_dim,
                dropout=dropout,
                compiled=compiled,
                cfg=cfg,
            )

        # --- Decoder head: predicts counts from cell_emb × gene_emb ---
        self.z_dim_rd = 1 if self.cfg.model.rda else 0
        self.z_dim_ds = 10 if self.cfg.model.get("dataset_correction", False) else 0
        self.z_dim = self.z_dim_rd + self.z_dim_ds

        bottleneck_dim = getattr(self.cfg.model, "decoder_bottleneck_dim", None)
        if bottleneck_dim is not None:
            # Project gene_emb and cell_emb to small bottleneck dim before concat
            self.gene_proj = nn.Linear(d_model, bottleneck_dim)
            self.cell_proj = nn.Linear(output_dim, bottleneck_dim)
            dec_input_dim = bottleneck_dim * 2 + self.z_dim
            self.binary_decoder = nn.Sequential(
                SkipBlock(dec_input_dim),
                nn.Linear(dec_input_dim, 1, bias=True),
            )
        else:
            self.gene_proj = None
            self.cell_proj = None
            self.binary_decoder = nn.Sequential(
                SkipBlock(output_dim + d_model + self.z_dim),
                SkipBlock(output_dim + d_model + self.z_dim),
                nn.Linear(output_dim + d_model + self.z_dim, 1, bias=True),
            )

        if compiled:
            self.binary_decoder = torch.compile(self.binary_decoder)

        # --- Dataset correction ---
        if getattr(self.cfg.model, "dataset_correction", False):
            self.dataset_embedder = nn.Linear(output_dim, self.z_dim_ds)
            num_dataset = get_dataset_cfg(self.cfg).num_datasets
            self.dataset_encoder = nn.Sequential(
                nn.Linear(output_dim, d_model),
                nn.SiLU(),
                nn.LayerNorm(d_model),
                nn.Dropout(0.1),
                nn.Linear(d_model, num_dataset),
            )
            self.dataset_loss = nn.CrossEntropyLoss()
        else:
            self.dataset_embedder = None

        # --- Loss criterion ---
        if self.cfg.loss.name == "cross_entropy":
            self.criterion = BCEWithLogitsLoss()
        elif self.cfg.loss.name == "mse":
            self.criterion = nn.MSELoss()
        elif self.cfg.loss.name == "wasserstein":
            self.criterion = WassersteinLoss()
        elif self.cfg.loss.name == "kl_divergence":
            self.criterion = KLDivergenceLoss(apply_normalization=self.cfg.loss.normalization)
        elif self.cfg.loss.name == "mmd":
            kernel = self.cfg.loss.get("kernel", "energy")
            self.criterion = MMDLoss(kernel=kernel)
        elif self.cfg.loss.name == "tabular":
            self.criterion = TabularLoss(shared=self.cfg.dataset.S)
        else:
            raise ValueError(f"Loss {self.cfg.loss.name} not supported")

        # --- Backward compat shims ---
        self.pe_embedding = None  # Set externally; delegates to self.tokenizer.pe_embedding
        self.protein_embeds = None
        self.step_ctr = 0
        self.true_top_genes = None
        self._last_val_de_check = 0
        self._last_val_perturbation_check = 0

    # --- Backward compat properties ---
    @property
    def encoder(self):
        return self.tokenizer.encoder

    @property
    def gene_embedding_layer(self):
        return self.tokenizer.gene_embedding_layer

    @property
    def cls_token(self):
        return self.tokenizer.cls_token

    @property
    def dataset_token(self):
        return getattr(self.tokenizer, "dataset_token", None)

    def on_save_checkpoint(self, checkpoint):
        try:
            if self.cfg is not None:
                checkpoint["cfg_yaml"] = OmegaConf.to_yaml(self.cfg)
        except Exception:
            pass

        try:
            if self.protein_embeds is not None:
                pe = self.protein_embeds
            else:
                from ..utils import get_embedding_cfg

                pe = torch.load(get_embedding_cfg(self.cfg).all_embeddings, map_location="cpu", weights_only=False)
            if isinstance(pe, dict):
                cpu_pe = {}
                for k, v in pe.items():
                    try:
                        cpu_pe[k] = v.detach().to("cpu") if hasattr(v, "detach") else torch.tensor(v, device="cpu")
                    except Exception:
                        cpu_pe[k] = v
                checkpoint["protein_embeds_dict"] = cpu_pe
        except Exception:
            pass

    def get_gene_embedding(self, genes):
        if self.protein_embeds is None:
            self.protein_embeds = torch.load(get_embedding_cfg(self.cfg).all_embeddings, weights_only=False)

        protein_embeds = [
            self.protein_embeds[x] if x in self.protein_embeds else torch.zeros(get_embedding_cfg(self.cfg).size)
            for x in genes
        ]
        protein_embeds = torch.stack(protein_embeds).to(self.device)
        if protein_embeds.sum() == 0:
            raise ValueError("No gene embeddings found")

        return self.gene_embedding_layer(protein_embeds)

    @staticmethod
    def resize_batch(cell_embeds, task_embeds, task_counts=None, sampled_rda=None, ds_emb=None):
        B, T = cell_embeds.size(0), task_embeds.size(0)
        A = task_embeds.unsqueeze(0).expand(B, -1, -1)
        C = cell_embeds.unsqueeze(1).expand(-1, T, -1)
        if sampled_rda is not None:
            reshaped_counts = sampled_rda.unsqueeze(1).expand(-1, T, -1)
            combine = torch.cat((A, C, reshaped_counts), dim=2)
        elif task_counts is not None:
            reshaped_counts = task_counts.unsqueeze(1).unsqueeze(2).expand(-1, T, -1)
            combine = torch.cat((A, C, reshaped_counts), dim=2)
        else:
            combine = torch.cat((A, C), dim=2)

        if ds_emb is not None:
            ds_emb = ds_emb.unsqueeze(1).expand(-1, T, -1)
            combine = torch.cat((combine, ds_emb), dim=2)

        return combine

    # --- Backward compat: _compute_embedding_for_batch delegates to tokenizer ---
    def _compute_embedding_for_batch(self, batch):
        out = self.tokenizer(batch)
        return out.task_gene_embs, out.task_counts, None, out.cell_embedding, out.dataset_emb

    def shared_step(self, batch, batch_idx):
        out = self.tokenizer(batch)

        X = out.task_gene_embs  # [B, n_task, d_model]
        Y = out.task_counts  # [B, n_task]
        embs = out.cell_embedding  # [B, output_dim]
        dataset_embs = out.dataset_emb  # [B, output_dim] or None

        # Bottleneck: project gene and cell embeddings to small dim before concat
        if self.gene_proj is not None:
            X_dec = self.gene_proj(X)  # [B, n_task, bottleneck_dim]
            z = self.cell_proj(embs).unsqueeze(1).expand(-1, X.shape[1], -1)
        else:
            X_dec = X
            z = embs.unsqueeze(1).expand(-1, X.shape[1], -1)  # CLS token

        if self.z_dim_rd == 1:
            mu = (
                torch.nan_to_num(
                    torch.nanmean(
                        Y.float().masked_fill(Y == 0, float("nan")),
                        dim=1,
                    ),
                    nan=0.0,
                )
                if self.cfg.model.rda
                else None
            )
            reshaped_counts = mu.unsqueeze(1).unsqueeze(2)
            reshaped_counts = reshaped_counts.expand(-1, X.shape[1], -1)
            combine = torch.cat((X_dec, z, reshaped_counts), dim=2)
        else:
            assert self.z_dim_rd == 0
            combine = torch.cat((X_dec, z), dim=2)

        if self.dataset_embedder is not None and dataset_embs is not None:
            ds_emb = self.dataset_embedder(dataset_embs)
            ds_emb = ds_emb.unsqueeze(1).expand(-1, X.shape[1], -1)
            combine = torch.cat((combine, ds_emb), dim=2)

        decs = self.binary_decoder(combine)

        target = Y
        if self.cfg.loss.name in ("mmd", "tabular"):
            downsample = self.cfg.model.num_downsample if self.training else 1
            loss = self.criterion(decs.squeeze(), target, downsample=downsample)
        else:
            loss = self.criterion(decs.squeeze(), target)

        if self.dataset_embedder is not None and dataset_embs is not None and out.dataset_nums is not None:
            dataset_pred = self.dataset_encoder(dataset_embs)
            dataset_labels = out.dataset_nums.to(self.device).long()
            dataset_loss = self.dataset_loss(dataset_pred, dataset_labels)
            if self.training:
                self.log("trainer/dataset_loss", dataset_loss)
                loss = loss + dataset_loss
            else:
                self.log("validation/dataset_loss", dataset_loss)

        return loss

    @torch.compile(disable=True)
    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("trainer/train_loss", loss)
        return loss

    @torch.compile(disable=True)
    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("validation/val_loss", loss)
        return loss

    def configure_optimizers(self):
        max_lr = self.max_lr
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=max_lr, weight_decay=self.cfg.optimizer.weight_decay)
        total_steps = self.trainer.estimated_stepping_batches

        lr_schedulers = [
            LinearLR(
                optimizer,
                start_factor=self.cfg.optimizer.start,
                end_factor=self.cfg.optimizer.end,
                total_iters=int(0.03 * total_steps),
            )
        ]
        lr_schedulers.append(CosineAnnealingLR(optimizer, eta_min=max_lr * 0.3, T_max=total_steps))
        scheduler = ChainedScheduler(lr_schedulers)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "train_loss", "interval": "step", "frequency": 1},
        }

    def update_config(self, new_cfg):
        """Update the model's config after loading from checkpoint."""
        self.cfg = new_cfg
