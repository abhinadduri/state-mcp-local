import torch

from state.tx.callbacks import GradNormCallback, gradient_norm
from state.tx.models.state_transition import StateTransitionPerturbationModel
from state.tx.optim import MuonWithAuxAdamW


class _GradientNormModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[3.0, 4.0]], dtype=torch.float32))
        self.bias = torch.nn.Parameter(torch.tensor([12.0], dtype=torch.float32))
        self.logged: list[str] = []

    def parameters(self, recurse: bool = True):
        return super().parameters(recurse=recurse)

    def log(self, name, value, **kwargs) -> None:
        self.logged.append(name)


def _make_small_state_model(**extra_kwargs) -> StateTransitionPerturbationModel:
    return StateTransitionPerturbationModel(
        input_dim=16,
        hidden_dim=16,
        output_dim=16,
        pert_dim=4,
        gene_dim=32,
        embed_key="X_state",
        output_space="all",
        transformer_backbone_key="llama",
        transformer_backbone_kwargs={
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "max_position_embeddings": 16,
            "bidirectional_attention": True,
        },
        cell_set_len=8,
        loss="mse",
        **extra_kwargs,
    )


def test_gradient_norm_uses_fused_parameter_norms() -> None:
    model = _GradientNormModule()
    model.weight.grad = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
    model.bias.grad = torch.tensor([12.0], dtype=torch.float32)

    assert gradient_norm(model) == 13.0


def test_gradnorm_callback_respects_log_interval() -> None:
    model = _GradientNormModule()
    model.weight.grad = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
    callback = GradNormCallback(log_interval=2)

    callback.on_before_optimizer_step(trainer=None, pl_module=model, optimizer=None)
    assert model.logged == []

    callback.on_before_optimizer_step(trainer=None, pl_module=model, optimizer=None)
    assert model.logged == ["train/gradient_norm"]


def test_state_model_logs_residual_metrics_and_builds_muon_optimizer() -> None:
    model = _make_small_state_model(
        optimizer="muon",
        residual_monitor_interval=1,
        residual_monitor_max_tokens=4,
    )
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, MuonWithAuxAdamW)

    batch = {
        "pert_emb": torch.randn(16, 4),
        "ctrl_cell_emb": torch.randn(16, 16),
        "pert_cell_emb": torch.randn(16, 16),
    }

    logged_names: list[str] = []

    def _record_log(name, value, **kwargs) -> None:
        logged_names.append(name)

    model.log = _record_log  # type: ignore[method-assign]
    model.train()

    loss = model.training_step(batch, batch_idx=0, padded=True)
    loss.backward()
    optimizer.step()

    residual_metrics = {
        "train/residual_input_rms",
        "train/residual_output_rms",
        "train/residual_delta_rms",
        "train/residual_output_gain",
        "train/residual_delta_ratio",
    }
    assert residual_metrics.issubset(set(logged_names))
    assert isinstance(model._residual_metrics, dict)
