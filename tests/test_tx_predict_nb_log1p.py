import numpy as np
import pytest
import torch

from state._cli._tx._predict import (
    _nb_log1p_eval_tensor,
    _nb_real_eval_needs_log1p_transform,
    _to_deseq2_counts_np,
)


def test_nb_log1p_eval_tensor_mean_matches_log1p_mu():
    mu = torch.tensor([[0.0, 0.5, 2.0, 10.0]], dtype=torch.float32)
    theta = torch.tensor([[1.0, 2.0, 5.0, 10.0]], dtype=torch.float32)
    out = _nb_log1p_eval_tensor(mu, theta, mode="mean")
    expected = torch.log1p(mu)
    assert torch.allclose(out, expected)


def test_nb_log1p_eval_tensor_delta_is_nonnegative_and_below_mean():
    mu = torch.tensor([[0.1, 1.0, 5.0, 20.0]], dtype=torch.float32)
    theta = torch.tensor([[0.8, 1.5, 4.0, 8.0]], dtype=torch.float32)
    out_delta = _nb_log1p_eval_tensor(mu, theta, mode="delta")
    out_mean = _nb_log1p_eval_tensor(mu, theta, mode="mean")

    assert torch.all(out_delta >= 0.0)
    assert torch.all(out_delta <= out_mean + 1e-6)


def test_nb_log1p_eval_tensor_delta_stronger_with_lower_dispersion():
    mu = torch.tensor([[0.5, 2.0, 8.0]], dtype=torch.float32)
    theta_high = torch.tensor([[20.0, 20.0, 20.0]], dtype=torch.float32)
    theta_low = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32)

    out_high_theta = _nb_log1p_eval_tensor(mu, theta_high, mode="delta")
    out_low_theta = _nb_log1p_eval_tensor(mu, theta_low, mode="delta")

    assert torch.all(out_low_theta <= out_high_theta + 1e-6)


def test_nb_log1p_eval_tensor_mc_is_nonnegative_and_below_mean():
    torch.manual_seed(0)
    mu = torch.tensor([[0.2, 1.0, 4.0, 10.0]], dtype=torch.float32)
    theta = torch.tensor([[0.6, 1.2, 2.0, 4.0]], dtype=torch.float32)

    out_mc = _nb_log1p_eval_tensor(mu, theta, mode="mc", mc_samples=64)
    out_mean = _nb_log1p_eval_tensor(mu, theta, mode="mean")

    assert torch.all(out_mc >= 0.0)
    assert torch.all(out_mc <= out_mean + 1e-6)


def test_nb_log1p_eval_tensor_mc_reflects_dispersion_strength():
    torch.manual_seed(0)
    mu = torch.tensor([[0.5, 2.0, 8.0]], dtype=torch.float32)
    theta_high = torch.tensor([[50.0, 50.0, 50.0]], dtype=torch.float32)
    theta_low = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32)

    torch.manual_seed(0)
    out_high_theta = _nb_log1p_eval_tensor(mu, theta_high, mode="mc", mc_samples=1024)
    torch.manual_seed(0)
    out_low_theta = _nb_log1p_eval_tensor(mu, theta_low, mode="mc", mc_samples=1024)

    assert torch.all(out_low_theta <= out_high_theta + 1e-5)


def test_nb_log1p_eval_tensor_mc_requires_positive_samples():
    mu = torch.tensor([[1.0]], dtype=torch.float32)
    theta = torch.tensor([[1.0]], dtype=torch.float32)
    with pytest.raises(ValueError):
        _nb_log1p_eval_tensor(mu, theta, mode="mc", mc_samples=0)


def test_nb_real_eval_needs_log1p_transform_scale_logic():
    # Count-space real targets should be transformed.
    assert _nb_real_eval_needs_log1p_transform(resolved_exp_counts=True, resolved_is_log1p=True) is True
    assert _nb_real_eval_needs_log1p_transform(resolved_exp_counts=True, resolved_is_log1p=False) is True

    # Log1p-space real targets should not be transformed again.
    assert _nb_real_eval_needs_log1p_transform(resolved_exp_counts=False, resolved_is_log1p=True) is False

    # If data are not marked log1p and exp_counts is disabled, transform defensively.
    assert _nb_real_eval_needs_log1p_transform(resolved_exp_counts=False, resolved_is_log1p=False) is True


def test_to_deseq2_counts_np_handles_both_scales():
    raw = np.array([[0.1, 2.4, 3.6]], dtype=np.float32)
    log1p_vals = np.log1p(raw)

    out_from_raw = _to_deseq2_counts_np(raw, from_log1p=False)
    out_from_log1p = _to_deseq2_counts_np(log1p_vals, from_log1p=True)

    assert np.array_equal(out_from_raw, np.round(np.clip(raw, 0.0, None)))
    assert np.array_equal(out_from_log1p, np.round(np.clip(raw, 0.0, None)))
