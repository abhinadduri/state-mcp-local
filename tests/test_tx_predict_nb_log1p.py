import torch

from state._cli._tx._predict import _nb_log1p_eval_tensor


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
