import torch

from state.tx.models.state_transition import StateTransitionPerturbationModel


def test_reduce_dispersion_set_median():
    nb_dispersion = torch.tensor(
        [
            [
                [1.0, 10.0],
                [3.0, 30.0],
                [5.0, 50.0],
            ]
        ]
    )
    reduced = StateTransitionPerturbationModel._reduce_dispersion_for_inference(nb_dispersion, mode="set_median")
    expected = torch.tensor(
        [
            [
                [3.0, 30.0],
                [3.0, 30.0],
                [3.0, 30.0],
            ]
        ]
    )
    assert torch.allclose(reduced, expected)


def test_reduce_dispersion_per_cell_identity():
    nb_dispersion = torch.rand(2, 4, 8)
    reduced = StateTransitionPerturbationModel._reduce_dispersion_for_inference(nb_dispersion, mode="per_cell")
    assert torch.allclose(reduced, nb_dispersion)


def test_sample_nb_counts_shape_and_nonnegative():
    model = object.__new__(StateTransitionPerturbationModel)
    model.nb_eps = 1e-8

    torch.manual_seed(0)
    mu = torch.full((2, 5, 7), 4.0)
    theta = torch.full((2, 5, 7), 2.5)

    samples = model._sample_nb_counts(mu, theta)
    assert samples.shape == mu.shape
    assert torch.all(samples >= 0.0)


def test_sample_nb_counts_matches_requested_mean():
    model = object.__new__(StateTransitionPerturbationModel)
    model.nb_eps = 1e-8

    torch.manual_seed(0)
    mu = torch.full((100_000,), 1.7)
    theta = torch.full((100_000,), 0.8)

    samples = model._sample_nb_counts(mu, theta)
    sample_mean = float(samples.mean())
    assert abs(sample_mean - 1.7) < 0.1


def test_compute_library_sizes_from_control_modes():
    model = object.__new__(StateTransitionPerturbationModel)
    model.nb_eps = 1e-8

    ctrl_counts = torch.tensor(
        [
            [
                [1.0, 2.0, 0.0],
                [3.0, 1.0, 0.0],
                [2.0, 2.0, 2.0],
            ]
        ]
    )

    per_cell = model._compute_library_sizes_from_control(ctrl_counts, mode="per_cell")
    set_median = model._compute_library_sizes_from_control(ctrl_counts, mode="set_median")

    expected_per_cell = torch.tensor([[[3.0], [4.0], [6.0]]])
    expected_set_median = torch.tensor([[[4.0]]])

    assert torch.allclose(per_cell, expected_per_cell)
    assert torch.allclose(set_median, expected_set_median)


def test_to_count_space_auto_keeps_continuous_transformed_log1p_values():
    model = object.__new__(StateTransitionPerturbationModel)
    model.nb_count_round_mode = "auto"

    x = torch.log1p(torch.tensor([[0.0, 2.2, 0.4]], dtype=torch.float32))
    out = model._to_count_space(x)
    expected = torch.tensor([[0.0, 2.2, 0.4]], dtype=torch.float32)
    assert torch.allclose(out, expected, atol=1e-6)


def test_to_count_space_always_rounds_transformed_log1p_values():
    model = object.__new__(StateTransitionPerturbationModel)
    model.nb_count_round_mode = "always"

    x = torch.log1p(torch.tensor([[0.0, 2.2, 0.4]], dtype=torch.float32))
    out = model._to_count_space(x)
    expected = torch.tensor([[0.0, 2.0, 0.0]], dtype=torch.float32)
    assert torch.allclose(out, expected, atol=1e-6)


def test_blend_nb_library_sizes_interpolates_between_modes():
    set_median = torch.tensor([[[4.0]]])
    per_cell = torch.tensor([[[3.0], [4.0], [6.0]]])

    out_alpha0 = StateTransitionPerturbationModel._blend_nb_library_sizes(set_median, per_cell, alpha=0.0)
    out_alpha05 = StateTransitionPerturbationModel._blend_nb_library_sizes(set_median, per_cell, alpha=0.5)
    out_alpha1 = StateTransitionPerturbationModel._blend_nb_library_sizes(set_median, per_cell, alpha=1.0)

    assert torch.allclose(out_alpha0, torch.tensor([[[4.0], [4.0], [4.0]]]))
    assert torch.allclose(out_alpha05, torch.tensor([[[3.5], [4.0], [5.0]]]))
    assert torch.allclose(out_alpha1, per_cell)


def test_rescale_nb_mean_between_library_modes():
    nb_mean = torch.tensor(
        [
            [
                [2.0, 4.0],
                [1.0, 3.0],
            ]
        ]
    )
    source_lib = torch.tensor([[[10.0]]])
    target_lib = torch.tensor([[[8.0], [12.0]]])

    out = StateTransitionPerturbationModel._rescale_nb_mean_between_library_modes(
        nb_mean,
        source_library_sizes=source_lib,
        target_library_sizes=target_lib,
        eps=1e-8,
    )
    expected = torch.tensor(
        [
            [
                [1.6, 3.2],
                [1.2, 3.6],
            ]
        ]
    )
    assert torch.allclose(out, expected)


def test_resolve_nb_embed_loss_weight_default_is_zero():
    weight, used_default = StateTransitionPerturbationModel._resolve_nb_embed_loss_weight(
        embed_key="X_state",
        kwargs={},
    )
    assert used_default is True
    assert weight == 0.0


def test_resolve_nb_embed_loss_weight_explicit_override():
    weight, used_default = StateTransitionPerturbationModel._resolve_nb_embed_loss_weight(
        embed_key="X_state",
        kwargs={"nb_embed_loss_weight": 0.25},
    )
    assert used_default is False
    assert weight == 0.25


def test_sparsemax_produces_valid_simplex_with_zeros():
    logits = torch.tensor([[3.0, 1.0, -2.0], [0.1, 0.1, 0.1]], dtype=torch.float32)
    out = StateTransitionPerturbationModel._sparsemax(logits, dim=-1)
    assert torch.all(out >= 0.0)
    assert torch.allclose(out.sum(dim=-1), torch.ones(out.shape[0]), atol=1e-6)
    # First row should be sparse (at least one exact zero).
    assert torch.any(out[0] == 0.0)


def test_apply_nb_scale_activation_sparsemax_branch():
    model = object.__new__(StateTransitionPerturbationModel)
    model.nb_px_scale_activation = "sparsemax"
    logits = torch.tensor([[2.0, 0.0, -1.0]], dtype=torch.float32)
    out = model._apply_nb_scale_activation(logits)
    assert torch.all(out >= 0.0)
    assert torch.allclose(out.sum(dim=-1), torch.ones(1), atol=1e-6)
    assert torch.any(out == 0.0)


def test_compute_nb_log1p_mse_per_set_zero_when_matching_counts():
    model = object.__new__(StateTransitionPerturbationModel)
    model.nb_eps = 1e-8

    target_counts = torch.tensor(
        [
            [[0.0, 1.0, 3.0], [2.0, 0.0, 4.0]],
            [[5.0, 2.0, 0.0], [1.0, 1.0, 1.0]],
        ]
    )
    nb_mean = target_counts.clone()

    per_set = model._compute_nb_log1p_mse_per_set(nb_mean, target_counts)
    assert per_set.shape == torch.Size([2])
    assert torch.allclose(per_set, torch.zeros_like(per_set), atol=1e-7)


def test_compute_nb_log1p_mse_per_set_increases_with_mismatch():
    model = object.__new__(StateTransitionPerturbationModel)
    model.nb_eps = 1e-8

    target_counts = torch.tensor(
        [
            [[0.0, 1.0, 3.0], [2.0, 0.0, 4.0]],
        ]
    )
    nb_mean_small_error = target_counts + 0.1
    nb_mean_large_error = target_counts + 2.0

    loss_small = model._compute_nb_log1p_mse_per_set(nb_mean_small_error, target_counts).item()
    loss_large = model._compute_nb_log1p_mse_per_set(nb_mean_large_error, target_counts).item()
    assert loss_large > loss_small


def test_get_nb_target_library_sizes_for_inference_uses_pert_counts_per_cell():
    model = object.__new__(StateTransitionPerturbationModel)
    model.nb_eps = 1e-8
    model.cell_sentence_len = 3

    batch = {
        "pert_cell_counts": torch.tensor(
            [
                [1.0, 2.0, 0.0],
                [3.0, 1.0, 0.0],
                [2.0, 2.0, 2.0],
            ]
        )
    }

    out = model._get_nb_target_library_sizes_for_inference(batch, padded=False)
    expected = torch.tensor([[[3.0], [4.0], [6.0]]])
    assert torch.allclose(out, expected)


def test_get_nb_target_library_sizes_for_inference_requires_targets():
    model = object.__new__(StateTransitionPerturbationModel)
    model.nb_eps = 1e-8
    model.cell_sentence_len = 2
    try:
        _ = model._get_nb_target_library_sizes_for_inference({}, padded=False)
    except RuntimeError as exc:
        assert "target_oracle" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError when pert_cell_counts is missing.")


def test_compute_nb_library_sizes_from_mean_sums_last_dim():
    nb_mean = torch.tensor(
        [
            [
                [1.0, 2.0, 0.0],
                [3.0, 1.0, 0.0],
            ]
        ]
    )
    libs = StateTransitionPerturbationModel._compute_nb_library_sizes_from_mean(nb_mean)
    expected = torch.tensor([[[3.0], [4.0]]])
    assert torch.allclose(libs, expected)


def test_compute_nb_library_mse_per_set_zero_for_matching_libraries():
    model = object.__new__(StateTransitionPerturbationModel)
    model.nb_eps = 1e-8

    target_counts = torch.tensor(
        [
            [[2.0, 1.0], [3.0, 0.0]],
            [[1.0, 1.0], [1.0, 1.0]],
        ]
    )
    nb_mean = target_counts.clone()
    per_set = model._compute_nb_library_mse_per_set(nb_mean, target_counts)
    assert per_set.shape == torch.Size([2])
    assert torch.allclose(per_set, torch.zeros_like(per_set), atol=1e-7)
