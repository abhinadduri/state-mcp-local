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
