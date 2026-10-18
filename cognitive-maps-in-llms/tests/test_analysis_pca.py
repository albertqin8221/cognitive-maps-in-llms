from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from cognitive_maps_in_llms.analysis import (
    RunResults,
    compute_pca_projection,
    plot_pca_activity_space,
    plot_pca_layer_trajectories,
)

pytestmark = pytest.mark.codex


@pytest.fixture
def recordings(tmp_path):
    directory = tmp_path / "model" / "prompt" / "activation_and_kv_recording"
    directory.mkdir(parents=True)
    rng = np.random.default_rng(5)
    Z = rng.normal(size=(4, 5, 2, 4, 6)).astype(np.float32)
    # Layer-dependent translations must not survive per-layer centering.
    Z += np.arange(4, dtype=np.float32)[None, None, None, :, None] * 30
    np.save(directory / "Z_gstla.npy", Z)
    metadata = {
        "activation_axes": ["graph_structure", "seed", "token", "layer", "activation"],
        "n_trials": 5,
        "start_seed": 10000,
        "context_landmarks": {"C1": 1},
        "relation_patterns": [
            {"name": name}
            for name in (
                "A --fwd--> B --fwd--> C",
                "A --rev--> B --fwd--> C",
                "A --fwd--> B --rev--> C",
                "A --rev--> B --rev--> C",
            )
        ],
    }
    (directory / "recordings_metadata.json").write_text(json.dumps(metadata))
    (directory / "completed.json").write_text("{}")
    return RunResults(artifacts_root=tmp_path).load_activation_recordings(
        model_preset="model",
        prompt_preset="prompt",
    )


def test_projection_matches_svd_and_layer_centering(recordings):
    assert isinstance(recordings.Z_gstla, np.memmap)
    assert not recordings.Z_gstla.flags.writeable
    before = recordings.Z_gstla.copy()
    projection = compute_pca_projection(recordings, token="C1", reference_layer=2)
    X = before[:, :, 1, 2, :].reshape(20, 6).astype(np.float64)
    _, singular_values, V = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)
    np.testing.assert_allclose(
        projection.components.T @ projection.components,
        V[:3].T @ V[:3],
        atol=2e-5,
    )
    np.testing.assert_allclose(
        projection.explained_variance, singular_values[:3] ** 2 / 19, rtol=2e-5
    )
    Z = before[:, :, 1].astype(np.float64)
    centered = Z - Z.mean(axis=(0, 1), keepdims=True)
    expected = centered @ projection.components.T
    np.testing.assert_allclose(projection.Y_gslq, expected, atol=2e-5)
    np.testing.assert_allclose(projection.Y_glq, expected.mean(axis=1), atol=2e-5)
    np.testing.assert_array_equal(recordings.Z_gstla, before)
    by_index = compute_pca_projection(recordings, token=1, reference_layer=2)
    np.testing.assert_array_equal(projection.Y_gslq, by_index.Y_gslq)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"token": "missing"},
        {"token": -1},
        {"token": True},
        {"reference_layer": 4},
        {"n_components": 0},
        {"n_components": 7},
    ],
)
def test_invalid_selection(recordings, kwargs):
    selection = {"token": "C1", "reference_layer": 2, **kwargs}
    with pytest.raises(ValueError):
        compute_pca_projection(recordings, **selection)


@pytest.mark.parametrize("value", [0, np.nan])
def test_undefined_pca(recordings, value):
    from dataclasses import replace

    bad = replace(
        recordings, Z_gstla=np.full(recordings.Z_gstla.shape, value, dtype=np.float32)
    )
    with pytest.raises(ValueError, match="zero variance|non-finite"):
        compute_pca_projection(bad, token="C1", reference_layer=2)


def test_incomplete_recordings_are_rejected(recordings):
    from cognitive_maps_in_llms.analysis.pca import load_activation_recordings

    (recordings.directory / "completed.json").unlink()
    with pytest.raises(FileNotFoundError, match="completed.json"):
        load_activation_recordings(recordings.directory)


def test_axis_metadata_is_validated(recordings):
    from cognitive_maps_in_llms.analysis.pca import load_activation_recordings

    metadata = dict(recordings.metadata, activation_axes=["wrong"])
    (recordings.directory / "recordings_metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="activation_axes"):
        load_activation_recordings(recordings.directory)


def test_plot_views_reuse_projection(recordings, tmp_path):
    projection = compute_pca_projection(recordings, token="C1", reference_layer=2)
    fig, ax = plt.subplots()
    plots = []
    try:
        space = plot_pca_activity_space(
            projection, components=(3, 1), layer_range=(1, 4), ax=ax, legend=False
        )
        plots.append(space)
        assert space.ax is ax and space.projection is projection
        assert ax.get_xlabel() == "PC3"
        assert len(space.artists["seed_trajectories"]) == 20
        line = space.artists["mean_trajectories"][0]
        np.testing.assert_array_equal(line.get_xdata(), projection.Y_glq[0, 1:4, 2])
        np.testing.assert_array_equal(line.get_ydata(), projection.Y_glq[0, 1:4, 0])
        assert line.get_label() == "A → B → C"
        three = plot_pca_activity_space(
            projection, components=(1, 2, 3), show_seed_trajectories=False
        )
        plots.append(three)
        assert three.ax.name == "3d"
        x, y, z = three.artists["mean_trajectories"][0].get_data_3d()
        np.testing.assert_array_equal(np.array([x, y, z]).T, projection.Y_glq[0])
        layers = plot_pca_layer_trajectories(
            projection, components=(2,), layer_range=(1, 3)
        )
        plots.append(layers)
        assert layers.axes.shape == (1,)
        line = layers.artists["mean_trajectories"][0]
        np.testing.assert_array_equal(line.get_xdata(), [1, 2])
        np.testing.assert_array_equal(line.get_ydata(), projection.Y_glq[0, 1:3, 1])
        for i, plot in enumerate(plots):
            for extension in ("svg", "pdf"):
                path = tmp_path / f"plot_{i}.{extension}"
                plot.save(path)
                assert path.stat().st_size > 0
    finally:
        plt.close(fig)
        for plot in plots:
            plt.close(plot.fig)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"components": (0, 1)},
        {"components": (1, 4)},
        {"components": (1, 1)},
        {"layer_range": (2, 2)},
        {"layer_range": (-1, 3)},
    ],
)
def test_invalid_plot_selection(recordings, kwargs):
    projection = compute_pca_projection(recordings, token=1, reference_layer=2)
    with pytest.raises(ValueError):
        plot_pca_activity_space(projection, **kwargs)


def test_seed_averaged_fit_matches_graph_mean_svd(recordings):
    before = recordings.Z_gstla.copy()
    projection = compute_pca_projection(
        recordings, token="C1", reference_layer=2, average_over_seeds=True,
    )
    assert projection.average_over_seeds is True
    X = before[:, :, 1, 2].mean(axis=1, dtype=np.float64)
    _, singular_values, V = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)
    np.testing.assert_allclose(
        projection.components.T @ projection.components,
        V[:3].T @ V[:3], atol=5e-5,
    )
    np.testing.assert_allclose(
        projection.explained_variance, singular_values[:3] ** 2 / 3, rtol=5e-5,
    )
    np.testing.assert_allclose(
        projection.explained_variance_ratio,
        singular_values[:3] ** 2 / (singular_values ** 2).sum(), rtol=5e-5,
    )
    Z = before[:, :, 1].astype(np.float64)
    expected = (Z - Z.mean(axis=(0, 1), keepdims=True)) @ projection.components.T
    assert projection.Y_gslq.shape == (4, 5, 4, 3)
    np.testing.assert_allclose(projection.Y_gslq, expected, atol=2e-5)
    np.testing.assert_allclose(projection.Y_glq, expected.mean(axis=1), atol=2e-5)
    np.testing.assert_array_equal(recordings.Z_gstla, before)


def test_seed_variation_does_not_define_averaged_basis(recordings):
    from dataclasses import replace

    # Name substitutions dominate feature 1, but cancel in each graph mean.
    Z = np.zeros(recordings.Z_gstla.shape, dtype=np.float32)
    Z[..., 0] = np.arange(4)[:, None, None, None]
    Z[..., 1] = (100 * np.arange(-2, 3))[None, :, None, None]
    source = replace(recordings, Z_gstla=Z)
    pooled = compute_pca_projection(source, token=1, reference_layer=2, n_components=1)
    averaged = compute_pca_projection(
        source, token=1, reference_layer=2, n_components=1, average_over_seeds=True,
    )
    assert pooled.average_over_seeds is False
    np.testing.assert_allclose(abs(pooled.components[0, 1]), 1, atol=1e-6)
    np.testing.assert_allclose(abs(averaged.components[0, 0]), 1, atol=1e-6)
    Z[..., 0] = 0
    with pytest.raises(ValueError, match="zero variance"):
        compute_pca_projection(
            source, token=1, reference_layer=2, n_components=1, average_over_seeds=True,
        )


def test_seed_averaged_component_bound(recordings):
    with pytest.raises(ValueError, match="n_components"):
        compute_pca_projection(
            recordings, token=1, reference_layer=2, n_components=4,
            average_over_seeds=True,
        )
    default = compute_pca_projection(recordings, token=1, reference_layer=2)
    explicit = compute_pca_projection(
        recordings, token=1, reference_layer=2, average_over_seeds=False,
    )
    np.testing.assert_array_equal(default.Y_gslq, explicit.Y_gslq)
