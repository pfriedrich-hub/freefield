import numpy as np
import pytest
from unittest.mock import MagicMock

import freefield.visualizations as visualizations


@pytest.fixture
def mock_axes(monkeypatch):
    """Replace the matplotlib 3D axes with a mock."""
    ax = MagicMock()

    monkeypatch.setattr(
        visualizations.plt,
        "figure",
        MagicMock(return_value=MagicMock()),
    )
    monkeypatch.setattr(
        visualizations,
        "Axes3D",
        MagicMock(return_value=ax),
    )

    return ax


def test_plot_sources_creates_3d_axes(monkeypatch):
    """plot_sources should create a figure and 3D axes."""
    figure = MagicMock()
    mock_figure = MagicMock(return_value=figure)
    mock_axes3d = MagicMock()

    monkeypatch.setattr(
        visualizations.plt,
        "figure",
        mock_figure,
    )
    monkeypatch.setattr(
        visualizations,
        "Axes3D",
        mock_axes3d,
    )

    visualizations.plot_sources(
        azimuth=np.array([0]),
        elevation=np.array([0]),
    )

    mock_figure.assert_called_once_with()
    mock_axes3d.assert_called_once_with(figure)


@pytest.mark.parametrize(
    "azimuth,elevation,expected_x,expected_y,expected_z",
    [
        (
            np.array([0]),
            np.array([0]),
            np.array([-1.6]),
            np.array([0.0]),
            np.array([0.0]),
        ),
        (
            np.array([90]),
            np.array([0]),
            np.array([0.0]),
            np.array([-1.6]),
            np.array([0.0]),
        ),
        (
            np.array([0]),
            np.array([90]),
            np.array([0.0]),
            np.array([0.0]),
            np.array([1.6]),
        ),
    ],
)
def test_plot_sources_calculates_expected_coordinates(
    mock_axes,
    azimuth,
    elevation,
    expected_x,
    expected_y,
    expected_z,
):
    """Source positions should be converted to the expected Cartesian coordinates."""
    visualizations.plot_sources(
        azimuth=azimuth,
        elevation=elevation,
    )

    source_call = mock_axes.scatter.call_args_list[0]

    x, y, z = source_call.args

    np.testing.assert_allclose(x, expected_x, atol=1e-10)
    np.testing.assert_allclose(y, expected_y, atol=1e-10)
    np.testing.assert_allclose(z, expected_z, atol=1e-10)


def test_plot_sources_uses_custom_scalar_distance(mock_axes):
    """A scalar distance should be applied to all sources."""
    azimuth = np.array([0, 90])
    elevation = np.array([0, 0])
    distance = 2.0

    visualizations.plot_sources(
        azimuth=azimuth,
        elevation=elevation,
        distance=distance,
    )

    source_call = mock_axes.scatter.call_args_list[0]
    x, y, z = source_call.args

    np.testing.assert_allclose(
        x,
        np.array([-2.0, 0.0]),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        y,
        np.array([0.0, -2.0]),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        z,
        np.array([0.0, 0.0]),
        atol=1e-10,
    )


def test_plot_sources_accepts_distance_array(mock_axes):
    """Individual distances should be applied to the corresponding sources."""
    azimuth = np.array([0, 90])
    elevation = np.array([0, 0])
    distance = np.array([1.0, 2.0])

    visualizations.plot_sources(
        azimuth=azimuth,
        elevation=elevation,
        distance=distance,
    )

    source_call = mock_axes.scatter.call_args_list[0]
    x, y, z = source_call.args

    np.testing.assert_allclose(
        x,
        np.array([-1.0, 0.0]),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        y,
        np.array([0.0, -2.0]),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        z,
        np.array([0.0, 0.0]),
        atol=1e-10,
    )


def test_plot_sources_plots_all_sources(mock_axes):
    """All supplied source positions should be passed to scatter."""
    azimuth = np.array([-90, 0, 90])
    elevation = np.array([0, 0, 0])

    visualizations.plot_sources(
        azimuth=azimuth,
        elevation=elevation,
    )

    source_call = mock_axes.scatter.call_args_list[0]

    x, y, z = source_call.args

    assert len(x) == 3
    assert len(y) == 3
    assert len(z) == 3


def test_plot_sources_plots_sources_with_expected_style(mock_axes):
    """Sources should be plotted as blue points."""
    visualizations.plot_sources(
        azimuth=np.array([0]),
        elevation=np.array([0]),
    )

    source_call = mock_axes.scatter.call_args_list[0]

    assert source_call.kwargs == {
        "c": "b",
        "marker": ".",
    }


def test_plot_sources_plots_listener_at_origin(mock_axes):
    """The listener should be plotted as a red circle at the origin."""
    visualizations.plot_sources(
        azimuth=np.array([0]),
        elevation=np.array([0]),
    )

    assert mock_axes.scatter.call_count == 2

    listener_call = mock_axes.scatter.call_args_list[1]

    assert listener_call.args == (0, 0, 0)
    assert listener_call.kwargs == {
        "c": "r",
        "marker": "o",
    }


