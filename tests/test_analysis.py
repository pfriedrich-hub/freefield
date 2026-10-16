import numpy as np
import pytest

import freefield

from freefield.analysis import double_to_single_pole, single_pole_to_polar, polar_to_single_pole, polar_to_cartesian, \
    eg, mad, mean_dir, rmse


@pytest.mark.parametrize(
    "azimuth_double, elevation_double, expected",
    [
        (0, 0, 0),
        (30, 0, 30),
        (-30, 0, -30),
        (45, 60, 63.43494882292201),
        (-45, 60, -63.43494882292201),
    ],
)
def test_double_to_single_pole(
    azimuth_double,
    elevation_double,
    expected,
):
    """Known double-pole coordinates are converted correctly."""
    result = double_to_single_pole(
        azimuth_double=azimuth_double,
        elevation_double=elevation_double,
    )

    assert result == pytest.approx(expected)


def test_double_to_single_pole_accepts_arrays():
    """The conversion can be applied element-wise to NumPy arrays."""
    azimuth_double = np.array([0, 30, -30])
    elevation_double = np.array([0, 0, 0])

    result = double_to_single_pole(
        azimuth_double=azimuth_double,
        elevation_double=elevation_double,
    )

    expected = np.array([0, 30, -30])
    np.testing.assert_allclose(result, expected)

@pytest.mark.parametrize("elevation", [-80, -45, 0, 45, 80])
def test_double_to_single_pole_zero_azimuth(elevation):
    result = double_to_single_pole(0, elevation)

    assert result == pytest.approx(0)

@pytest.mark.parametrize(
    "azimuth, elevation",
    [
        (15, 20),
        (30, 45),
        (60, 70),
    ],
)
def test_double_to_single_pole_is_symmetric_in_azimuth(
    azimuth,
    elevation,
):
    positive = double_to_single_pole(azimuth, elevation)
    negative = double_to_single_pole(-azimuth, elevation)

    assert negative == pytest.approx(-positive)

@pytest.mark.parametrize(
    "azimuth, elevation",
    [
        (15, 20),
        (30, 45),
        (-45, 60),
    ],
)
def test_double_to_single_pole_is_symmetric_in_elevation(
    azimuth,
    elevation,
):
    above = double_to_single_pole(azimuth, elevation)
    below = double_to_single_pole(azimuth, -elevation)

    assert below == pytest.approx(above)


@pytest.mark.parametrize(
    "azimuth, elevation, expected",
    [
        (0, 0, (0, -90)),
        (30, 0, (-30, -90)),
        (-30, 0, (30, -90)),
        (45, 60, (-45, -30)),
        (-45, 60, (45, -30)),
    ],
)
def test_single_pole_to_polar(
        azimuth,
        elevation,
        expected
):
    result = single_pole_to_polar(azimuth=azimuth, elevation=elevation)

    assert result == pytest.approx(expected)

def test_single_pole_to_polar_accepts_arrays():
    azimuth = np.array([0, 45, -45])
    elevation = np.array([0, 60, 60])

    expected_phi = np.array([0, -45, 45])
    expected_theta = np.array([-90, -30, -30])

    phi, theta = single_pole_to_polar(
        azimuth=azimuth,
        elevation=elevation,
    )

    np.testing.assert_allclose(phi, expected_phi)
    np.testing.assert_allclose(theta, expected_theta)

@pytest.mark.parametrize(
    "phi, theta, expected",
    [
        (0, 0, (0, 90)),
        (0, -90, (0, 0)),
        (30, 0, (-30, 90)),
        (-30, 0, (30, 90)),
        (0, 60, (0, 150))
    ]
)
def test_polar_to_single_pole(
        phi,
        theta,
        expected
):
    result = polar_to_single_pole(phi=phi, theta=theta)
    assert result == pytest.approx(expected, result)

def test_polar_to_single_pole_accepts_arrays():
    phi = np.array([0, 30, -30])
    theta = np.array([0, -90, 60])
    expected_azimuth = np.array([0, -30, 30])
    expected_elevation = np.array([90, 0, 150])

    azimuth, elevation = polar_to_single_pole(phi=phi, theta=theta)

    np.testing.assert_allclose(azimuth, expected_azimuth)
    np.testing.assert_allclose(elevation, expected_elevation)

@pytest.mark.parametrize(
    "phi, theta, expected",
    [
        # positive z-pole
        (0, 0, (0, 0, 1)),
        (90, 0, (0, 0, 1)),

        # equator-plane
        (0, 90, (1, 0, 0)),
        (90, 90, (0, 1, 0)),
        (180, 90, (-1, 0, 0)),
        (-90, 90, (0, -1, 0)),

        # negative z-pole
        (0, 180, (0, 0, -1)),

        # in-between axes
        (45, 90, (
            np.sqrt(0.5),
            np.sqrt(0.5),
            0,
        )),
    ],
)
def test_polar_to_cartesian_known_coordinates(phi, theta, expected):
    """Known polar coordinates are converted correctly."""
    result = polar_to_cartesian(phi=phi, theta=theta)

    np.testing.assert_allclose(
        result,
        expected,
        atol=1e-12,
    )


def test_polar_to_cartesian_accepts_arrays():
    """The conversion is applied element-wise to NumPy arrays."""
    phi = np.array([0, 90, 180, -90])
    theta = np.array([90, 90, 90, 90])

    x, y, z = polar_to_cartesian(
        phi=phi,
        theta=theta,
    )

    expected_x = np.array([1, 0, -1, 0])
    expected_y = np.array([0, 1, 0, -1])
    expected_z = np.array([0, 0, 0, 0])

    np.testing.assert_allclose(x, expected_x, atol=1e-12)
    np.testing.assert_allclose(y, expected_y, atol=1e-12)
    np.testing.assert_allclose(z, expected_z, atol=1e-12)


@pytest.mark.parametrize(
    "phi, theta",
    [
        (0, 0),
        (0, 90),
        (45, 45),
        (-45, 60),
        (90, 120),
        (180, 90),
    ],
)
def test_polar_to_cartesian_returns_unit_vector(phi, theta):
    """Every angular coordinate describes a point on the unit sphere."""
    x, y, z = polar_to_cartesian(
        phi=phi,
        theta=theta,
    )

    vector_length = np.sqrt(x**2 + y**2 + z**2)

    assert vector_length == pytest.approx(1)


def test_polar_to_cartesian_array_results_have_input_shape():
    """Array inputs produce coordinate arrays with the same shape."""
    phi = np.array([
        [0, 45],
        [90, 180],
    ])
    theta = np.array([
        [0, 45],
        [90, 180],
    ])

    x, y, z = polar_to_cartesian(
        phi=phi,
        theta=theta,
    )

    assert x.shape == phi.shape
    assert y.shape == phi.shape
    assert z.shape == phi.shape


@pytest.fixture
def localization_data():
    """
    Example localization data.

    Columns:
        0: trial or observation index
        1: speaker index
        2: perceived azimuth
        3: perceived elevation
    """
    return np.array(
        [
            [0, 1, 0, 0],
            [1, 1, 3, 4],
            [2, 1, 6, 8],
            [3, 2, 100, 100],
            [4, 2, 200, 200],
        ],
        dtype=float,
    )


@pytest.mark.parametrize(
    "speaker, expected",
    [
        (1, np.array([[3.0, 4.0]])),
        (2, np.array([[150.0, 150.0]])),
    ],
)
def test_mean_dir_averages_selected_speaker_only(
    localization_data,
    speaker,
    expected,
):
    """Only responses belonging to the selected speaker are averaged."""
    result = mean_dir(
        data=localization_data,
        speaker=speaker,
    )

    np.testing.assert_allclose(result, expected)


def test_mad_with_explicit_reference_direction(localization_data):
    """
    MAD is the mean Euclidean distance from the reference direction.

    For speaker 1 and reference direction (0, 0), the distances are:
        0, 5 and 10

    Their mean is 5.
    """
    result = mad(
        data=localization_data,
        speaker=1,
        ref_dir=np.array([0.0, 0.0]),
    )

    assert result == pytest.approx(5.0)


def test_mad_uses_mean_direction_by_default(localization_data):
    """
    When no reference direction is supplied, the mean direction is used.

    Speaker 1 responses:
        (0, 0), (3, 4), (6, 8)

    Mean direction:
        (3, 4)

    Distances from the mean:
        5, 0 and 5
    """
    expected = 10 / 3

    result = mad(
        data=localization_data,
        speaker=1,
    )

    assert result == pytest.approx(expected)


def test_mad_uses_selected_speaker_only(localization_data):
    """Responses belonging to other speakers do not affect the MAD."""
    data_without_other_speaker = localization_data[
        localization_data[:, 1] == 1
    ]

    result_with_all_speakers = mad(
        data=localization_data,
        speaker=1,
        ref_dir=np.array([0.0, 0.0]),
    )
    result_with_selected_speaker = mad(
        data=data_without_other_speaker,
        speaker=1,
        ref_dir=np.array([0.0, 0.0]),
    )

    assert result_with_all_speakers == pytest.approx(
        result_with_selected_speaker
    )


def test_rmse_with_explicit_reference_direction(localization_data):
    """
    RMSE is computed from the squared Euclidean distances.

    For speaker 1 and reference direction (0, 0), the distances are:
        0, 5 and 10

    RMSE:
        sqrt((0² + 5² + 10²) / 3)
    """
    expected = np.sqrt(125 / 3)

    result = rmse(
        data=localization_data,
        speaker=1,
        ref_dir=np.array([0.0, 0.0]),
    )

    assert result == pytest.approx(expected)


def test_rmse_uses_mean_direction_by_default(localization_data):
    """
    When no reference direction is supplied, the mean direction is used.

    The distances from the mean direction are:
        5, 0 and 5
    """
    expected = np.sqrt(50 / 3)

    result = rmse(
        data=localization_data,
        speaker=1,
    )

    assert result == pytest.approx(expected)


def test_rmse_uses_selected_speaker_only(localization_data):
    """Responses belonging to other speakers do not affect the RMSE."""
    data_without_other_speaker = localization_data[
        localization_data[:, 1] == 1
    ]

    result_with_all_speakers = rmse(
        data=localization_data,
        speaker=1,
        ref_dir=np.array([0.0, 0.0]),
    )
    result_with_selected_speaker = rmse(
        data=data_without_other_speaker,
        speaker=1,
        ref_dir=np.array([0.0, 0.0]),
    )

    assert result_with_all_speakers == pytest.approx(
        result_with_selected_speaker
    )


def test_rmse_is_not_smaller_than_mad(localization_data):
    """
    The root mean square distance cannot be smaller than the
    arithmetic mean of the same non-negative distances.
    """
    reference = np.array([0.0, 0.0])

    mad_result = mad(
        data=localization_data,
        speaker=1,
        ref_dir=reference,
    )
    rmse_result = rmse(
        data=localization_data,
        speaker=1,
        ref_dir=reference,
    )

    assert rmse_result >= mad_result


def test_eg_without_speaker_positions_returns_elevation_iqr():
    """
    Without physical speaker positions, EG returns the interquartile
    range of all perceived elevations.
    """
    data = np.array(
        [
            [0, 0, 0, 0],
            [1, 1, 0, 10],
            [2, 2, 0, 20],
            [3, 3, 0, 30],
        ],
        dtype=float,
    )

    result = eg(data=data)

    # 75th percentile: 22.5
    # 25th percentile: 7.5
    assert result == pytest.approx(15.0)


def test_eg_returns_one_for_perfect_localization():
    """
    EG is 1 when perceived and physical elevations are identical.
    """
    data = np.array(
        [
            [0, 0, 0, 0],
            [1, 1, 0, 10],
            [2, 2, 0, 20],
            [3, 3, 0, 30],
        ],
        dtype=float,
    )

    speaker_positions = np.array(
        [
            [0, 0],
            [0, 10],
            [0, 20],
            [0, 30],
        ],
        dtype=float,
    )

    result = eg(
        data=data,
        speaker_positions=speaker_positions,
    )

    assert result == pytest.approx(1.0)


def test_eg_uses_speaker_indices_to_find_physical_elevations():
    """
    Physical elevations are selected using the speaker index in column 1.

    The presentation order is deliberately different from the order in
    speaker_positions.
    """
    data = np.array(
        [
            [0, 2, 0, 20],
            [1, 0, 0, 0],
            [2, 3, 0, 30],
            [3, 1, 0, 10],
        ],
        dtype=float,
    )

    speaker_positions = np.array(
        [
            [0, 0],
            [0, 10],
            [0, 20],
            [0, 30],
        ],
        dtype=float,
    )

    result = eg(
        data=data,
        speaker_positions=speaker_positions,
    )

    assert result == pytest.approx(1.0)


"""@pytest.mark.xfail(
    strict=True,
    reason=(
        "eg currently regresses physical elevation on perceived elevation, "
        "although its docstring describes perceived elevation regressed on "
        "physical elevation."
    ),
)"""
def test_eg_regresses_perceived_on_physical_elevation():
    """
    According to the docstring, EG should be the slope of perceived
    elevation versus physical elevation.

    Physical elevations:
        0, 10, 20, 30

    Perceived elevations:
        0, 20, 40, 60

    The documented EG is therefore 2.
    """
    data = np.array(
        [
            [0, 0, 0, 0],
            [1, 1, 0, 20],
            [2, 2, 0, 40],
            [3, 3, 0, 60],
        ],
        dtype=float,
    )

    speaker_positions = np.array(
        [
            [0, 0],
            [0, 10],
            [0, 20],
            [0, 30],
        ],
        dtype=float,
    )

    result = eg(
        data=data,
        speaker_positions=speaker_positions,
    )

    assert result == pytest.approx(2.0)
