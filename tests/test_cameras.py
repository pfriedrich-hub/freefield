"""
Script for testing the functions of cameras.py


"""

import importlib
import logging
from unittest.mock import MagicMock
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers and fixtures
# ---------------------------------------------------------------------------

camera_module = importlib.import_module("freefield.cameras")
Cameras = camera_module.Cameras
FlirCams = camera_module.FlirCams
WebCams = camera_module.WebCams


# ---------------------------------------------------------------------------
# initialize
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("kind", "camera_class_name"),
    [
        ("flir", "FlirCams"),
        ("webcam", "WebCams"),
    ],
)
def test_initialize_returns_correct_camera_kind(
        monkeypatch,
        kind,
        camera_class_name,
):
    camera_mock = MagicMock()
    expected = object()
    camera_mock.return_value = expected

    monkeypatch.setattr(
        camera_module,
        camera_class_name,
        camera_mock,
    )

    result = camera_module.initialize(kind)

    camera_mock.assert_called_once_with()
    assert result is expected


@pytest.mark.parametrize(
    ("kind", "camera_class_name"),
    [
        ("FLIR", "FlirCams"),
        ("fLIR", "FlirCams"),
        ("WebCam", "WebCams"),
        ("webcam", "WebCams"),
    ],
)
def initialize_is_case_insensitive(
        monkeypatch,
        kind,
        camera_class_name,
):
    camera_mock = MagicMock()
    expected = object()
    camera_mock.return_value = expected

    monkeypatch.setattr(
        camera_module,
        camera_class_name,
        camera_mock,
    )

    result = camera_module.initialize(kind)

    camera_mock.assert_called_once_with()
    assert result is expected


def test_initialize_raises_value_error_for_unsupported_camera_types():
    with pytest.raises(
            ValueError
    ):
        camera_module.initialize(
            kind="invalid"
        )

# ---------------------------------------------------------------------------
# Cameras ()
# ---------------------------------------------------------------------------

# __init__
def test_init_sets_correct_default_state():
    camera = camera_module.Cameras()

    assert camera.imsize is None
    assert camera.model is None
    assert isinstance(camera.calibration, dict)
    assert len(camera.calibration) == 0
    assert camera.n_cams is None

# set_detectin_threshold
def test_set_detection_threshold_sets_threshold():
    camera = camera_module.Cameras()

    camera.model = MagicMock()
    expected = 7.0

    camera.set_detection_threshold(expected)

    assert camera.model.threshold == expected

# get_head_pose
def test_get_head_pose_raises_import_error_if_model_unspecified():
    camera = camera_module.Cameras()
    camera.model = None

    with pytest.raises(
            ImportError
    ):
        camera.get_head_pose()

def test_get_head_pose_calls_acquire_images_and_pose_from_image_correctly():
    n_images = 2
    n_cams = 3
    camera = camera_module.Cameras()

    images = np.zeros(
        (20, 30, n_images, n_cams),
        dtype="uint8",
    )

    # Give each image a unique value so that call order can be verified.
    for i_cam in range(n_cams):
        for i_image in range(n_images):
            images[:, :, i_image, i_cam] = i_cam * n_images + i_image

    expected_images = [
        images[:, :, i_image, i_cam]
        for i_cam in range(n_cams)
        for i_image in range(n_images)
    ]

    camera.acquire_images = MagicMock(return_value=images)
    camera.model = MagicMock()
    camera.model.pose_from_image.return_value = (10, 20, None)

    camera.get_head_pose(
        convert=False,
        average_axis=None,
        n_images=n_images,
    )

    camera.acquire_images.assert_called_once_with(n_images)

    assert camera.model.pose_from_image.call_count == n_images * n_cams

    for actual_call, expected_image in zip(
        camera.model.pose_from_image.call_args_list,
        expected_images,
    ):
        actual_image = actual_call.args[0]
        np.testing.assert_array_equal(actual_image, expected_image)


def test_get_head_pose_stores_azimuth_and_elevation_correctly():
    n_images = 2
    n_cams = 2

    camera = camera_module.Cameras()
    images = np.zeros(
        (20, 30, n_images, n_cams),
        dtype="uint8",
    )

    camera.acquire_images = MagicMock(return_value=images)
    camera.model = MagicMock()

    camera.model.pose_from_image.side_effect = [
        (10, 20, None),
        (11, 21, None),
        (12, 22, None),
        (13, 23, None),
    ]

    result = camera.get_head_pose(
        convert=False,
        average_axis=None,
        n_images=n_images,
    )

    expected = np.array([
        [
            [10, 12],
            [11, 13],
        ],
        [
            [20, 22],
            [21, 23],
        ],
    ])

    np.testing.assert_array_equal(result, expected)

def test_get_head_pose_changes_image_resolution_if_specified():
    n_images = 2
    n_cams = 2

    camera = camera_module.Cameras()
    images = np.zeros(
        (20, 30, n_images, n_cams),
        dtype="uint8",
    )

    camera.acquire_images = MagicMock(return_value=images)
    camera.change_image_res = MagicMock(
        side_effect=lambda image, resolution: image
    )
    camera.model = MagicMock()
    camera.model.pose_from_image.return_value = (10, 20, None)

    # resolution == 1: image resolution should not be changed
    camera.get_head_pose(
        convert=False,
        average_axis=None,
        n_images=n_images,
        resolution=1,
    )

    camera.change_image_res.assert_not_called()

    camera.change_image_res.reset_mock()

    # resolution < 1: every image should be resized
    camera.get_head_pose(
        convert=False,
        average_axis=None,
        n_images=n_images,
        resolution=0.5,
    )

    assert camera.change_image_res.call_count == n_images * n_cams

def test_get_head_pose_converts_coordinates_if_specified():
    n_images = 2
    n_cams = 2

    camera = camera_module.Cameras()
    images = np.zeros(
        (20, 30, n_images, n_cams),
        dtype="uint8",
    )

    camera.acquire_images = MagicMock(return_value=images)
    camera.model = MagicMock()
    camera.model.pose_from_image.return_value = (10, 20, None)

    camera.convert_coordinates = MagicMock()

    camera.get_head_pose(
        convert=True,
        average_axis=None,
        n_images=n_images,
    )

    camera.convert_coordinates.assert_called_once()

    camera.convert_coordinates.reset_mock()

    camera.get_head_pose(
        convert=False,
        average_axis=None,
        n_images=n_images,
    )

    camera.convert_coordinates.assert_not_called()


def test_get_head_pose_averages_specified_axis_correctly():
    n_images = 2
    n_cams = 2

    camera = camera_module.Cameras()
    images = np.zeros(
        (20, 30, n_images, n_cams),
        dtype="uint8",
    )

    camera.acquire_images = MagicMock(return_value=images)
    camera.model = MagicMock()

    pose_from_image_results = [
        (10, 5, None),    # cam0, image0
        (20, 10, None),   # cam0, image1
        (30, 15, None),   # cam1, image0
        (40, 20, None),   # cam1, image1
    ]

    expected_axis_none = np.array([
        # azimuth
        [
            [10, 30],
            [20, 40],
        ],

        # elevation
        [
            [5, 15],
            [10, 20],
        ],
    ])

    expected_axis_0 = np.array([
        [7.5, 22.5],
        [15, 30],
    ])

    expected_axis_1 = np.array([
        [15, 35],
        [7.5, 17.5],
    ])

    expected_axis_2 = np.array([
        [20, 30],
        [10, 15],
    ])

    args = [0, 1, 2, None]
    expected = [
        expected_axis_0,
        expected_axis_1,
        expected_axis_2,
        expected_axis_none,
    ]

    for axis, expected_result in zip(args, expected):
        camera.model.pose_from_image.reset_mock()
        camera.model.pose_from_image.side_effect = pose_from_image_results

        result = camera.get_head_pose(
            convert=False,
            average_axis=axis,
            n_images=n_images,
        )

        np.testing.assert_array_equal(
            result,
            expected_result,
        )
# change_image_res

@pytest.mark.parametrize(
    "resolution",
    [1, 0.5, 2, 0.1]
)
def test_change_image_res_returns_expected_dimensions(resolution):
    height = 20
    width = 30
    image = np.zeros((height, width), dtype="uint8")
    camera = camera_module.Cameras()
    camera.imsize = (height, width)

    result = camera.change_image_res(image=image, resolution=resolution)

    expected_shape = (
        int(height * resolution),
        int(width * resolution),
    )

    assert isinstance(result, np.ndarray)
    assert result.shape == expected_shape

def test_change_image_res_raises_value_error_if_res_is_zero():
    height = 20
    width = 30
    image = np.zeros((height, width), dtype="uint8")
    camera = camera_module.Cameras()
    camera.imsize = (height, width, None)

    with pytest.raises(
        ValueError
    ):
        camera.change_image_res(
            image=image,
            resolution=0
        )

def test_change_image_res_resizes_example_image():
    image = np.array(
        [
            [0, 0, 255, 255],
            [0, 0, 255, 255],
            [100, 100, 200, 200],
            [100, 100, 200, 200],
        ],
        dtype="uint8",
    )

    camera = camera_module.Cameras()
    camera.imsize = image.shape

    result = camera.change_image_res(
        image=image,
        resolution=0.5,
    )

    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)
    assert result.dtype == np.uint8


# convert coordinates
def test_convert_coordinates_raises_error_for_empty_calibration():
    n_cams = 2
    n_images = 2

    pose = np.zeros([2, n_cams, n_images])
    camera = camera_module.Cameras()

    assert not camera.calibration
    with pytest.raises(
        ValueError
    ):
        camera.convert_coordinates(pose=pose)

def test_convert_coordinates_uses_correct_calibration_coefficients():
    camera = camera_module.Cameras()

    camera.calibration = {
        "cam0": {
            "azimuth": {"a": 1, "b": 2},
            "elevation": {"a": 10, "b": 3},
        },
        "cam1": {
            "azimuth": {"a": -1, "b": 4},
            "elevation": {"a": -10, "b": 5},
        },
    }

    pose = np.array([
        # azimuth
        [
            [1.0, 2.0],   # image 0: cam0, cam1
            [3.0, 4.0],   # image 1: cam0, cam1
        ],

        # elevation
        [
            [5.0, 6.0],   # image 0: cam0, cam1
            [7.0, 8.0],   # image 1: cam0, cam1
        ],
    ])

    result = camera.convert_coordinates(pose.copy())

    expected = np.array([
        # azimuth
        [
            [3.0, 7.0],    # cam0: 1 + 2*1, cam1: -1 + 4*2
            [7.0, 15.0],   # cam0: 1 + 2*3, cam1: -1 + 4*4
        ],

        # elevation
        [
            [25.0, 20.0],  # cam0: 10 + 3*5, cam1: -10 + 5*6
            [31.0, 30.0],  # cam0: 10 + 3*7, cam1: -10 + 5*8
        ],
    ])

    np.testing.assert_array_equal(result, expected)

# calibrate
def test_calibrate_raises_value_error_for_different_coordinate_lengths():
    camera = camera_module.Cameras()

    world_coordinates = [
        np.array([0, 0]),
        np.array([1, 1]),
    ]
    camera_coordinates = [
        np.array([[0], [0]]),
    ]

    with pytest.raises(
        ValueError,
        match="Camera and world coordinates must be of the same shape",
    ):
        camera.calibrate(
            world_coordinates,
            camera_coordinates,
            plot=False,
        )


def test_calibrate_stores_correct_calibration_coefficients_for_each_camera():
    camera = camera_module.Cameras()
    camera.n_cams = 2

    # World coordinates:
    # measurement x [azimuth, elevation]
    world_coordinates = np.array([
        [0, 10],
        [2, 20],
        [4, 30],
        [6, 40],
    ], dtype=float)

    # Camera coordinates are chosen so that:
    #
    # cam0 azimuth:   world = 1 + 2 * camera
    # cam1 azimuth:   world = -2 + 4 * camera
    # cam0 elevation: world = 5 + 5 * camera
    # cam1 elevation: world = -10 + 10 * camera
    #
    # shape:
    # measurement x angle x camera
    camera_coordinates = np.array([
        [
            [-0.5, 0.5],   # azimuth: cam0, cam1
            [1.0, 2.0],    # elevation: cam0, cam1
        ],
        [
            [0.5, 1.0],
            [3.0, 3.0],
        ],
        [
            [1.5, 1.5],
            [5.0, 4.0],
        ],
        [
            [2.5, 2.0],
            [7.0, 5.0],
        ],
    ], dtype=float)

    camera.calibrate(
        world_coordinates,
        camera_coordinates,
        plot=False,
    )

    expected = {
        "cam0": {
            "azimuth": {"a": 1.0, "b": 2.0},
            "elevation": {"a": 5.0, "b": 5.0},
        },
        "cam1": {
            "azimuth": {"a": -2.0, "b": 4.0},
            "elevation": {"a": -10.0, "b": 10.0},
        },
    }

    assert set(camera.calibration.keys()) == {
        "cam0",
        "cam1",
    }

    for cam, angles in expected.items():
        assert set(camera.calibration[cam].keys()) == {
            "azimuth",
            "elevation",
        }

        for angle, coefficients in angles.items():
            np.testing.assert_allclose(
                camera.calibration[cam][angle]["a"],
                coefficients["a"],
            )
            np.testing.assert_allclose(
                camera.calibration[cam][angle]["b"],
                coefficients["b"],
            )


@pytest.mark.parametrize(
    "correlation",
    [0.84, -0.84],
)
def test_calibrate_logs_warning_for_low_correlation(
    monkeypatch,
    caplog,
    correlation,
):
    camera = camera_module.Cameras()
    camera.n_cams = 1

    world_coordinates = np.array([
        [0, 0],
        [1, 1],
    ], dtype=float)

    camera_coordinates = np.array([
        [[0], [0]],
        [[1], [1]],
    ], dtype=float)

    linregress_mock = MagicMock(
        return_value=(
            1.0,          # b
            0.0,          # a
            correlation,  # r
            None,
            None,
        )
    )

    monkeypatch.setattr(
        camera_module.stats,
        "linregress",
        linregress_mock,
    )

    with caplog.at_level(logging.WARNING):
        camera.calibrate(
            world_coordinates,
            camera_coordinates,
            plot=False,
        )

    correlation_warnings = [
        record
        for record in caplog.records
        if "Correlation for camera" in record.message
    ]

    assert len(correlation_warnings) == 2
    assert "camera 0 azimuth" in correlation_warnings[0].message
    assert "camera 0 elevation" in correlation_warnings[1].message


@pytest.mark.parametrize(
    "correlation",
    [0.85, -0.85, 1.0, -1.0],
)
def test_calibrate_does_not_log_warning_for_sufficient_correlation(
    monkeypatch,
    caplog,
    correlation,
):
    camera = camera_module.Cameras()
    camera.n_cams = 1

    world_coordinates = np.array([
        [0, 0],
        [1, 1],
    ], dtype=float)

    camera_coordinates = np.array([
        [[0], [0]],
        [[1], [1]],
    ], dtype=float)

    linregress_mock = MagicMock(
        return_value=(
            1.0,
            0.0,
            correlation,
            None,
            None,
        )
    )

    monkeypatch.setattr(
        camera_module.stats,
        "linregress",
        linregress_mock,
    )

    with caplog.at_level(logging.WARNING):
        camera.calibrate(
            world_coordinates,
            camera_coordinates,
            plot=False,
        )

    correlation_warnings = [
        record
        for record in caplog.records
        if "Correlation for camera" in record.message
    ]

    assert correlation_warnings == []


def test_calibrate_does_not_plot_if_plot_is_false(monkeypatch):
    camera = camera_module.Cameras()
    camera.n_cams = 1

    world_coordinates = np.array([
        [0, 0],
        [1, 1],
    ], dtype=float)

    camera_coordinates = np.array([
        [[0], [0]],
        [[1], [1]],
    ], dtype=float)

    subplots_mock = MagicMock()
    show_mock = MagicMock()

    monkeypatch.setattr(
        camera_module.plt,
        "subplots",
        subplots_mock,
    )
    monkeypatch.setattr(
        camera_module.plt,
        "show",
        show_mock,
    )

    camera.calibrate(
        world_coordinates,
        camera_coordinates,
        plot=False,
    )

    subplots_mock.assert_not_called()
    show_mock.assert_not_called()


def test_calibrate_plots_calibration_if_plot_is_true(monkeypatch):
    camera = camera_module.Cameras()
    camera.n_cams = 1

    world_coordinates = np.array([
        [0, 0],
        [1, 1],
    ], dtype=float)

    camera_coordinates = np.array([
        [[0], [0]],
        [[1], [1]],
    ], dtype=float)

    fig_mock = MagicMock()
    axes = [
        MagicMock(),
        MagicMock(),
    ]

    subplots_mock = MagicMock(
        return_value=(fig_mock, axes)
    )
    show_mock = MagicMock()

    monkeypatch.setattr(
        camera_module.plt,
        "subplots",
        subplots_mock,
    )
    monkeypatch.setattr(
        camera_module.plt,
        "show",
        show_mock,
    )

    camera.calibrate(
        world_coordinates,
        camera_coordinates,
        plot=True,
    )

    subplots_mock.assert_called_once_with(2)
    fig_mock.suptitle.assert_called_once_with(
        "World vs Camera Coordinates"
    )

    for axis in axes:
        axis.scatter.assert_called_once()
        axis.plot.assert_called_once()
        axis.set_title.assert_called_once()
        axis.legend.assert_called_once()
        axis.set_xlabel.assert_called_once_with(
            "camera coordinates in degree"
        )
        axis.set_ylabel.assert_called_once_with(
            "world coordinates in degree"
        )

    show_mock.assert_called_once_with()

# ---------------------------------------------------------------------
# WebCams.__init__
# ---------------------------------------------------------------------

def test_webcams_init_detects_cameras_and_sets_state(monkeypatch):
    cam0 = MagicMock()
    cam1 = MagicMock()
    no_cam = MagicMock()

    cam0.isOpened.return_value = True
    cam1.isOpened.return_value = True
    no_cam.isOpened.return_value = False

    cv2_mock = MagicMock()
    cv2_mock.VideoCapture.side_effect = [
        cam0,
        cam1,
        no_cam,
    ]

    monkeypatch.setattr(camera_module, "cv2", cv2_mock)
    monkeypatch.setattr(camera_module, "headpose", False)

    acquire_mock = MagicMock(
        return_value=np.zeros((20, 30, 1, 2), dtype="uint8")
    )
    monkeypatch.setattr(
        camera_module.WebCams,
        "acquire_images",
        acquire_mock,
    )

    camera = camera_module.WebCams()

    assert camera.cams == [cam0, cam1]
    assert camera.n_cams == 2
    assert camera.imsize == (20, 30)
    assert camera.model is None

    acquire_mock.assert_called_once_with(n_images=1)


def test_webcams_init_creates_pose_estimator_if_headpose_available(
    monkeypatch,
):
    no_cam = MagicMock()
    no_cam.isOpened.return_value = False

    cv2_mock = MagicMock()
    cv2_mock.VideoCapture.return_value = no_cam

    model = MagicMock()
    headpose_mock = MagicMock()
    headpose_mock.PoseEstimator.return_value = model

    monkeypatch.setattr(camera_module, "cv2", cv2_mock)
    monkeypatch.setattr(camera_module, "headpose", headpose_mock)

    monkeypatch.setattr(
        camera_module.WebCams,
        "acquire_images",
        MagicMock(
            return_value=np.zeros((20, 30), dtype="uint8")
        ),
    )

    camera = camera_module.WebCams()

    headpose_mock.PoseEstimator.assert_called_once_with()
    assert camera.model is model


# ---------------------------------------------------------------------
# WebCams.acquire_images
#
# Here __init__ is deliberately skipped so that no real hardware
# initialization can happen.
# ---------------------------------------------------------------------

def test_webcams_acquire_images_returns_expected_images(monkeypatch):
    camera = object.__new__(camera_module.WebCams)

    camera.imsize = (2, 3)
    camera.n_cams = 2

    cam0 = MagicMock()
    cam1 = MagicMock()
    camera.cams = [cam0, cam1]

    image0 = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ],
        dtype="uint8",
    )
    image1 = np.array(
        [
            [10, 20, 30],
            [40, 50, 60],
        ],
        dtype="uint8",
    )

    cam0.retrieve.return_value = (True, image0)
    cam1.retrieve.return_value = (True, image1)

    cv2_mock = MagicMock()
    cv2_mock.CAP_PROP_FRAME_COUNT = 2

    monkeypatch.setattr(camera_module, "cv2", cv2_mock)

    result = camera.acquire_images(n_images=2)

    expected = np.zeros((2, 3, 2, 2), dtype="uint8")

    expected[:, :, 0, 0] = image0
    expected[:, :, 1, 0] = image0
    expected[:, :, 0, 1] = image1
    expected[:, :, 1, 1] = image1

    np.testing.assert_array_equal(result, expected)

    assert result.dtype == np.uint8
    assert cam0.grab.call_count == 4
    assert cam1.grab.call_count == 4


def test_webcams_acquire_images_converts_color_images_to_grayscale(
    monkeypatch,
):
    camera = object.__new__(camera_module.WebCams)

    camera.imsize = (2, 3)
    camera.n_cams = 1

    cam = MagicMock()
    camera.cams = [cam]

    color_image = np.zeros((2, 3, 3), dtype="uint8")
    grayscale_image = np.ones((2, 3), dtype="uint8")

    cam.retrieve.return_value = (True, color_image)

    cv2_mock = MagicMock()
    cv2_mock.CAP_PROP_FRAME_COUNT = 1
    cv2_mock.COLOR_BGR2GRAY = 123
    cv2_mock.cvtColor.return_value = grayscale_image

    monkeypatch.setattr(camera_module, "cv2", cv2_mock)

    result = camera.acquire_images(n_images=1)

    cv2_mock.cvtColor.assert_called_once_with(
        color_image,
        cv2_mock.COLOR_BGR2GRAY,
    )

    np.testing.assert_array_equal(
        result[:, :, 0, 0],
        grayscale_image,
    )


def test_webcams_acquire_images_does_not_convert_grayscale_images(
    monkeypatch,
):
    camera = object.__new__(camera_module.WebCams)

    camera.imsize = (2, 3)
    camera.n_cams = 1

    cam = MagicMock()
    camera.cams = [cam]

    grayscale_image = np.zeros((2, 3), dtype="uint8")
    cam.retrieve.return_value = (True, grayscale_image)

    cv2_mock = MagicMock()
    cv2_mock.CAP_PROP_FRAME_COUNT = 1

    monkeypatch.setattr(camera_module, "cv2", cv2_mock)

    camera.acquire_images(n_images=1)

    cv2_mock.cvtColor.assert_not_called()


def test_webcams_acquire_images_logs_warning_if_acquisition_fails(
    monkeypatch,
    caplog,
):
    camera = object.__new__(camera_module.WebCams)

    camera.imsize = (2, 3)
    camera.n_cams = 1

    cam = MagicMock()
    camera.cams = [cam]

    image = np.zeros((2, 3), dtype="uint8")
    cam.retrieve.return_value = (False, image)

    cv2_mock = MagicMock()
    cv2_mock.CAP_PROP_FRAME_COUNT = 1

    monkeypatch.setattr(camera_module, "cv2", cv2_mock)

    with caplog.at_level(logging.WARNING):
        camera.acquire_images(n_images=1)

    assert "could not acquire image" in caplog.text


def test_webcams_halt_releases_all_cameras():
    camera = object.__new__(camera_module.WebCams)

    cam0 = MagicMock()
    cam1 = MagicMock()

    camera.cams = [cam0, cam1]

    camera.halt()

    cam0.release.assert_called_once_with()
    cam1.release.assert_called_once_with()


# ---------------------------------------------------------------------
# FlirCams.__init__
# ---------------------------------------------------------------------

def test_flircams_init_raises_if_pyspin_is_unavailable(monkeypatch):
    monkeypatch.setattr(camera_module, "PySpin", False)

    with pytest.raises(
        ValueError,
        match="PySpin module required",
    ):
        camera_module.FlirCams()


def test_flircams_init_initializes_detected_cameras(monkeypatch):
    cam0 = MagicMock()
    cam1 = MagicMock()

    cams = MagicMock()
    cams.GetSize.return_value = 2
    cams.__iter__.return_value = iter([cam0, cam1])

    system = MagicMock()
    system.GetCameras.return_value = cams

    pyspin_mock = MagicMock()
    pyspin_mock.System.GetInstance.return_value = system

    monkeypatch.setattr(camera_module, "PySpin", pyspin_mock)
    monkeypatch.setattr(camera_module, "headpose", False)

    acquire_mock = MagicMock(
        return_value=np.zeros((20, 30, 1, 2), dtype="uint8")
    )
    monkeypatch.setattr(
        camera_module.FlirCams,
        "acquire_images",
        acquire_mock,
    )

    camera = camera_module.FlirCams()

    assert camera.system is system
    assert camera.cams is cams
    assert camera.n_cams == 2
    assert camera.imsize == (20, 30)
    assert camera.model is None

    cam0.Init.assert_called_once_with()
    cam1.Init.assert_called_once_with()

    acquire_mock.assert_called_once_with(n_images=1)


def test_flircams_init_creates_pose_estimator_if_available(
    monkeypatch,
):
    cam = MagicMock()

    cams = MagicMock()
    cams.GetSize.return_value = 1
    cams.__iter__.return_value = iter([cam])

    system = MagicMock()
    system.GetCameras.return_value = cams

    pyspin_mock = MagicMock()
    pyspin_mock.System.GetInstance.return_value = system

    model = MagicMock()
    headpose_mock = MagicMock()
    headpose_mock.PoseEstimator.return_value = model

    monkeypatch.setattr(camera_module, "PySpin", pyspin_mock)
    monkeypatch.setattr(camera_module, "headpose", headpose_mock)

    monkeypatch.setattr(
        camera_module.FlirCams,
        "acquire_images",
        MagicMock(
            return_value=np.zeros((20, 30, 1, 1), dtype="uint8")
        ),
    )

    camera = camera_module.FlirCams()

    headpose_mock.PoseEstimator.assert_called_once_with()
    assert camera.model is model


# ---------------------------------------------------------------------
# FlirCams.acquire_images
# ---------------------------------------------------------------------

def test_flircams_acquire_images_returns_expected_images(
    monkeypatch,
):
    camera = object.__new__(camera_module.FlirCams)

    camera.imsize = (2, 3)
    camera.n_cams = 1

    node = MagicMock()
    continuous = MagicMock()
    continuous.GetValue.return_value = 1
    node.GetEntryByName.return_value = continuous

    node_map = MagicMock()
    node_map.GetNode.return_value = node

    image = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ],
        dtype="uint8",
    )

    converted_image = MagicMock()
    converted_image.GetNDArray.return_value = image

    image_result = MagicMock()
    image_result.IsIncomplete.return_value = False
    image_result.Convert.return_value = converted_image

    cam = MagicMock()
    cam.GetNodeMap.return_value = node_map
    cam.GetNextImage.return_value = image_result

    camera.cams = [cam]

    pyspin_mock = MagicMock()
    pyspin_mock.CEnumerationPtr.side_effect = lambda value: value
    pyspin_mock.IsAvailable.return_value = True
    pyspin_mock.IsWritable.return_value = True
    pyspin_mock.IsReadable.return_value = True
    pyspin_mock.PixelFormat_Mono8 = 1
    pyspin_mock.HQ_LINEAR = 2

    monkeypatch.setattr(camera_module, "PySpin", pyspin_mock)
    monkeypatch.setattr(
        camera_module.time,
        "sleep",
        MagicMock(),
    )

    result = camera.acquire_images(n_images=2)

    expected = np.zeros((2, 3, 2, 1), dtype="uint8")
    expected[:, :, 0, 0] = image
    expected[:, :, 1, 0] = image

    np.testing.assert_array_equal(result, expected)

    cam.BeginAcquisition.assert_called_once_with()
    cam.EndAcquisition.assert_called_once_with()

    assert cam.GetNextImage.call_count == 2
    assert image_result.Release.call_count == 2


@pytest.mark.parametrize(
    ("available", "writable"),
    [
        (False, True),
        (True, False),
    ],
)
def test_flircams_acquire_images_raises_if_acquisition_mode_cannot_be_set(
    monkeypatch,
    available,
    writable,
):
    camera = object.__new__(camera_module.FlirCams)

    camera.imsize = (2, 3)
    camera.n_cams = 1

    node = MagicMock()

    node_map = MagicMock()
    node_map.GetNode.return_value = node

    cam = MagicMock()
    cam.GetNodeMap.return_value = node_map

    camera.cams = [cam]

    pyspin_mock = MagicMock()
    pyspin_mock.CEnumerationPtr.side_effect = lambda value: value
    pyspin_mock.IsAvailable.return_value = available
    pyspin_mock.IsWritable.return_value = writable

    monkeypatch.setattr(camera_module, "PySpin", pyspin_mock)

    with pytest.raises(
        ValueError,
        match="Unable to set acquisition to continuous",
    ):
        camera.acquire_images(n_images=1)


@pytest.mark.parametrize(
    ("available", "readable"),
    [
        (False, True),
        (True, False),
    ],
)
def test_flircams_acquire_images_raises_if_continuous_mode_cannot_be_read(
    monkeypatch,
    available,
    readable,
):
    camera = object.__new__(camera_module.FlirCams)

    camera.imsize = (2, 3)
    camera.n_cams = 1

    continuous = MagicMock()

    node = MagicMock()
    node.GetEntryByName.return_value = continuous

    node_map = MagicMock()
    node_map.GetNode.return_value = node

    cam = MagicMock()
    cam.GetNodeMap.return_value = node_map

    camera.cams = [cam]

    pyspin_mock = MagicMock()
    pyspin_mock.CEnumerationPtr.side_effect = lambda value: value

    # First IsAvailable call is for acquisition node,
    # second is for the Continuous entry.
    pyspin_mock.IsAvailable.side_effect = [
        True,
        available,
    ]
    pyspin_mock.IsWritable.return_value = True
    pyspin_mock.IsReadable.return_value = readable

    monkeypatch.setattr(camera_module, "PySpin", pyspin_mock)

    with pytest.raises(
        ValueError,
        match="Unable to set acquisition to continuous",
    ):
        camera.acquire_images(n_images=1)


def test_flircams_acquire_images_raises_for_incomplete_image(
    monkeypatch,
):
    camera = object.__new__(camera_module.FlirCams)

    camera.imsize = (2, 3)
    camera.n_cams = 1

    continuous = MagicMock()
    continuous.GetValue.return_value = 1

    node = MagicMock()
    node.GetEntryByName.return_value = continuous

    node_map = MagicMock()
    node_map.GetNode.return_value = node

    image_result = MagicMock()
    image_result.IsIncomplete.return_value = True
    image_result.GetImageStatus.return_value = 5

    cam = MagicMock()
    cam.GetNodeMap.return_value = node_map
    cam.GetNextImage.return_value = image_result

    camera.cams = [cam]

    pyspin_mock = MagicMock()
    pyspin_mock.CEnumerationPtr.side_effect = lambda value: value
    pyspin_mock.IsAvailable.return_value = True
    pyspin_mock.IsWritable.return_value = True
    pyspin_mock.IsReadable.return_value = True

    monkeypatch.setattr(camera_module, "PySpin", pyspin_mock)
    monkeypatch.setattr(
        camera_module.time,
        "sleep",
        MagicMock(),
    )

    with pytest.raises(
        ValueError,
        match="Image incomplete",
    ):
        camera.acquire_images(n_images=1)


def test_flircams_halt_deinitializes_cameras_and_releases_system():
    camera = object.__new__(camera_module.FlirCams)

    cam0 = MagicMock()
    cam1 = MagicMock()

    cam0.IsInitialized.return_value = True
    cam1.IsInitialized.return_value = True

    cams = MagicMock()
    cams.__iter__.return_value = iter([cam0, cam1])

    system = MagicMock()

    camera.cams = cams
    camera.system = system

    camera.halt()

    cam0.DeInit.assert_called_once_with()
    cam1.DeInit.assert_called_once_with()

    cams.Clear.assert_called_once_with()
    system.ReleaseInstance.assert_called_once_with()