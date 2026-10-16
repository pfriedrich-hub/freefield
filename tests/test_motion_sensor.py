import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy
import pytest

import freefield.motion_sensor as motion_sensor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class PoseSequenceState:
    """
    Fake State object returning a sequence of sensor poses.

    motion_sensor.get_pose accesses self.device.pose twice per iteration:
        self.device.pose.yaw
        self.device.pose.roll

    Therefore each pose is returned for two consecutive property accesses.
    """

    def __init__(self, poses, connected=True):
        self._poses = poses
        self._pose_accesses = 0

        self.device = SimpleNamespace(
            is_connected=connected,
            board=object(),
        )

    @property
    def pose(self):
        index = min(
            self._pose_accesses // 2,
            len(self._poses) - 1,
        )

        yaw, roll = self._poses[index]

        self._pose_accesses += 1

        return SimpleNamespace(
            yaw=yaw,
            roll=roll,
        )


def _patch_connect_dependencies(monkeypatch, discovered_devices):
    """
    Patch all external MbientLab dependencies required by Sensor.connect().
    """

    # ------------------------------------------------------------------
    # Fake BLE scanner
    # ------------------------------------------------------------------

    handler = {}

    scanner = MagicMock()

    def set_handler(callback):
        handler["callback"] = callback

    def start():
        for mac, name in discovered_devices:
            handler["callback"](
                SimpleNamespace(
                    mac=mac,
                    name=name,
                )
            )

    scanner.set_handler.side_effect = set_handler
    scanner.start.side_effect = start

    monkeypatch.setattr(
        motion_sensor,
        "BleScanner",
        scanner,
        raising=False,
    )

    # ------------------------------------------------------------------
    # Fake time module
    #
    # Important:
    # Do NOT patch motion_sensor.time.time directly because
    # motion_sensor.time refers to Python's global time module.
    # That would also affect logging, which internally calls time.time().
    # ------------------------------------------------------------------

    fake_time = SimpleNamespace(
        time=MagicMock(
            side_effect=[
                0.0,
                2.1,
            ]
        ),
        sleep=MagicMock(),
    )

    monkeypatch.setattr(
        motion_sensor,
        "time",
        fake_time,
    )

    # ------------------------------------------------------------------
    # Fake MetaWear hardware device
    # ------------------------------------------------------------------

    board = object()

    hardware_device = MagicMock()
    hardware_device.board = board
    hardware_device.is_connected = False

    connect_attempts = {
        "count": 0,
    }

    def connect():
        connect_attempts["count"] += 1

        # Simulate one failed connection attempt to test retry logic
        if connect_attempts["count"] == 1:
            raise RuntimeError(
                "temporary connection failure"
            )

        hardware_device.is_connected = True

    hardware_device.connect.side_effect = connect

    metawear = MagicMock(
        return_value=hardware_device,
    )

    monkeypatch.setattr(
        motion_sensor,
        "MetaWear",
        metawear,
        raising=False,
    )

    # ------------------------------------------------------------------
    # Fake State
    # ------------------------------------------------------------------

    callback = object()

    state = SimpleNamespace(
        device=hardware_device,
        callback=callback,
        pose=None,
        samples=0,
    )

    state_factory = MagicMock(
        return_value=state,
    )

    monkeypatch.setattr(
        motion_sensor,
        "State",
        state_factory,
    )

    # ------------------------------------------------------------------
    # Fake libmetawear
    # ------------------------------------------------------------------

    signal = object()

    libmetawear = MagicMock()

    libmetawear.mbl_mw_sensor_fusion_get_data_signal.return_value = signal

    monkeypatch.setattr(
        motion_sensor,
        "libmetawear",
        libmetawear,
        raising=False,
    )

    # ------------------------------------------------------------------
    # Fake enum values
    # ------------------------------------------------------------------

    monkeypatch.setattr(
        motion_sensor,
        "SensorFusionMode",
        SimpleNamespace(
            IMU_PLUS="IMU_PLUS",
            NDOF="NDOF",
            COMPASS="COMPASS",
            M4G="M4G",
        ),
        raising=False,
    )

    monkeypatch.setattr(
        motion_sensor,
        "SensorFusionAccRange",
        SimpleNamespace(
            _8G="8G",
        ),
        raising=False,
    )

    monkeypatch.setattr(
        motion_sensor,
        "SensorFusionGyroRange",
        SimpleNamespace(
            _2000DPS="2000DPS",
        ),
        raising=False,
    )

    monkeypatch.setattr(
        motion_sensor,
        "SensorFusionData",
        SimpleNamespace(
            EULER_ANGLE="EULER_ANGLE",
        ),
        raising=False,
    )

    return SimpleNamespace(
        scanner=scanner,
        metawear=metawear,
        hardware_device=hardware_device,
        state=state,
        state_factory=state_factory,
        libmetawear=libmetawear,
        board=board,
        signal=signal,
        callback=callback,
        time=fake_time,
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def test_state_initializes_device_and_callback(monkeypatch):
    callback = object()

    callback_factory = MagicMock(
        return_value=callback,
    )

    monkeypatch.setattr(
        motion_sensor,
        "FnVoid_VoidP_DataP",
        callback_factory,
        raising=False,
    )

    device = object()

    state = motion_sensor.State(device)

    assert state.device is device
    assert state.samples == 0
    assert state.pose is None
    assert state.callback is callback

    callback_factory.assert_called_once()

    callback_function = callback_factory.call_args.args[0]

    assert callback_function.__self__ is state
    assert callback_function.__func__ is motion_sensor.State.data_handler


def test_state_data_handler_updates_pose_and_sample_count(monkeypatch):
    monkeypatch.setattr(
        motion_sensor,
        "FnVoid_VoidP_DataP",
        lambda callback: callback,
        raising=False,
    )

    parsed_pose = object()

    parse_value = MagicMock(
        return_value=parsed_pose,
    )

    monkeypatch.setattr(
        motion_sensor,
        "parse_value",
        parse_value,
        raising=False,
    )

    state = motion_sensor.State(
        device=object(),
    )

    data = object()

    state.data_handler(
        ctx=None,
        data=data,
    )

    assert state.pose is parsed_pose
    assert state.samples == 1

    parse_value.assert_called_once_with(data)


def test_state_data_handler_increments_samples_for_every_callback(monkeypatch):
    monkeypatch.setattr(
        motion_sensor,
        "FnVoid_VoidP_DataP",
        lambda callback: callback,
        raising=False,
    )

    poses = [
        object(),
        object(),
        object(),
    ]

    parse_value = MagicMock(
        side_effect=poses,
    )

    monkeypatch.setattr(
        motion_sensor,
        "parse_value",
        parse_value,
        raising=False,
    )

    state = motion_sensor.State(
        device=object(),
    )

    state.data_handler(
        None,
        object(),
    )

    state.data_handler(
        None,
        object(),
    )

    state.data_handler(
        None,
        object(),
    )

    assert state.samples == 3
    assert state.pose is poses[-1]


# ---------------------------------------------------------------------------
# Sensor.connect
# ---------------------------------------------------------------------------

def test_connect_finds_sensor_connects_and_configures_device(monkeypatch):
    dependencies = _patch_connect_dependencies(
        monkeypatch,
        discovered_devices=[
            (
                "AA:BB:CC:DD",
                "MetaWear",
            ),
        ],
    )

    sensor = motion_sensor.Sensor()

    result = sensor.connect()

    # connect() modifies self.device but does not explicitly return anything
    assert result is None

    dependencies.scanner.set_handler.assert_called_once()
    dependencies.scanner.start.assert_called_once()
    dependencies.scanner.stop.assert_called_once()

    dependencies.metawear.assert_called_once_with(
        "AA:BB:CC:DD"
    )

    dependencies.state_factory.assert_called_once_with(
        dependencies.hardware_device
    )

    # First attempt fails, second succeeds
    assert dependencies.hardware_device.connect.call_count == 2

    assert sensor.device is dependencies.state

    dependencies.libmetawear.mbl_mw_settings_set_connection_parameters.assert_called_once_with(
        dependencies.board,
        7.5,
        7.5,
        0,
        6000,
    )

    dependencies.libmetawear.mbl_mw_sensor_fusion_set_mode.assert_called_once_with(
        dependencies.board,
        "IMU_PLUS",
    )

    dependencies.libmetawear.mbl_mw_sensor_fusion_set_acc_range.assert_called_once_with(
        dependencies.board,
        "8G",
    )

    dependencies.libmetawear.mbl_mw_sensor_fusion_set_gyro_range.assert_called_once_with(
        dependencies.board,
        "2000DPS",
    )

    dependencies.libmetawear.mbl_mw_sensor_fusion_write_config.assert_called_once_with(
        dependencies.board,
    )

    dependencies.libmetawear.mbl_mw_sensor_fusion_get_data_signal.assert_called_once_with(
        dependencies.board,
        "EULER_ANGLE",
    )

    dependencies.libmetawear.mbl_mw_datasignal_subscribe.assert_called_once_with(
        dependencies.signal,
        None,
        dependencies.callback,
    )

    dependencies.libmetawear.mbl_mw_sensor_fusion_enable_data.assert_called_once_with(
        dependencies.board,
        "EULER_ANGLE",
    )

    dependencies.libmetawear.mbl_mw_sensor_fusion_start.assert_called_once_with(
        dependencies.board,
    )


def test_connect_returns_none_if_no_sensor_is_found(monkeypatch):
    scanner = MagicMock()

    monkeypatch.setattr(
        motion_sensor,
        "BleScanner",
        scanner,
        raising=False,
    )

    # Replace motion_sensor.time completely instead of patching
    # time.time() on Python's shared time module.
    fake_time = SimpleNamespace(
        time=MagicMock(
            side_effect=[
                0.0,
                2.1,
            ]
        ),
        sleep=MagicMock(),
    )

    monkeypatch.setattr(
        motion_sensor,
        "time",
        fake_time,
    )

    monkeypatch.setattr(
        "builtins.input",
        lambda: "n",
    )

    sensor = motion_sensor.Sensor()

    result = sensor.connect()

    assert result is None
    assert sensor.device is None

    scanner.set_handler.assert_called_once()
    scanner.start.assert_called_once()

    # Because the method returns before the stop() call,
    # stop() should not have been called.
    scanner.stop.assert_not_called()


def test_connect_selects_requested_sensor_if_multiple_are_found(monkeypatch):
    dependencies = _patch_connect_dependencies(
        monkeypatch,
        discovered_devices=[
            (
                "AA:AA:AA:AA",
                "MetaWear",
            ),
            (
                "BB:BB:BB:BB",
                "MetaWear",
            ),
        ],
    )

    monkeypatch.setattr(
        "builtins.input",
        lambda: "1",
    )

    sensor = motion_sensor.Sensor()

    sensor.connect()

    dependencies.metawear.assert_called_once_with(
        "BB:BB:BB:BB"
    )


# ---------------------------------------------------------------------------
# Sensor.get_pose
# ---------------------------------------------------------------------------

def test_get_pose_returns_mean_of_valid_measurements():
    sensor = motion_sensor.Sensor()

    sensor.device = PoseSequenceState(
        poses=[
            (10, 20),
            (12, 22),
            (11, 21),
        ],
    )

    pose = sensor.get_pose(
        n_datapoints=3,
        calibrate=False,
    )

    numpy.testing.assert_allclose(
        pose,
        numpy.array(
            [
                11.0,
                21.0,
            ]
        ),
    )


def test_get_pose_converts_psychoacoustic_azimuth():
    sensor = motion_sensor.Sensor()

    sensor.device = PoseSequenceState(
        poses=[
            (
                270,
                30,
            ),
        ],
    )

    pose = sensor.get_pose(
        n_datapoints=1,
        calibrate=False,
        convention="psychoacoustics",
    )

    numpy.testing.assert_allclose(
        pose,
        numpy.array(
            [
                -90.0,
                30.0,
            ]
        ),
    )


def test_get_pose_converts_physics_azimuth():
    sensor = motion_sensor.Sensor()

    sensor.device = PoseSequenceState(
        poses=[
            (
                90,
                30,
            ),
        ],
    )

    pose = sensor.get_pose(
        n_datapoints=1,
        calibrate=False,
        convention="physics",
    )

    numpy.testing.assert_allclose(
        pose,
        numpy.array(
            [
                270.0,
                30.0,
            ]
        ),
    )


def test_get_pose_rejects_invalid_convention():
    sensor = motion_sensor.Sensor()

    sensor.device = PoseSequenceState(
        poses=[
            (
                30,
                20,
            ),
        ],
    )

    with pytest.raises(
        ValueError,
        match='Convention must be "psychoacoustics" or "physics"',
    ):
        sensor.get_pose(
            n_datapoints=1,
            calibrate=False,
            convention="invalid",
        )


def test_get_pose_ignores_nan_and_zero_measurements():
    sensor = motion_sensor.Sensor()

    sensor.device = PoseSequenceState(
        poses=[
            (
                numpy.nan,
                20,
            ),
            (
                0,
                20,
            ),
            (
                30,
                40,
            ),
        ],
    )

    pose = sensor.get_pose(
        n_datapoints=1,
        calibrate=False,
    )

    numpy.testing.assert_allclose(
        pose,
        numpy.array(
            [
                30.0,
                40.0,
            ]
        ),
    )


def test_get_pose_applies_calibration_offset():
    sensor = motion_sensor.Sensor()

    sensor.device = PoseSequenceState(
        poses=[
            (
                30,
                20,
            ),
        ],
    )

    sensor.pose_offset = numpy.array(
        [
            10,
            5,
        ]
    )

    pose = sensor.get_pose(
        n_datapoints=1,
        calibrate=True,
    )

    numpy.testing.assert_allclose(
        pose,
        numpy.array(
            [
                20.0,
                15.0,
            ]
        ),
    )


def test_get_pose_does_not_apply_offset_when_calibration_disabled():
    sensor = motion_sensor.Sensor()

    sensor.device = PoseSequenceState(
        poses=[
            (
                30,
                20,
            ),
        ],
    )

    sensor.pose_offset = numpy.array(
        [
            10,
            5,
        ]
    )

    pose = sensor.get_pose(
        n_datapoints=1,
        calibrate=False,
    )

    numpy.testing.assert_allclose(
        pose,
        numpy.array(
            [
                30.0,
                20.0,
            ]
        ),
    )


def test_get_pose_warns_if_device_is_not_calibrated(caplog):
    sensor = motion_sensor.Sensor()

    sensor.device = PoseSequenceState(
        poses=[
            (
                30,
                20,
            ),
        ],
    )

    sensor.pose_offset = None

    with caplog.at_level(logging.WARNING):
        pose = sensor.get_pose(
            n_datapoints=1,
            calibrate=True,
        )

    numpy.testing.assert_allclose(
        pose,
        numpy.array(
            [
                30.0,
                20.0,
            ]
        ),
    )

    assert "Device not calibrated" in caplog.text


def test_get_pose_reconnects_if_connection_is_lost(monkeypatch):
    sensor = motion_sensor.Sensor()

    sensor.device = PoseSequenceState(
        poses=[
            (
                30,
                20,
            ),
        ],
        connected=False,
    )

    reconnect = MagicMock()

    sensor.connect = reconnect

    monkeypatch.setattr(
        "builtins.input",
        lambda: "Y",
    )

    result = sensor.get_pose(
        n_datapoints=1,
        calibrate=False,
    )

    assert result is None

    reconnect.assert_called_once_with()


def test_get_pose_does_not_reconnect_if_user_declines(monkeypatch):
    sensor = motion_sensor.Sensor()

    sensor.device = PoseSequenceState(
        poses=[
            (
                30,
                20,
            ),
        ],
        connected=False,
    )

    reconnect = MagicMock()

    sensor.connect = reconnect

    monkeypatch.setattr(
        "builtins.input",
        lambda: "n",
    )

    result = sensor.get_pose(
        n_datapoints=1,
        calibrate=False,
    )

    assert result is None

    reconnect.assert_not_called()


# ---------------------------------------------------------------------------
# Sensor.halt
# ---------------------------------------------------------------------------

def test_halt_stops_unsubscribes_and_disconnects(monkeypatch):
    sensor = motion_sensor.Sensor()

    board = object()
    signal = object()

    hardware_device = SimpleNamespace(
        board=board,
        disconnect=MagicMock(),
    )

    state = SimpleNamespace(
        device=hardware_device,
    )

    sensor.device = state

    libmetawear = MagicMock()

    libmetawear.mbl_mw_sensor_fusion_get_data_signal.return_value = signal

    monkeypatch.setattr(
        motion_sensor,
        "libmetawear",
        libmetawear,
        raising=False,
    )

    monkeypatch.setattr(
        motion_sensor,
        "SensorFusionData",
        SimpleNamespace(
            EULER_ANGLE="EULER_ANGLE",
        ),
        raising=False,
    )

    # Same principle as above:
    # replace motion_sensor.time rather than modifying global time.sleep.
    fake_time = SimpleNamespace(
        sleep=MagicMock(),
    )

    monkeypatch.setattr(
        motion_sensor,
        "time",
        fake_time,
    )

    sensor.halt()

    libmetawear.mbl_mw_sensor_fusion_stop.assert_called_once_with(
        board,
    )

    libmetawear.mbl_mw_sensor_fusion_get_data_signal.assert_called_once_with(
        board,
        "EULER_ANGLE",
    )

    libmetawear.mbl_mw_datasignal_unsubscribe.assert_called_once_with(
        signal,
    )

    libmetawear.mbl_mw_debug_disconnect.assert_called_once_with(
        board,
    )

    fake_time.sleep.assert_called_once_with(
        0.5,
    )

    hardware_device.disconnect.assert_called_once_with()

    assert sensor.device is None


def test_halt_does_nothing_if_no_device_is_connected(monkeypatch):
    sensor = motion_sensor.Sensor()

    libmetawear = MagicMock()

    monkeypatch.setattr(
        motion_sensor,
        "libmetawear",
        libmetawear,
        raising=False,
    )

    sensor.device = None

    sensor.halt()

    assert sensor.device is None
    assert libmetawear.mock_calls == []


# ---------------------------------------------------------------------------
# Sensor.set_fusion_mode
# ---------------------------------------------------------------------------

def test_set_fusion_mode_sets_requested_mode(monkeypatch):
    sensor = motion_sensor.Sensor()

    board = object()

    sensor.device = SimpleNamespace(
        device=SimpleNamespace(
            board=board,
        )
    )

    libmetawear = MagicMock()

    monkeypatch.setattr(
        motion_sensor,
        "libmetawear",
        libmetawear,
        raising=False,
    )

    monkeypatch.setattr(
        motion_sensor,
        "SensorFusionMode",
        SimpleNamespace(
            NDOF="NDOF_VALUE",
            COMPASS="COMPASS_VALUE",
            M4G="M4G_VALUE",
            IMU_PLUS="IMU_PLUS_VALUE",
        ),
        raising=False,
    )

    sensor.set_fusion_mode(
        "NDoF",
    )

    libmetawear.mbl_mw_sensor_fusion_set_mode.assert_called_once_with(
        board,
        "NDOF_VALUE",
    )

    libmetawear.mbl_mw_sensor_fusion_write_config.assert_called_once_with(
        board,
    )


def test_set_fusion_mode_raises_for_unknown_mode(monkeypatch):
    sensor = motion_sensor.Sensor()

    board = object()

    sensor.device = SimpleNamespace(
        device=SimpleNamespace(
            board=board,
        )
    )

    libmetawear = MagicMock()

    monkeypatch.setattr(
        motion_sensor,
        "libmetawear",
        libmetawear,
        raising=False,
    )

    monkeypatch.setattr(
        motion_sensor,
        "SensorFusionMode",
        SimpleNamespace(
            NDOF="NDOF_VALUE",
        ),
        raising=False,
    )

    with pytest.raises(AttributeError):
        sensor.set_fusion_mode(
            "not_a_mode",
        )

    libmetawear.mbl_mw_sensor_fusion_set_mode.assert_not_called()
    libmetawear.mbl_mw_sensor_fusion_write_config.assert_not_called()

def test_get_pose_filters_outliers():
    sensor = motion_sensor.Sensor()

    sensor.device = PoseSequenceState(
        poses=[
            (10, 20),
            (11, 21),
            (9, 19),
            (10, 20),
            (100, 120),
        ]
    )

    pose = sensor.get_pose(
        n_datapoints=5,
        calibrate=False,
    )

    numpy.testing.assert_allclose(
        pose,
        numpy.array([10.0, 20.0]),
        atol=1.0,
    )