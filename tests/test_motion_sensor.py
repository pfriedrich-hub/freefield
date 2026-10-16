import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy
import pytest

import freefield.motion_sensor as motion_sensor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class PacketStream:
    """
    Fake State object delivering a sequence of sensor packets in time.

    Packet i arrives at i * dt on a fake clock, like the BLE callback would:
    `samples` counts the packets that have arrived, `pose` is the latest one.
    Once the list is exhausted the last packet stays cached and `samples`
    stops changing, i.e. the sensor goes silent.

    `clock` replaces motion_sensor.time: it only advances on sleep(), so
    time-outs and calibration windows run deterministically without real
    waiting.
    """

    def __init__(self, poses, connected=True, dt=0.01):
        self._poses = list(poses)
        self._dt = dt
        self._t = 0.0
        self.device = SimpleNamespace(
            is_connected=connected,
            board=object(),
        )
        self.clock = SimpleNamespace(time=self._time, sleep=self._sleep)

    def _time(self):
        return self._t

    def _sleep(self, seconds):
        self._t += seconds

    def skip_to(self, index):
        """Jump the clock to the arrival of packet `index`."""
        self._t = max(self._t, index * self._dt)

    @property
    def samples(self):
        return min(int(self._t / self._dt + 1e-9), len(self._poses) - 1)

    @property
    def pose(self):
        yaw, roll = self._poses[self.samples]
        return SimpleNamespace(yaw=yaw, roll=roll)


def _sensor_with_stream(monkeypatch, poses, connected=True):
    stream = PacketStream(poses, connected=connected)
    monkeypatch.setattr(motion_sensor, "time", stream.clock)
    sensor = motion_sensor.Sensor()
    sensor.device = stream
    return sensor


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
                5.1,
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

    # Do not leave a real at-exit halt() behind for the fake device.
    monkeypatch.setattr(
        motion_sensor,
        "atexit",
        SimpleNamespace(register=MagicMock(), unregister=MagicMock()),
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
        "NDOF",
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
                5.1,
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

    # The scan is stopped before giving up.
    scanner.stop.assert_called_once()


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

def test_get_pose_returns_mean_of_valid_measurements(monkeypatch):
    sensor = _sensor_with_stream(
        monkeypatch,
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


def test_get_pose_converts_psychoacoustic_azimuth(monkeypatch):
    sensor = _sensor_with_stream(
        monkeypatch,
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


def test_get_pose_converts_physics_azimuth(monkeypatch):
    sensor = _sensor_with_stream(
        monkeypatch,
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


def test_get_pose_rejects_invalid_convention(monkeypatch):
    sensor = _sensor_with_stream(
        monkeypatch,
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


def test_get_pose_ignores_nan_and_zero_measurements(monkeypatch):
    sensor = _sensor_with_stream(
        monkeypatch,
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


def test_get_pose_applies_calibration_offset(monkeypatch):
    sensor = _sensor_with_stream(
        monkeypatch,
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


def test_get_pose_does_not_apply_offset_when_calibration_disabled(monkeypatch):
    sensor = _sensor_with_stream(
        monkeypatch,
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


def test_get_pose_warns_if_device_is_not_calibrated(caplog, monkeypatch):
    sensor = _sensor_with_stream(
        monkeypatch,
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
    sensor = _sensor_with_stream(
        monkeypatch,
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
    sensor = _sensor_with_stream(
        monkeypatch,
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
        is_connected=True,
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

def test_get_pose_filters_outliers(monkeypatch):
    sensor = _sensor_with_stream(
        monkeypatch,
        poses=[
            (10, 20),
            (11, 21),
            (9, 19),
            (10, 20),
            (100, 120),
        ],
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

# ---------------------------------------------------------------------------
# Sensor.calibrate
# ---------------------------------------------------------------------------

def _still(az, el, n, jitter=0.1, seed=0):
    rng = numpy.random.default_rng(seed)
    return [(az + rng.uniform(-jitter, jitter), el + rng.uniform(-jitter, jitter))
            for _ in range(n)]


def test_calibrate_sets_offset_to_median_of_still_window(monkeypatch):
    poses = _still(30, 10, 100)
    poses[3] = (200, 160)  # a single glitched packet must not become the zero
    sensor = _sensor_with_stream(monkeypatch, poses)

    info = sensor.calibrate()

    assert info["stable"]
    assert info["attempts"] == 1
    assert info["n_samples"] >= 5
    assert info["residual"] < 0.5
    numpy.testing.assert_allclose(sensor.pose_offset, [30, 10], atol=0.2)
    assert sensor.is_calibrated
    assert sensor.last_calibration is info
    assert len(sensor.calibration_log) == 1
    numpy.testing.assert_allclose(sensor.calibration_log[0], [30, 10], atol=0.2)


def test_calibrate_waits_for_head_to_settle(monkeypatch):
    moving = [(az, 5) for az in numpy.linspace(60, 20, 40)]  # head turning back to center
    poses = moving + _still(20, 5, 60)
    sensor = _sensor_with_stream(monkeypatch, poses)

    info = sensor.calibrate()

    assert info["stable"]
    numpy.testing.assert_allclose(sensor.pose_offset, [20, 5], atol=0.2)


def test_calibrate_is_wrap_safe_around_180(monkeypatch):
    # raw yaw alternates between 179.9 and 180.1 (= -179.9)
    poses = [(179.9 if i % 2 else 180.1, 3) for i in range(100)]
    sensor = _sensor_with_stream(monkeypatch, poses)

    sensor.calibrate()

    assert abs(abs(sensor.pose_offset[0]) - 180) < 0.2
    pose = sensor.get_pose()  # single packet: within the +-0.1 deg jitter of the zero
    numpy.testing.assert_allclose(pose, [0, 0], atol=0.25)


def test_calibrate_does_not_block_if_head_never_settles(monkeypatch, caplog):
    rng = numpy.random.default_rng(1)
    poses = [(rng.uniform(-20, 20), rng.uniform(-20, 20)) for _ in range(1000)]
    sensor = _sensor_with_stream(monkeypatch, poses)

    with caplog.at_level(logging.WARNING):
        info = sensor.calibrate(timeout_s=1.5)

    assert not info["stable"]
    assert info["attempts"] == 3
    assert "did not settle" in caplog.text
    assert len(sensor.calibration_log) == 1


def test_calibrate_with_silent_sensor_uses_cached_packet(monkeypatch, caplog):
    sensor = _sensor_with_stream(monkeypatch, [(25, 4)])  # one packet, then silence

    with caplog.at_level(logging.WARNING):
        info = sensor.calibrate()

    numpy.testing.assert_allclose(sensor.pose_offset, [25, 4])
    assert info["residual"] is None
    assert "no new samples" in caplog.text


def test_calibrate_appends_to_log_on_every_call(monkeypatch):
    sensor = _sensor_with_stream(monkeypatch, _still(10, 2, 100) + _still(12, 6, 100, seed=1))

    sensor.calibrate()
    sensor.device.skip_to(100)  # skip to the second posture
    sensor.calibrate()

    assert len(sensor.calibration_log) == 2
    numpy.testing.assert_allclose(sensor.calibration_log[1], [12, 6], atol=0.2)


def test_calibrate_requires_connected_sensor():
    sensor = motion_sensor.Sensor()

    with pytest.raises(RuntimeError, match="not connected"):
        sensor.calibrate()


def test_get_pose_relative_to_calibration(monkeypatch):
    poses = _still(30, 10, 60, jitter=0) + [(75, 25)] * 3
    sensor = _sensor_with_stream(monkeypatch, poses)
    sensor.calibrate()

    sensor.device.skip_to(60)  # participant turned towards the sound
    pose = sensor.get_pose()

    numpy.testing.assert_allclose(pose, [45, 15], atol=1e-6)


def test_get_pose_wraps_calibrated_azimuth(monkeypatch):
    sensor = _sensor_with_stream(monkeypatch, [(-170, 0.5)])
    sensor.pose_offset = numpy.array([170.0, 0.0])

    pose = sensor.get_pose()

    numpy.testing.assert_allclose(pose, [20, 0.5])


def test_get_pose_physics_convention_after_calibration(monkeypatch):
    sensor = _sensor_with_stream(monkeypatch, [(40, 5)])
    sensor.pose_offset = numpy.array([10.0, 0.0])

    pose = sensor.get_pose(convention="physics")

    # 30 deg in psychoacoustic convention -> 330 deg in physics convention
    numpy.testing.assert_allclose(pose, [330, 5])


def test_get_pose_returns_none_without_valid_data(monkeypatch, caplog):
    sensor = _sensor_with_stream(monkeypatch, [(numpy.nan, numpy.nan)])

    with caplog.at_level(logging.WARNING):
        pose = sensor.get_pose(calibrate=False)

    assert pose is None
    assert "Could not detect head pose" in caplog.text


def test_connect_resets_calibration(monkeypatch):
    _patch_connect_dependencies(
        monkeypatch,
        discovered_devices=[("AA:BB:CC:DD", "MetaWear")],
    )
    sensor = motion_sensor.Sensor()
    sensor.pose_offset = numpy.array([1.0, 2.0])

    sensor.connect()

    assert sensor.pose_offset is None
    assert not sensor.is_calibrated


def test_connect_uses_requested_fusion_mode(monkeypatch):
    dependencies = _patch_connect_dependencies(
        monkeypatch,
        discovered_devices=[("AA:BB:CC:DD", "MetaWear")],
    )
    sensor = motion_sensor.Sensor(fusion_mode="IMU_PLUS")

    sensor.connect()

    dependencies.libmetawear.mbl_mw_sensor_fusion_set_mode.assert_called_once_with(
        dependencies.board,
        "IMU_PLUS",
    )


# ---------------------------------------------------------------------------
# Releasing the connection
# ---------------------------------------------------------------------------

def _halt_fixture(monkeypatch, connected=True):
    board = object()
    hardware_device = SimpleNamespace(board=board, is_connected=connected, disconnect=MagicMock())
    libmetawear = MagicMock()
    monkeypatch.setattr(motion_sensor, "libmetawear", libmetawear, raising=False)
    monkeypatch.setattr(motion_sensor, "SensorFusionData",
                        SimpleNamespace(EULER_ANGLE="EULER_ANGLE"), raising=False)
    monkeypatch.setattr(motion_sensor, "time", SimpleNamespace(sleep=MagicMock()))
    sensor = motion_sensor.Sensor()
    sensor.device = SimpleNamespace(device=hardware_device)
    return sensor, hardware_device, libmetawear


def test_halt_releases_device_even_if_a_step_fails(monkeypatch):
    sensor, hardware_device, libmetawear = _halt_fixture(monkeypatch)
    libmetawear.mbl_mw_sensor_fusion_stop.side_effect = RuntimeError("board not responding")

    sensor.halt()

    assert sensor.device is None
    libmetawear.mbl_mw_debug_disconnect.assert_called_once()
    hardware_device.disconnect.assert_called_once_with()


def test_halt_after_lost_connection_only_releases_host_side(monkeypatch):
    sensor, hardware_device, libmetawear = _halt_fixture(monkeypatch, connected=False)

    sensor.halt()

    assert sensor.device is None
    assert libmetawear.mock_calls == []
    hardware_device.disconnect.assert_called_once_with()


def test_halt_is_idempotent(monkeypatch):
    sensor, hardware_device, libmetawear = _halt_fixture(monkeypatch)

    sensor.halt()
    sensor.halt()

    hardware_device.disconnect.assert_called_once_with()


def test_connect_registers_halt_at_exit_and_halt_unregisters(monkeypatch):
    _patch_connect_dependencies(monkeypatch, discovered_devices=[("AA:BB:CC:DD", "MetaWear")])
    fake_atexit = SimpleNamespace(register=MagicMock(), unregister=MagicMock())
    monkeypatch.setattr(motion_sensor, "atexit", fake_atexit)
    sensor = motion_sensor.Sensor()

    sensor.connect()
    fake_atexit.register.assert_called_once_with(sensor.halt)

    sensor.halt()
    fake_atexit.unregister.assert_called_once_with(sensor.halt)


def test_connect_releases_existing_connection_first(monkeypatch):
    _patch_connect_dependencies(monkeypatch, discovered_devices=[("AA:BB:CC:DD", "MetaWear")])
    sensor = motion_sensor.Sensor()
    old_state = object()
    sensor.device = old_state
    halted = []
    monkeypatch.setattr(sensor, "halt", lambda: halted.append(sensor.device) or setattr(sensor, "device", None))

    sensor.connect()

    assert halted == [old_state]
