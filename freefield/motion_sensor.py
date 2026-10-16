"""
Head tracking with an MbientLab MetaMotion (MetaWear) sensor.

The sensor streams Euler angles over Bluetooth LE (~100 Hz). Every packet is
stored in a `State` object together with a running packet counter, so the
readout functions can tell a *new* sensor update from a re-read of the last
cached one.

Typical use for a localization response::

    sensor = Sensor()
    sensor.connect()
    # participant looks at the (0, 0) reference, then:
    sensor.calibrate()          # zero = median over a still 0.25 s window
    # ... stimulus, participant turns head and presses a button ...
    response = sensor.get_pose()  # (azimuth, elevation) relative to the zero
    sensor.halt()

Within `freefield`, use `freefield.initialize(..., sensor_tracking=True)`,
`freefield.calibrate_sensor()` and `freefield.get_head_pose()` instead.
"""
from __future__ import print_function
import atexit
import time
import numpy
import logging
try:
    from mbientlab.warble import *
    from mbientlab.metawear import *
except ModuleNotFoundError:
    mbientlab = None
    logging.warning('Could not import mbientlab - working with motion sensor is disabled')


def _wrap180(angle):
    """Wrap an angle (deg) to [-180, 180)."""
    return (angle + 180) % 360 - 180


class State:
    def __init__(self, device):
        self.device = device
        self.samples = 0
        self.callback = FnVoid_VoidP_DataP(self.data_handler)
        self.pose = None

    # callback
    def data_handler(self, ctx, data):
        self.pose = parse_value(data)
        self.samples += 1


class Sensor:
    """
    MetaMotion head tracker.

    Args:
        fusion_mode (str): Sensor fusion mode set on `connect()`, see
            `set_fusion_mode`. 'NDOF' (default) also uses the magnetometer, so
            yaw is drift-free but absolute; 'IMU_PLUS' (accelerometer + gyro)
            gives a relative yaw that drifts slowly. Pitch/roll -- and thus the
            elevation reading -- are referenced to gravity in both modes.

    Attributes:
        device (State | None): Connected sensor state, None if not connected.
        pose_offset (numpy.ndarray | None): Raw (azimuth, elevation) that is
            subtracted by `get_pose(calibrate=True)`; set by `calibrate()`.
        calibration_log (list): Raw [azimuth, elevation] of every `calibrate()`
            call since the sensor was created, in order (bookkeeping only, see
            `calibrate`).
        last_calibration (dict | None): Details of the most recent calibration.
    """

    def __init__(self, fusion_mode='NDOF'):
        self.device = None
        self.fusion_mode = fusion_mode
        self.pose_offset = None
        self.calibration_log = []
        self.last_calibration = None

    @property
    def is_calibrated(self):
        return self.pose_offset is not None

    def connect(self, scan_s=5):
        """
        Scan the Bluetooth environment for the motion sensor and connect to the device.
        Args:
            scan_s (float): Duration of one scan for devices in seconds.
        Returns:
            None. The connected sensor is stored in `self.device`; it stays None
            if no sensor was found and the user declined to scan again.
        """
        if self.device is not None:  # release an existing connection, else the sensor is not advertising
            self.halt()

        def handler(result):
            devices[result.mac] = result.name
        devices = {}
        BleScanner.set_handler(handler)
        BleScanner.start()
        # make a choice if multiple sensors are found
        while True:
            logging.info("Scanning for motion sensor")
            t_start = time.time()
            while not time.time() > t_start + scan_s:
                time.sleep(0.1)  # scanning for devices time
            if 'MetaWear' in devices.values():
                mac_list = [mac for mac, name in devices.items() if name == 'MetaWear']
                if len(mac_list) > 1:
                    logging.warning('More than one motion sensor detected.\nChoose a sensor:')
                    for idx, mac_id in enumerate(mac_list):
                        print(f'{idx} {mac_id}\n')
                    address = mac_list[int(input())]
                else:
                    address = mac_list[0]
                break
            else:
                logging.warning("Could not find motion sensor. Retry? (Y/n)")
                if input().upper() == 'Y':
                    continue
                else:
                    BleScanner.stop()
                    return None

        BleScanner.stop()
        logging.info("Connecting to motion sensor (MAC: %s)" % (address))
        device = MetaWear(address)
        while not device.is_connected:
            try:
                device.connect()
            except:
                logging.debug('Connecting to motion sensor')
        sensor = (State(device))
        logging.debug("Configuring motion sensor")
        # setup ble
        libmetawear.mbl_mw_settings_set_connection_parameters(sensor.device.board, 7.5, 7.5, 0, 6000)
        # sensor fusion: NDOF = accelerometer + gyro + magnetometer, IMU_PLUS = accelerometer + gyro
        libmetawear.mbl_mw_sensor_fusion_set_mode(sensor.device.board,
                                                  getattr(SensorFusionMode, self.fusion_mode.upper()))
        libmetawear.mbl_mw_sensor_fusion_set_acc_range(sensor.device.board, SensorFusionAccRange._8G)
        libmetawear.mbl_mw_sensor_fusion_set_gyro_range(sensor.device.board, SensorFusionGyroRange._2000DPS)
        libmetawear.mbl_mw_sensor_fusion_write_config(sensor.device.board)
        # get euler angle signal and subscribe
        signal = libmetawear.mbl_mw_sensor_fusion_get_data_signal(sensor.device.board, SensorFusionData.EULER_ANGLE)
        libmetawear.mbl_mw_datasignal_subscribe(signal, None, sensor.callback)
        # start acc, gyro, mag
        libmetawear.mbl_mw_sensor_fusion_enable_data(sensor.device.board, SensorFusionData.EULER_ANGLE)
        libmetawear.mbl_mw_sensor_fusion_start(sensor.device.board)
        self.device = sensor
        # disconnect when the interpreter exits, even if the script never calls halt()
        atexit.register(self.halt)
        # a (re)connected sensor may report in a different frame: force a new calibration
        self.pose_offset = None
        logging.info('Motion sensor connected and running')

    # ------------------------------------------------------------------
    # readout
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(pose):
        """Reject NaN, out-of-range and exact-zero (uninitialized) values."""
        return not any(numpy.isnan(pose)) and all(1e-6 <= abs(_pose) <= 360 for _pose in pose)

    def _read_raw(self):
        """Latest sensor packet as raw (azimuth, elevation) in deg, azimuth
        wrapped to [-180, 180). None if no packet arrived yet or it is invalid."""
        p = self.device.pose
        if p is None:
            return None
        pose = numpy.array((p.yaw, p.roll), dtype=float)
        if not self._validate(pose):
            return None
        pose[0] = _wrap180(pose[0])
        return pose

    def _wait_new_sample(self, last, timeout=0.5):
        """Block until the sensor delivers a packet after count `last`. Returns
        the new count, or None if nothing arrived within `timeout` s."""
        t_end = time.time() + timeout
        while self.device.samples == last:
            if time.time() > t_end:
                return None
            time.sleep(0.001)
        return self.device.samples

    def _collect(self, n, timeout=0.5):
        """Up to `n` valid raw poses from DISTINCT sensor packets, starting with
        the current one. Fewer if the sensor stalls for `timeout` s."""
        poses = []
        last = self.device.samples
        p = self._read_raw()
        while True:
            if p is not None:
                poses.append(p)
                if len(poses) >= n:
                    break
            nxt = self._wait_new_sample(last, timeout)
            if nxt is None:
                break
            last = nxt
            p = self._read_raw()
        return poses

    @staticmethod
    def _robust_center(az, el):
        """Median and median absolute deviation of azimuth (wrap-safe around
        +-180) and elevation. Returns (az, el, mad_az, mad_el)."""
        az, el = numpy.asarray(az, float), numpy.asarray(el, float)
        ref = az[-1]
        d = _wrap180(az - ref)
        med_d = numpy.median(d)
        az_c = _wrap180(ref + med_d)
        el_c = numpy.median(el)
        return (float(az_c), float(el_c),
                float(numpy.median(numpy.abs(d - med_d))),
                float(numpy.median(numpy.abs(el - el_c))))

    def get_pose(self, n_datapoints=1, calibrate=True, print_pose=False, convention='psychoacoustics',
                 timeout=0.5):
        """
        Read the head orientation from the motion sensor.
        Args:
            n_datapoints (int): Number of distinct sensor packets (~10 ms apart) over which the median
                orientation is taken. 1 returns the most recent packet.
            calibrate (boolean): Whether to subtract the offset set by `calibrate()`.
            print_pose (boolean): If true, log the orientation.
            convention (str): Convention of the spherical coordinate system. Can be 'physics'
                (azimuth in [0, 360), counter-clockwise) or 'psychoacoustics' (azimuth in [-180, 180)).
            timeout (float): Give up if the sensor delivers no new packet for this many seconds.
        Returns:
            pose (numpy.ndarray | None): (azimuth, elevation) in degrees, None if the connection was lost or
            the sensor delivered no valid data.
        """
        if convention not in ('psychoacoustics', 'physics'):
            raise ValueError('Convention must be "psychoacoustics" or "physics"!')
        if not self.device.device.is_connected:
            logging.warning('Sensor connection lost! Reconnect? (Y/n)')
            if input().upper() == 'Y':
                self.connect()
            return None  # the caller should recalibrate and read again
        poses = self._collect(n_datapoints, timeout)
        if not poses:
            logging.warning("Could not detect head pose (no valid sensor data)")
            return None
        az, el, _, _ = self._robust_center([p[0] for p in poses], [p[1] for p in poses])
        pose = numpy.array((az, el))
        if calibrate:
            if not self.is_calibrated:
                logging.warning("Device not calibrated")
            else:
                pose = self._apply_offset(pose)
        if convention == 'physics':
            pose[0] = (-pose[0]) % 360
        if print_pose:
            logging.info('head pose: azimuth: %.1f, elevation: %.1f' % (pose[0], pose[1]))
        return pose

    def _apply_offset(self, pose):
        """Raw (psychoacoustic) pose -> pose relative to `pose_offset`."""
        return numpy.array((_wrap180(pose[0] - self.pose_offset[0]),
                            pose[1] - self.pose_offset[1]))

    # ------------------------------------------------------------------
    # calibration
    # ------------------------------------------------------------------

    def calibrate(self, window_s=0.25, stable_mad=0.5, timeout_s=1.5,
                  check_n=5, check_tol=3.0, max_attempts=3):
        """
        Set the 0 deg azimuth / 0 deg elevation offset to the current head pose.
        Call this while the participant faces the reference direction (e.g. right
        after the button press that confirms they look at the center speaker).

        The offset is the MEDIAN over `window_s` seconds of DISTINCT sensor updates,
        taken once that window is still: median absolute deviation <= `stable_mad`
        deg on both axes. A single glitched packet therefore cannot become the zero.

        After setting the offset, `check_n` fresh calibrated samples must have a
        median within `check_tol` deg of (0, 0); otherwise it re-calibrates, up
        to `max_attempts` times. Never blocks: if the head never settles within
        `timeout_s`, or every attempt fails the check, it keeps the last result
        and logs a warning.

        Every call appends the raw offset to `calibration_log`. With the
        elevation referenced to gravity, this is the head pitch at each zero;
        a step change between consecutive trials marks e.g. a headband refit.
        It is not checked live -- store it with your data to audit afterwards.

        Args:
            window_s (float): Duration of the window over which the offset is taken.
            stable_mad (float): Max. median absolute deviation (deg) for the window to count as still.
            timeout_s (float): Max. time to wait for a still window.
            check_n (int): Number of samples for the post-calibration check.
            check_tol (float): Max. distance (deg) from (0, 0) in the post-calibration check.
            max_attempts (int): Max. number of calibration attempts.
        Returns:
            dict: offset_az, offset_el (raw sensor frame, deg), spread_az, spread_el (MAD, deg),
            stable (bool), n_samples, residual (deg, from the check, None if no data), attempts.
            Also stored in `self.last_calibration`.
        """
        if self.device is None:
            raise RuntimeError("Motion sensor not connected. Call connect() first.")
        info = None
        for attempt in range(1, max_attempts + 1):
            az, el, mad_az, mad_el, stable, n = self._measure_offset(
                window_s, stable_mad, timeout_s)
            self.pose_offset = numpy.array((az, el), dtype=float)
            residual = self._check_zero(check_n)
            info = dict(offset_az=az, offset_el=el, spread_az=mad_az,
                        spread_el=mad_el, stable=bool(stable), n_samples=int(n),
                        residual=residual, attempts=attempt)
            if residual is not None and residual <= check_tol:
                break
            logging.warning('Calibration check failed (attempt %d/%d): pose %s deg '
                            'from zero right after calibrating -- recalibrating',
                            attempt, max_attempts,
                            'unknown' if residual is None else '%.1f' % residual)
        else:
            logging.warning('Calibration check still failing after %d attempts; '
                            'keeping the last offset', max_attempts)
        if not info['stable']:
            logging.warning('Head did not settle within %.1f s; calibrated on the '
                            'last %.2f s anyway (spread az %.1f, el %.1f deg)',
                            timeout_s, window_s, info['spread_az'], info['spread_el'])
        self.last_calibration = info
        self.calibration_log.append([float(info['offset_az']), float(info['offset_el'])])
        logging.debug('Sensor calibration complete: %s', info)
        return info

    def _measure_offset(self, window_s, stable_mad, timeout_s):
        """Sliding window of distinct raw samples until it is still (or timeout)."""
        buf = []                       # (t, az, el), raw
        t_stop = time.time() + timeout_s
        last = self.device.samples
        stable = False
        while True:
            nxt = self._wait_new_sample(last)
            if nxt is None:
                logging.warning('Motion sensor delivered no new samples for 0.5 s '
                                'during calibration')
                if time.time() > t_stop:
                    break
                continue
            last = nxt
            p = self._read_raw()
            if p is None:
                continue
            now = time.time()
            buf.append((now, float(p[0]), float(p[1])))
            while buf[0][0] < now - window_s:
                buf.pop(0)
            if len(buf) >= 5 and now - buf[0][0] >= 0.8 * window_s:
                _, _, mad_az, mad_el = self._robust_center(
                    [b[1] for b in buf], [b[2] for b in buf])
                if mad_az <= stable_mad and mad_el <= stable_mad:
                    stable = True
                    break
            if now > t_stop:
                break
        if not buf:  # sensor silent the whole time: fall back to the cached packet
            p = self._read_raw()
            if p is None:
                raise RuntimeError('Motion sensor delivered no valid data; cannot calibrate.')
            buf = [(time.time(), float(p[0]), float(p[1]))]
        az, el, mad_az, mad_el = self._robust_center(
            [b[1] for b in buf], [b[2] for b in buf])
        return az, el, mad_az, mad_el, stable, len(buf)

    def _check_zero(self, n):
        """Median distance (deg) of `n` fresh calibrated samples from (0, 0);
        None if the sensor delivered nothing."""
        az, el = [], []
        last = self.device.samples
        while len(az) < n:
            nxt = self._wait_new_sample(last)
            if nxt is None:
                break
            last = nxt
            p = self._read_raw()
            if p is None:
                continue
            p = self._apply_offset(p)
            az.append(float(p[0]))
            el.append(float(p[1]))
        if not az:
            return None
        return float(numpy.hypot(numpy.median(az), numpy.median(el)))

    # ------------------------------------------------------------------
    # device control
    # ------------------------------------------------------------------

    def halt(self):
        """
        Stop streaming and disconnect the motion sensor. Safe to call repeatedly and on a sensor whose connection is
        already lost: every step is attempted, and `self.device` is released even if a step fails. Called automatically
        when the Python interpreter exits.
        """
        if self.device is None:
            return
        state, self.device = self.device, None
        try:
            atexit.unregister(self.halt)
        except Exception:
            pass
        board = state.device.board

        def _try(step, *args):
            try:
                step(*args)
            except Exception as exc:
                logging.warning('Motion sensor halt: %s failed (%s)', getattr(step, '__name__', step), exc)

        if state.device.is_connected:
            _try(libmetawear.mbl_mw_sensor_fusion_stop, board)
            # unsubscribe to signal
            try:
                signal = libmetawear.mbl_mw_sensor_fusion_get_data_signal(board, SensorFusionData.EULER_ANGLE)
                _try(libmetawear.mbl_mw_datasignal_unsubscribe, signal)
            except Exception as exc:
                logging.warning('Motion sensor halt: unsubscribing failed (%s)', exc)
            # ask the board to drop the link
            _try(libmetawear.mbl_mw_debug_disconnect, board)
            time.sleep(0.5)
        # release the connection on the host side
        _try(state.device.disconnect)
        logging.info('Motion sensor disconnected')

    def set_fusion_mode(self, fusion_mode):
        """
        Change the fusion mode of the sensor. Recalibrate afterwards.
        Arguments:
            fusion_mode (str): Fusion mode of the sensor.
            NDoF: Calculates absolute orientation from accelerometer, gyro, and magnetometer
            IMU_Plus: Calculates relative orientation in space from accelerometer and gyro data
            Compass: Determines geographic direction from th Earth’s magnetic field
            M4G: Similar to IMUPlus except rotation is detected with the magnetometer
        """
        mode = getattr(SensorFusionMode, fusion_mode.upper())
        libmetawear.mbl_mw_sensor_fusion_set_mode(self.device.device.board, mode)
        libmetawear.mbl_mw_sensor_fusion_write_config(self.device.device.board)
        self.fusion_mode = fusion_mode
        logging.info(f'Sensor fusion mode set to {fusion_mode}')
