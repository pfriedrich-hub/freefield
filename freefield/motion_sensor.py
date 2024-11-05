from __future__ import print_function
import time
import numpy
import logging
try:
    import mbientlab.warble
    import mbientlab.metawear
    # from mbientlab.warble import *
    # from mbientlab.metawear import *
except  ModuleNotFoundError:
    mbientlab = None
    logging.warning('Could not import pywin32 - working with TDT device is disabled')

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

class Sensor():
    def __init__(self):
        self.device = None
        self.pose_offset = None

    def connect(self):
        """
        Scan Bluetooth environment for the motion sensor and connect to the device.
        Returns:
            (instance of Sensor): Object for handling the initialized sensor.
        """
        def handler(result):
            devices[result.mac] = result.name
        devices = {}
        BleScanner.set_handler(handler)
        BleScanner.start()
        # make a choice if multiple sensors are found
        while True:
            logging.info("Scanning for motion sensor")
            t_start = time.time()
            while not time.time() > t_start + 2:
                time.sleep(0.1)  # scanning for devices time
            if 'MetaWear' in devices.values():
                mac_list = []
                for idx, device in enumerate(devices.values()):
                    if device == 'MetaWear':
                        mac_list.append(list(devices.keys())[idx])
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
                    return None

        # alternative: search for max 20 seconds and continue if any sensor is found
        # while not 'MetaWear' in devices.values():
        #     time.sleep(0.1)  # scanning for devices time
        #     if time.time() > t_start + 20:
        #         logging.warning("Could not find motion sensor")
        #         return None
        # for idx, device in enumerate(devices.values()):
        #     if device == 'MetaWear':
        #         address = list(devices.keys())[idx]

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
        # setup quaternionA
        libmetawear.mbl_mw_sensor_fusion_set_mode(sensor.device.board, SensorFusionMode.NDOF)
        libmetawear.mbl_mw_sensor_fusion_set_mode(sensor.device.board, SensorFusionMode.IMU_PLUS)
        libmetawear.mbl_mw_sensor_fusion_set_acc_range(sensor.device.board, SensorFusionAccRange._8G)
        libmetawear.mbl_mw_sensor_fusion_set_gyro_range(sensor.device.board, SensorFusionGyroRange._2000DPS)
        libmetawear.mbl_mw_sensor_fusion_write_config(sensor.device.board)
        # get quat signal and subscribe
        signal = libmetawear.mbl_mw_sensor_fusion_get_data_signal(sensor.device.board, SensorFusionData.EULER_ANGLE)
        libmetawear.mbl_mw_datasignal_subscribe(signal, None, sensor.callback)
        # start acc, gyro, mag
        libmetawear.mbl_mw_sensor_fusion_enable_data(sensor.device.board, SensorFusionData.EULER_ANGLE)
        libmetawear.mbl_mw_sensor_fusion_start(sensor.device.board)
        self.device = sensor
        logging.info('Motion sensor connected and running')

    def get_pose(self, n_datapoints=30, calibrate=True, print_pose=False):
        """
        Read orientation in polar angle from the motion sensor.
        Args:
            n_datapoints (int): Number of data points from which an average orientation is calculated.
            calibrate (boolean): Whether to subtract an offset from the orientation.
            print_pose (boolean): If true, continuously print out orientation.

        Returns:
            pose (numpy.ndarray): Sensor orientation in polar angles.
        """
        pose_log = numpy.zeros((n_datapoints, 2))
        n = 0
        while n < n_datapoints:  # filter invalid values
            pose = numpy.array((self.device.pose.yaw, self.device.pose.roll))
            if not any(numpy.isnan(pose)) and all(-180 <= _pose <= 360 for _pose in pose)\
                    and not any(-1e-3 <= _pose <= 1e-3 for _pose in pose):
                if pose[0] > 180:  # todo fix this
                    pose[0] -= 360
                pose_log[n] = pose
                n += 1
            if not self.device.device.is_connected:
                logging.warning('Sensor connection lost! Reconnect? (Y/n)')
                if input().upper() == 'Y':
                    # self.halt()  # probably unnecessary
                    self.connect()
                return None  # break from the loop and return pose=None
        d = numpy.abs(pose_log - numpy.median(pose_log))  # deviation from median
        mdev = numpy.median(d)  # median deviation
        s = d / mdev if mdev else numpy.zeros_like(d)  # factorized mean deviation to detect outliers
        pose = numpy.array((numpy.mean(pose_log[:, 0][(s < 2)[:, 0]]), numpy.mean(pose_log[:, 1][(s < 2)[:, 1]])))
        if print_pose:
            if all(pose):
                logging.debug('head pose: azimuth: %.1f, elevation: %.1f' % (pose[0], pose[1]))
            else:
                logging.warning("Could not detect head pose")
        if calibrate is True:
            if self.pose_offset is None:
                logging.warning("Device not calibrated")
            else:
                if all(pose):
                    pose = pose - self.pose_offset
                else:
                    logging.warning("Could not detect head pose")
        return pose

    def halt(self):
        """
        Disconnect the motion sensor.
        """
        if self.device:
            libmetawear.mbl_mw_sensor_fusion_stop(self.device.device.board)
            # unsubscribe to signal
            signal = libmetawear.mbl_mw_sensor_fusion_get_data_signal(self.device.device.board, SensorFusionData.EULER_ANGLE);
            libmetawear.mbl_mw_datasignal_unsubscribe(signal)
            # disconnect
            libmetawear.mbl_mw_debug_disconnect(self.device.device.board)
            # while not self.device.device.is_connected:
            time.sleep(0.5)
            self.device.device.disconnect()
            self.device = None
            logging.info('Motion sensor disconnected')




