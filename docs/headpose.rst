Head Pose Estimation
####################

Head tracking with a motion sensor
----------------------------------

Freefield can read the head orientation from an MbientLab MetaMotion (MetaWear) sensor mounted on a headband. The
sensor streams its orientation over Bluetooth LE at ~100 Hz; this is used to record localization responses: the
participant turns the head towards the perceived sound source and presses a button.

The sensor support needs the optional dependency ``mbientlab``:

.. code-block:: bash

  pip install mbientlab

Initialize the setup with ``sensor_tracking=True`` to scan for and connect to the sensor:

.. code-block:: python

  import freefield
  freefield.initialize(setup="dome", default="loctest_freefield", sensor_tracking=True)

Calibration
^^^^^^^^^^^

The sensor reports orientation in its own frame, so it has to be zeroed while the participant faces the reference
direction (0° azimuth, 0° elevation). Do this **before every response** -- the headband can shift during a session,
and without the magnetometer (``IMU_PLUS`` mode) the yaw drifts slowly:

.. code-block:: python

  freefield.calibrate_sensor(led_feedback=True, button_control="processor")

This lights up the LED of the center speaker, waits for the button press and then sets the zero to the *median* head
orientation over a 0.25 s window of distinct sensor updates, once that window is still (median absolute deviation
<= 0.5° on both axes). A single glitched sensor packet therefore cannot become the zero. Right after calibrating, a
few fresh samples must lie within 3° of (0, 0); otherwise the calibration is repeated (up to three times). The call
never blocks for long: if the head does not settle within 1.5 s it calibrates on the last window and logs a warning.
These parameters can be passed as keyword arguments (see :meth:`motion_sensor.Sensor.calibrate`). Use
``button_control=None`` to calibrate immediately if your script handles the prompt itself.

``calibrate_sensor`` returns a dict describing the calibration (raw offset, spread, whether the head was still,
residual of the check, number of attempts). In addition, the raw offset of every calibration is appended to
``freefield.SENSOR.calibration_log``. Because elevation is referenced to gravity, this is the head pitch at every
zero: a step change between trials marks e.g. a refitted headband. It is not checked live -- save it with your data to
audit a session afterwards.

Reading the head pose
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

  freefield.wait_for_button()  # participant points the head at the sound and presses the button
  response = freefield.get_head_pose(method="sensor", convention="psychoacoustics")

returns (azimuth, elevation) in degrees relative to the last calibration. With ``convention="psychoacoustics"``
azimuth lies in [-180°, 180°); with ``"physics"`` it lies in [0°, 360°), counted counter-clockwise. ``None`` is
returned if the sensor connection was lost or no valid data arrived.

Using the sensor without the setup
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`motion_sensor.Sensor` class does not depend on the TDT processors and can be used in any experiment
(e.g. with headphones or VR):

.. code-block:: python

  from freefield.motion_sensor import Sensor

  sensor = Sensor()          # fusion_mode="NDOF" (default) or "IMU_PLUS"
  sensor.connect()
  for trial in trials:
      input("Look at the center and press Enter")
      sensor.calibrate()
      # ... present the stimulus, wait for the response ...
      response = sensor.get_pose()
  sensor.halt()

``fusion_mode="NDOF"`` fuses accelerometer, gyroscope and magnetometer, so yaw does not drift but can be distorted by
strong magnetic fields (e.g. close to loudspeaker magnets). ``"IMU_PLUS"`` ignores the magnetometer; yaw then drifts
slowly, which per-trial calibration compensates. ``get_pose(n_datapoints=n)`` returns the median over ``n`` distinct
sensor updates instead of only the most recent one.
