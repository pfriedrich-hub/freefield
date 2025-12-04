import copy
import time
from pathlib import Path
import datetime
from copy import deepcopy
import pickle
from dataclasses import dataclass
import logging
import numpy as np
import slab
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from freefield import DIR, Processors, cameras, motion_sensor

logging.basicConfig(level=logging.INFO)
slab.Signal.set_default_samplerate(48828)  # default samplerate for generating sounds, filters etc.
# Initialize global variables:
CAMERAS = cameras.Cameras()
PROCESSORS = Processors()
SENSOR = motion_sensor.Sensor()
SPEAKERS = []  # list of all the loudspeakers in the active setup
SETUP = ""  # the currently active setup - "dome" or "arc"

def initialize(setup, default=None, device=None, zbus=True, connection="GB", camera=None, sensor_tracking=False,
               calibration_file=None):
    """
    Initialize the device and load table (and calibration) for the selected setup. Once initialized,
    the setup runs until `halt()` is called. Initialzing device which are already running will flush them.

    Arguments:
        setup (str): which setup to load, can be 'dome', 'arc' or 'headphones'
        default (str | None): initialize the setup using one of the default settings which are:
            'play_rec': play sounds using two RX8s and record them with a RP2
            'play_birec': same as 'play_rec' but record from two microphone channels
            'loctest_freefield': sound localization test under freefield conditions
            'loctest_headphones': localization test with headphones
            'cam_calibration': calibrate cameras for headpose estimation
        device (list | None): A list which contains the name given to the device, it's model and the path to the
            .rcx file to be loaded. To initialize multiple devices at once, pass a list of lists where each sub-list
             contains [name, model, file] for one device.
        zbus (bool): whether or not to initialize the zbus interface for sending triggers.
        connection (str): type of connection to device, can be "GB" (optical) or "USB"
        camera (str | None): kind of camera that is initialized, can be "webcam", "flir".
        sensor_tracking (boolean): If True, initialize head tracking sensor.
    Examples:
        >>> from freefield import initialize, DIR
        >>> # initialize the dome setup with one RX8 processor along with the FLIR cameras:
        >>> initialize(setup="dome", device=['RX8', 'RX8', 'play_buf.rcx'], camera="flir")
        >>> # initialize the arc with two RX8's which are names "RX81" and "RX82":
        >>> initialize(setup="arc", device=[['RX81', 'RX8', 'play_buf.rcx'], ['RX82', 'RX8', 'play_buf.rcx']])
        >>> # use the default settings for a freefield localization test:
        >>> initialize(setup="dome", default="loctest_freefield")
    """
    global PROCESSORS, CAMERAS, SETUP, SPEAKERS, SENSOR
    # initialize device
    SETUP = setup
    # if bool(device) == bool(default):
    #     raise ValueError("You have to specify a device OR a default_mode")
    if device is not None:
        PROCESSORS.initialize(device, zbus, connection)
    elif default is not None:
        PROCESSORS.initialize_default(default)
    if camera is not None:
        CAMERAS = cameras.initialize('flir')
    if sensor_tracking:
        SENSOR.connect()
    SPEAKERS = read_speaker_table()  # load the table containing the information about the loudspeakers
    try:
        load_equalization(calibration_file)  # load the default equalization
    except FileNotFoundError:
        logging.warning("Could not load loudspeaker equalization! Use 'load_equalization' or 'equalize_speakers' \n"
              "to load an existing equalization or measure and compute a new one.")


@dataclass
class Speaker:
    """
    Class for handling the loudspeakers which are usually loaded using `read_speaker_table().`.
    """
    index: int  # the index number of the speaker
    analog_channel: int  # the number of the analog channel to which the speaker is attached
    analog_proc: str  # the processor to whose analog I/O the speaker is attached
    digital_proc: str  # the processor to whose digital I/O the speaker's LED is attached
    azimuth: float  # the azimuth angle of the speaker
    elevation: float  # the azimuth angle of the speaker
    digital_channel: int  # the int value of the bitmask for the digital channel to which the speakers LED is attached
    level: float = None  # the constant for level equalization
    filter: slab.Filter = None  # filter for equalizing the filters transfer function

    def __repr__(self):
        if (self.level is None) and (self.filter is None):
            calibrated = "NOT calibrated"
        else:
            calibrated = "calibrated"
        return f"<speaker {self.index} at azimuth {self.azimuth} and elevation {self.elevation}, {calibrated}>"


def read_speaker_table():
    """
    Read table containing loudspeaker information from a file and initialize a `Speaker` instance for each entry.

    Returns:
        (list): a list of instances of the `Speaker` class.
    """
    speakers = []
    table_file = DIR / 'data' / 'tables' / Path(f'speakertable_{SETUP}.txt')
    table = np.loadtxt(table_file, skiprows=1, delimiter=",", dtype=str)
    for row in table:
        speakers.append(Speaker(index=int(row[0]), analog_channel=int(row[1]), analog_proc=row[2],
                                azimuth=float(row[3]), digital_channel=int(row[5]) if row[5] else None,
                                elevation=float(row[4]), digital_proc=row[6] if row[6] else None))
    return speakers


def load_equalization(file=None):
    """
    Load a loudspeaker equalization from a pickle file and set the `level` and `filter` attribute of each speaker
        in the global speakers list.

    Arguments:
        file (str | None): Path to the pickle file storing the equalization. If None, the function will
            try to load the equalization from the default file.
    """
    if file is None:
        file = DIR / 'data' / f'calibration_{SETUP}.pkl'
    else:
        file = Path(file)
    if file.exists():
        with open(file, "rb") as f:
            equalization = pickle.load(f)
        for index in equalization.keys():
            speaker = pick_speakers(picks=int(index))[0]
            speaker.level = equalization[index]["level"]
            speaker.filter = equalization[index]["filter"]
    else:
        raise FileNotFoundError(f"Could not load equalization file {file}!")


# Wrappers for Processor operations read, write, trigger and halt:
def write(tag, value, processors):
    """
    Write data to processor(s) by setting a `tag` on one or multiple device to a given value.
    The same tag can be set to the same value on multiple device by passing a list of names.

    Arguments:
        tag (str): Name of the tag in the .rcx file where the `value` is written to.
        value (int | float | array) : Value that is written to the tag. If an array, it must be one dimensional.
            The data type of the value must match the tag in the .rcx file, otherwise this function will fail without
            raising an error.
        processors (str | list) : string (or list of strings) with the name(s) of the processor(s) to write to.

    Examples:
        import freefield
        freefield.initialize(setup="dome", default="play_rec")
        write('data', 1000, ['RX81', 'RX82']) # set the value of tag 'playbuflen' on RX81 & RX82 to 1000
        import numpy
        data = numpy.random.randn(1000)
        write('data', data, "RX81") # write data array to the tag 'data' on the RX81
    """
    PROCESSORS.write(tag, value, processors)


def read(tag, processor, n_samples=1):
    """
    Read data from the tag of one processor.

    Args:
        tag (str): Name of the tag in the .rcx file where data is read from.
        processor (str) : Name of the processor to read from.
        n_samples (int): Number of samples to read. Any data exceeding this will be ignored.

    Returns:
        (int, float, list): data read from the tag
    """
    value = PROCESSORS.read(tag, processor, n_samples)
    return value


def play(kind='zBusA', proc=None):
    """
    Use software or the zBus-interface (must be initialized) to send trigger(s) to the processor(s). The zBus
    triggers are sent to all device and ensure simultaneous triggering. For the software triggers, once has to
    specify the processor(s).

    Args:
        kind (str, int): Trigger to use. Possible strings are 'zBusA' or 'zBusB' which uses one of the two zBus
            options to trigger all devices simultaneously (the zBus system must be initialized). Integers can be in the
            range 1 - 10 and refer to software triggers for a single processor.
        proc (None, str): Processor to trigger. Only needed if a software trigger is used
        """
    PROCESSORS.trigger(kind=kind, proc=proc)

def halt():
    """
    Halt all devices in the setup, data stored in the working memory of the processors or cameras will be lost.
    """
    PROCESSORS.halt()
    CAMERAS.halt()
    SENSOR.halt()


def wait_to_finish_playing(proc="all", tag="playback"):
    """
    Busy wait until the device finished playing.

    For this function to work, the rcx-circuit must have a tag that is 1
    while output is generated and 0 otherwise. The default name for this
    kind of tag is "playback". "playback" is read repeatedly for each device
    followed by a short sleep if the value is 1.

    Args:
        proc (str, list of str): name(s) of the processor(s) to wait for.
        tag (str): name of the tag that signals if something is played
    """
    if proc == "all":
        proc = list(PROCESSORS.processors.keys())
    elif isinstance(proc, str):
        proc = [proc]
    logging.debug(f'Waiting for {tag} on {proc}.')
    while any(PROCESSORS.read(tag, n_samples=1, proc=p) for p in proc):
        time.sleep(0.01)
    logging.debug('Done waiting.')


def wait_for_button(proc="RP2", tag="response"):
    """
    Busy wait until the response button was pressed. Repeatedly read a tag from a processor and do a busy wait while
        0 is returned.

    Args:
        proc (str): Processor from which the tag is read.
        tag (str): Tag which is read.

    """
    while not PROCESSORS.read(tag=tag, proc=proc):
        time.sleep(0.1)  # wait until button is pressed


def pick_speakers(picks):
    """
    Either return the speaker at given coordinates (azimuth, elevation) or the
    speaker with a specific index number.

    Args:
        picks (list of lists, list, int): index number of the speaker

    Returns:
        (list):
    """
    if isinstance(picks, (list, np.ndarray)):
        if all(isinstance(p, Speaker) for p in picks):
            speakers = picks
        elif all(isinstance(p, (int, np.int64, np.int32)) for p in picks):
            speakers = [s for s in SPEAKERS if s.index in picks]
        else:
            speakers = [s for s in SPEAKERS if (s.azimuth, s.elevation) in picks]
    elif isinstance(picks, (int, np.int64, np.int32)):
        speakers = [s for s in SPEAKERS if s.index == picks]
    elif isinstance(picks, Speaker):
        speakers = [picks]
    else:
        speakers = [s for s in SPEAKERS if (s.azimuth == picks[0] and s.elevation == picks[1])]
    if len(speakers) == 0:
        print("no speaker found that matches the criterion - returning empty list")
    return speakers


def all_leds():
    # Temporary hack: return all speakers from the table which have a LED attached
    return [s for s in SPEAKERS if s.digital_channel is not None]


def shift_setup(delta_azi, delta_ele):
    """
    Shift the setup (relative to the lister) by adding some delta value
    in azimuth and elevation. This can be used when chaning the position of
    the chair where the listener is sitting - moving the chair to the right
    is equivalent to shifting the setup to the left. Changes are not saved to
    the speaker table.

    Args:
        delta_azi (float): azimuth by which the setup is shifted, positive value means shifting right
        delta_ele (float): elevation by which the setup is shifted, positive value means shifting up
    """
    # TODO: first convert to cartesian coordinates then move
    global SPEAKERS
    for speaker in SPEAKERS:
        speaker.azimuth += delta_azi  # azimuth
        speaker.elevation += delta_ele  # elevation
    print(f"shifting the loudspeaker array by {delta_azi} in azimuth and {delta_ele} in elevation")


def set_signal_and_speaker(signal, speaker, equalize=True, data_tag='data', chan_tag='chan', n_samples_tag='playbuflen'):
    """
    Load a signal into the processor buffer and set the output channel to match the speaker.
    The processor is chosen automatically depending on the speaker.

        Args:
            signal (array-like): signal to load to the buffer, must be one-dimensional
            speaker (Speaker, int) : speaker to play the signal from, can be index number or [azimuth, elevation]
            equalize (bool): if True (=default) apply loudspeaker equalization
            data_tag ('string'): Name of the tag feeding into the signal buffer
            chan_tag ('string'): Name of the tag setting the output channel number
            play_tag ('string'): Name of the tag connected to the playback switch
    """
    signal = slab.Sound(signal)
    speaker = pick_speakers(speaker)[0]
    if equalize:
        logging.info('Applying calibration.')  # apply level and frequency calibration
        to_play = apply_equalization(signal, speaker)
    else:
        to_play = signal
    PROCESSORS.write(tag=n_samples_tag, value=to_play.n_samples, processors=['RX81', 'RX82'])
    PROCESSORS.write(tag=chan_tag, value=speaker.analog_channel, processors=speaker.analog_proc)
    PROCESSORS.write(tag=data_tag, value=to_play.data, processors=speaker.analog_proc)
    other_procs = set([s.analog_proc for s in SPEAKERS])
    other_procs.remove(speaker.analog_proc)  # set the analog output of other processors to non existent number 99
    PROCESSORS.write(tag=chan_tag, value=99, processors=other_procs)


def set_signal_headphones(signal, speaker, equalize=True, data_tags=['data_l', 'data_r'], chan_tags=['chan_l', 'chan_r'],
                          n_samples_tag='playbuflen'):
    """
    Load a signal into the processor buffer and set the output channels to headphones.

        Args:
            speaker (string): A string specifying the headphone speakers to play from.
                Can be 'left', 'right', or 'both'.
            signal (array-like): signal to load to the buffer, must be one-dimensional
            equalize (bool): if True (=default) apply loudspeaker equalization
            data_tags (List): A list containing the names of the tags feeding into the signal buffers
            chan_tags (List): A list containing the names of the tags setting the output channel numbers
            play_tag ('string'): Name of the tag connected to the playback switch
    """
    speakers = SPEAKERS
    if speaker == 'both':
        idx = slice(0, 2)
        if signal.n_channels == 1:
            signal = slab.Binaural(signal)
    if speaker == 'left':
        idx = slice(0, 1)
    if speaker == 'right':
        idx = slice(1, 2)
    to_play = copy.deepcopy(signal)
    PROCESSORS.write(tag=n_samples_tag, value=signal.n_samples, processors='RP2')
    for i, (speaker, ch_tag, data_tag) in enumerate(zip(speakers[idx], chan_tags[idx], data_tags[idx])):
        if equalize:
            logging.info('Applying calibration.')  # apply level and frequency calibration
            to_play = apply_equalization(signal=signal.channel(i), speaker=i).data
        elif not equalize:
            to_play = signal.channel(i).data
        PROCESSORS.write(tag=ch_tag, value=speaker.analog_channel, processors=speaker.analog_proc)
        PROCESSORS.write(tag=data_tag, value=to_play, processors=speaker.analog_proc)

def set_speaker(speaker):
    """
    Set the analog channel on the processor corresponding to the selected speaker
    Args:
        speaker: the speaker to be selected
    Returns:
        None
    """
    speaker = pick_speakers(speaker)[0]
    PROCESSORS.write(tag='chan', value=speaker.analog_channel, processors=speaker.analog_proc)
    other_procs = set([s.analog_proc for s in SPEAKERS])
    other_procs.remove(speaker.analog_proc)  # set the analog output of other processors to non existent number 99
    PROCESSORS.write(tag='chan', value=99, processors=other_procs)

# def set_speaker(speaker, equalize=True):
#     speaker = pick_speakers(speaker)[0]
#     PROCESSORS.write(tag='chan', value=speaker.analog_channel, processors=speaker.analog_proc)
#     other_procs = set([s.analog_proc for s in SPEAKERS])
#     other_procs.remove(speaker.analog_proc)  # set the analog output of other processors to non existent number 99
#     PROCESSORS.write(tag='chan', value=99, processors=other_procs)

def play_and_record(speaker, sound, compensate_delay=True, compensate_attenuation=False, equalize=True,
                    recording_samplerate=48828):
    """
    Play the signal from a speaker and return the recording. Delay compensation
    means making the buffer of the recording processor n samples longer and then
    throwing the first n samples away when returning the recording so sig and
    rec still have the same length. For this to work, the circuits rec_buf.rcx
    and play_buf.rcx have to be initialized on RP2 and RX8s and the mic must
    be plugged in.
    Parameters:
        speaker: integer between 1 and 48, index number of the speaker
        sound: instance of slab.Sound, signal that is played from the speaker
        compensate_delay: bool, compensate the delay between play and record
        compensate_attenuation:
        equalize:
        recording_samplerate: samplerate of the recording
    Returns:
        rec: 1-D array, recorded signal
    """
    write(tag="playbuflen", value=sound.n_samples, processors=["RX81", "RX82"])
    set_signal_and_speaker(sound, speaker, equalize)
    if compensate_delay:
        n_delay = get_recording_delay(play_from="RX8", rec_from="RP2")
        n_delay += int(.00325 * recording_samplerate)  # empirically tested for 100kHz
    else:
        n_delay = 0
    rec_n_samples = int(sound.duration * recording_samplerate)
    write(tag="playbuflen", value=rec_n_samples + n_delay, processors="RP2")
    play()
    wait_to_finish_playing()
    if PROCESSORS.mode == "play_rec":  # read the data from buffer and skip the first n_delay samples
        rec = read(tag='data', processor='RP2', n_samples=rec_n_samples + n_delay)[n_delay:]
        rec = slab.Sound(rec, samplerate=recording_samplerate)
    elif PROCESSORS.mode == "play_birec":  # read data for left and right ear from buffer
        rec_l = read(tag='datal', processor='RP2', n_samples=rec_n_samples + n_delay)[n_delay:]
        rec_r = read(tag='datar', processor='RP2', n_samples=rec_n_samples + n_delay)[n_delay:]
        rec = slab.Binaural([rec_l, rec_r], samplerate=recording_samplerate)
    else:
        raise ValueError("Setup must be initialized in mode 'play_rec' or 'play_birec'!")
    if sound.samplerate != recording_samplerate:
        rec = rec.resample(sound.samplerate)
    if compensate_attenuation:
        if isinstance(rec, slab.Binaural):
            iid = rec.left.level - rec.right.level
            rec.level = sound.level
            rec.left.level += iid
        else:
            rec.level = sound.level
    return rec

def play_and_record_headphones(speaker, sound, compensate_delay=True, distance=0, compensate_attenuation=False,
                               equalize=True, recording_samplerate=48828):
    """
    Play the signal from a speaker and return the recording. Delay compensation
    means making the buffer of the recording processor n samples longer and then
    throwing the first n samples away when returning the recording so sig and
    rec still have the same length. For this to work, the circuits rec_buf.rcx
    and play_buf.rcx have to be initialized on RP2 and RX8s and the mic must
    be plugged in.
    Parameters:
        speaker (string): A string specifying the headphone speakers to play from.
                Can be 'left', 'right', or 'both'.
        sound: instance of slab.Sound, signal that is played from the speaker
        distance: distance between sound sources and microphone (symmetric)
        compensate_delay: bool, compensate the delay between play and record
        compensate_attenuation:
        equalize:
        recording_samplerate: samplerate of the recording
    Returns:
        rec: 2-D array, recorded signal
    """
    fs_out = sound.samplerate  # original samplerate of input signal
    if sound.samplerate != recording_samplerate:
        sound = sound.resample(recording_samplerate)
    if PROCESSORS.mode != "bi_play_rec":  # read data for left and right ear from buffer
        raise ValueError("Setup must be initialized in mode 'bi_play_rec'.")
    if compensate_delay:
        n_delay = get_recording_delay(play_from="RP2", rec_from="RP2", distance=distance)
        n_delay += int(.0014 * recording_samplerate)  # empirically tested
    else:
        n_delay = 0
    rec_n_samples = int(sound.duration * recording_samplerate)
    write(tag="recbuflen", value=rec_n_samples + n_delay, processors="RP2")
    set_signal_headphones(signal=sound, speaker=speaker, equalize=equalize)
    play()
    wait_to_finish_playing(tag='recording')
    if speaker == 'both':
        rec = slab.Binaural([read(tag='datal', processor='RP2', n_samples=rec_n_samples + n_delay)[n_delay:],
                             read(tag='datar', processor='RP2', n_samples=rec_n_samples + n_delay)[n_delay:]],
                            samplerate=recording_samplerate)
    elif speaker == 'left':
        rec = slab.Sound(read(tag='datal', processor='RP2', n_samples=rec_n_samples + n_delay)[n_delay:],
                         samplerate=recording_samplerate)
    elif speaker == 'right':
        rec = slab.Sound(read(tag='datar', processor='RP2', n_samples=rec_n_samples + n_delay)[n_delay:],
                         samplerate=recording_samplerate)
    if sound.samplerate != recording_samplerate:
        rec = rec.resample(fs_out)
    if compensate_attenuation:
        if isinstance(rec, slab.Binaural):
            iid = rec.left.level - rec.right.level
            rec.level = sound.level
            rec.left.level += iid
        else:
            rec.level = sound.level
    return rec

def get_recording_delay(distance=1.4, sample_rate=48828, play_from=None, rec_from=None):
    """
        Calculate the delay it takes for played sound to be recorded. Depends
        on the distance of the microphone from the speaker and on the device
        digital-to-analog and analog-to-digital conversion delays.

        Args:
            distance (float): distance between listener and speaker array in meters
            sample_rate (int): sample rate under which the system is running
            play_from (str): processor used for digital to analog conversion
            rec_from (str): processor used for analog to digital conversion

    """
    n_sound_traveling = int(distance / 343 * sample_rate)
    if play_from:
        if play_from == "RX8":
            n_da = 24
        elif play_from == "RP2":
            n_da = 30
        else:
            logging.warning(f"dont know D/A-delay for processor type {play_from}...")
            n_da = 0
    else:
        n_da = 0
    if rec_from:
        if rec_from == "RX8":
            n_ad = 47
        elif rec_from == "RP2":
            n_ad = 65
        else:
            logging.warning(f"dont know A/D-delay for processor type {rec_from}...")
            n_ad = 0
    else:
        n_ad = 0
    return n_sound_traveling + n_da + n_ad

def apply_equalization(signal, speaker, level=True, frequency=True):
    """
    Apply level correction and frequency equalization to a signal

    Args:
        signal: signal to calibrate
        speaker: index number, coordinates or row from the speaker table. Determines which calibration is used
        level:
        frequency:
    Returns:
        slab.Sound: calibrated copy of signal
    """
    signal = slab.Sound(signal)
    speaker = pick_speakers(speaker)[0]
    equalized_signal = deepcopy(signal)
    if level:
        if speaker.level is None:
            raise ValueError("speaker not level-equalized! Load an existing equalization of calibrate the setup!")
        equalized_signal.level += speaker.level
    if frequency:
        if speaker.filter is None:
            raise ValueError("speaker not frequency-equalized! Load an existing equalization of calibrate the setup!")
        equalized_signal = speaker.filter.apply(equalized_signal)
    return equalized_signal

def equalize_headphones(bandwidth=1/10, threshold=.3, low_cutoff=100, high_cutoff=16000, alpha=1.0, file_name=None):
    """
       Equalize the headphones in two steps. First: equalize over all
       level differences by a constant for each speaker. Second: remove spectral
       difference by inverse filtering. For more details on how the
       inverse filters are computed see the documentation of slab.Filter.equalizing_filterbank

       Args:
           bandwidth (float): Width of the filters, used to divide the signal into subbands, in octaves. A small
               bandwidth results in a fine tuned transfer function which is useful for equalizing small notches.
           threshold (float): Threshold for level equalization. Correct level only for speakers that deviate more
               than <threshold> dB from reference speaker
           low_cutoff (int | float): The lower limit of frequency equalization range in Hz.
           high_cutoff (int | float): The upper limit of frequency equalization range in Hz.
           alpha (float): Filter regularization parameter. Values below 1.0 reduce the filter's effect, values above
               amplify it. WARNING: large filter gains may result in temporal distortions of the sound
           file_name (string): Name of the file to store equalization parameters.

       """
    global SETUP
    if not PROCESSORS.mode == "bi_play_rec":
        PROCESSORS.initialize_default(mode="bi_play_rec")
        SETUP = 'headphones'
    sound = slab.Binaural.chirp(duration=0.1, level=85, from_frequency=low_cutoff, to_frequency=high_cutoff, kind='linear')
    speakers = SPEAKERS
    # reference_speaker = 'left'
    # don't do level calibration for now
    # temp_recs = []
    # for i in range(20):
    #     rec = play_and_record_headphones(reference_speaker, sound, equalize=False)
    #     temp_recs.append(rec.data)
    # target = slab.Sound(data=np.mean(temp_recs, axis=0))
    # # # use original signal as reference - WARNING could result in unrealistic equalization filters,
    # #  can be used for HRTF measurement calibration to get really flat chirp spectra
    # baseline_amp = target.level
    # target = deepcopy(sound)
    # target.level = baseline_amp
    # temp_recs = []
    # for i in range(20):
    #     rec = play_and_record_headphones(speaker='both', sound=sound, equalize=False)
    #     temp_recs.append(rec.data)
    # rec = slab.Sound(data=np.mean(temp_recs, axis=0))
    #     # recordings.append(numpy.mean(temp_recs, axis=0))
    # rec.data[:, np.logical_and(rec.level > target.level - threshold,
    #                                      rec.level < target.level + threshold)] = target.data
    # equalization_levels = target.level - rec.level
    equalization_levels = [0, 0]
    recordings = []
    attenuated = deepcopy(sound)
    attenuated.level += equalization_levels
    for i in range(20):
        rec = play_and_record_headphones(speaker='both', sound=attenuated, equalize=False)
        recordings.append(rec.data)
    recording = slab.Binaural(data=np.mean(recordings, axis=0))

    filter_bank = slab.Filter.equalizing_filterbank(sound.channel(0), recording, low_cutoff=low_cutoff,
                                                    high_cutoff=high_cutoff, bandwidth=bandwidth, alpha=alpha)
    equalization = {f"{speakers[i].index}": {"level": equalization_levels[i], "filter": filter_bank.channel(i)}
                    for i in range(len(speakers))}
    if file_name is None:  # use the default filename and rename teh existing file
        file_name = DIR / 'data' / f'calibration_{SETUP}.pkl'
    else:
        file_name = DIR / 'data' / f'calibration_{SETUP}_{file_name}.pkl'
    # if file_name.exists():  # move the old calibration to the log folder
        # date = datetime.datetime.now().strftime("_%Y-%m-%d-%H-%M-%S")
        # file_name = file_name.parent / (file_name.stem + date + file_name.suffix)
        # file_name.rename(file_name.parent / (file_name.stem + date + file_name.suffix))
    with open(file_name, 'wb') as f:  # save the newly recorded calibration
        pickle.dump(equalization, f, pickle.HIGHEST_PROTOCOL)
    logging.info(f'Saved equalization to {file_name}')
    return equalization

def equalize_speakers(speakers="all", reference_speaker=23, bandwidth=1 / 10, threshold=.3,
                      low_cutoff=200, high_cutoff=16000, alpha=1.0, file_name=None):
    """
    Equalize the loudspeaker array in two steps. First: equalize over all
    level differences by a constant for each speaker. Second: remove spectral
    difference by inverse filtering. For more details on how the
    inverse filters are computed see the documentation of slab.Filter.equalizing_filterbank

    Args:
        speakers (list, string): Select speakers for equalization. Can be a list of speaker indices or 'all'
        reference_speaker: Select speaker for reference level and frequency response
        bandwidth (float): Width of the filters, used to divide the signal into subbands, in octaves. A small
            bandwidth results in a fine tuned transfer function which is useful for equalizing small notches.
        threshold (float): Threshold for level equalization. Correct level only for speakers that deviate more
            than <threshold> dB from reference speaker
        low_cutoff (int | float): The lower limit of frequency equalization range in Hz.
        high_cutoff (int | float): The upper limit of frequency equalization range in Hz.
        alpha (float): Filter regularization parameter. Values below 1.0 reduce the filter's effect, values above
            amplify it. WARNING: large filter gains may result in temporal distortions of the sound
        file_name (string): Name of the file to store equalization parameters.

    """
    if not PROCESSORS.mode == "play_rec":
        PROCESSORS.initialize_default(mode="play_rec")
    sound = slab.Sound.chirp(duration=0.1, from_frequency=low_cutoff, to_frequency=high_cutoff)
    if speakers == "all":  # use the whole speaker table
        speakers = SPEAKERS
    else:
        speakers = pick_speakers(picks=speakers)
    reference_speaker = pick_speakers(reference_speaker)[0]
    equalization_levels = _level_equalization(speakers, sound, reference_speaker, threshold)
    filter_bank, rec = _frequency_equalization(speakers, sound, reference_speaker, equalization_levels,
                                               bandwidth, low_cutoff, high_cutoff, alpha, threshold)
    equalization = {f"{speakers[i].index}": {"level": equalization_levels[i], "filter": filter_bank.channel(i)}
                    for i in range(len(speakers))}
    if file_name is None:  # use the default filename and rename teh existing file
        file_name = DIR / 'data' / f'calibration_{SETUP}.pkl'
    else:
        file_name = Path(file_name)
    if file_name.exists():  # move the old calibration to the log folder
        date = datetime.datetime.now().strftime("_%Y-%m-%d-%H-%M-%S")
        file_name.rename(file_name.parent / (file_name.stem + date + file_name.suffix))
    with open(file_name, 'wb') as f:  # save the newly recorded calibration
        pickle.dump(equalization, f, pickle.HIGHEST_PROTOCOL)


def _level_equalization(speakers, sound, reference_speaker, threshold):
    """
    Record the signal from each speaker in the list and return the level of each
    speaker relative to the target speaker(target speaker must be in the list)
    """
    target_recording = play_and_record(reference_speaker, sound, equalize=False)
    recordings = []
    for speaker in speakers:
        recordings.append(play_and_record(speaker, sound, equalize=False))
    recordings = slab.Sound(recordings)
    recordings.data[:, np.logical_and(recordings.level > target_recording.level-threshold,
                    recordings.level < target_recording.level+threshold)] = target_recording.data
    equalization_levels = target_recording.level - recordings.level
    recordings.data[:, recordings.level < threshold] = target_recording.data  # thresholding
    return target_recording.level / recordings.level


def _frequency_equalization(speakers, sound, reference_speaker, calibration_levels, bandwidth,
                            low_cutoff, high_cutoff, alpha, threshold):
    """
    play the level-equalized signal, record and compute and a bank of inverse filter
    to equalize each speaker relative to the target one. Return filterbank and recordings
    """
    reference = play_and_record(reference_speaker, sound, equalize=False)
    recordings = []
    for speaker, level in zip(speakers, calibration_levels):
        attenuated = deepcopy(sound)
        attenuated.level += level
        recordings.append(play_and_record(speaker, attenuated, equalize=False))
    recordings = slab.Sound(recordings)
    filter_bank = slab.Filter.equalizing_filterbank(reference, recordings, low_cutoff=low_cutoff,
                                                    high_cutoff=high_cutoff, bandwidth=bandwidth, alpha=alpha)
    # check for notches in the filter:
    transfer_function = filter_bank.tf(show=False)[1][0:900, :]
    if (transfer_function < -30).sum() > 0:
        print("Some of the equalization filters contain deep notches - try adjusting the parameters.")
    return filter_bank, recordings


def test_equalization(speakers="all"):
    """
    Test the effectiveness of the speaker equalization
    """
    if not PROCESSORS.mode == "play_rec":
        PROCESSORS.initialize_default(mode="play_rec")
    not_equalized = slab.Sound.whitenoise(duration=.5)
    # the recordings from the un-equalized, the level equalized and the fully equalized sounds
    rec_raw, rec_level, rec_full = [], [], []
    if speakers == "all":  # use the whole speaker table
        speakers = SPEAKERS
    else:
        speakers = pick_speakers(SPEAKERS)
    for speaker in speakers:
        level_equalized = apply_equalization(not_equalized, speaker=speaker, level=True, frequency=False)
        full_equalized = apply_equalization(not_equalized, speaker=speaker, level=True, frequency=True)
        rec_raw.append(play_and_record(speaker, not_equalized, equalize=False))
        rec_level.append(play_and_record(speaker, level_equalized, equalize=False))
        rec_full.append(play_and_record(speaker, full_equalized, equalize=False))
    return slab.Sound(rec_raw), slab.Sound(rec_level), slab.Sound(rec_full)


def spectral_range(signal, bandwidth=1 / 5, low_cutoff=50, high_cutoff=20000, thresh=3,
                   plot=True, log=True):
    """
    Compute the range of differences in power spectrum for all channels in
    the signal. The signal is devided into bands of equivalent rectangular
    bandwidth (ERB - see More& Glasberg 1982) and the level is computed for
    each frequency band and each channel in the recording. To show the range
    of spectral difference across channels the minimum and maximum levels
    across channels are computed. Can be used for example to check the
    effect of loud speaker equalization.
    """
    # generate ERB-spaced filterbank:
    filter_bank = slab.Filter.cos_filterbank(length=1000, bandwidth=bandwidth,
                                             low_cutoff=low_cutoff, high_cutoff=high_cutoff,
                                             samplerate=signal.samplerate)
    center_freqs, _, _ = slab.Filter._center_freqs(low_cutoff, high_cutoff, bandwidth)
    center_freqs = slab.Filter._erb2freq(center_freqs)
    # create arrays to write data into:
    levels = np.zeros((signal.n_channels, filter_bank.n_channels))
    max_level, min_level = np.zeros(filter_bank.n_channels), np.zeros(filter_bank.n_channels)
    for i in range(signal.n_channels):  # compute ERB levels for each channel
        levels[i] = filter_bank.apply(signal.channel(i)).level
    for i in range(filter_bank.n_channels):  # find max and min for each frequency
        max_level[i] = max(levels[:, i])
        min_level[i] = min(levels[:, i])
    difference = max_level - min_level
    if plot is True or isinstance(plot, Axes):
        if isinstance(plot, Axes):
            ax = plot
        else:
            fig, ax = plt.subplots(1)
        # frequencies where the difference exceeds the threshold
        bads = np.where(difference > thresh)[0]
        for y in [max_level, min_level]:
            if log is True:
                ax.semilogx(center_freqs, y, color="black", linestyle="--")
            else:
                ax.plot(center_freqs, y, color="black", linestyle="--")
        for bad in bads:
            ax.fill_between(center_freqs[bad - 1:bad + 1], max_level[bad - 1:bad + 1],
                            min_level[bad - 1:bad + 1], color="red", alpha=.6)
    return difference


def get_head_pose(method='sensor', convention='psychoacoustics'):
    """
    Wrapper for the get headpose methods of the camera and sensor classes

    Args:
        method (string): Method use for headpose estimation. Can be "camera" or "sensor"
        convention (string): Convention of the spherical coordinate system. Can be 'physics' or 'psychoacoustics'.

    Returns:
        head_pose (numpy.ndarray): Array containing spheric coordinates of
        the current head orientation: (Azimuth, Elevation)
    """
    if method.lower() == 'camera':
        if not CAMERAS.n_cams:
            raise ValueError("No cameras initialized!")
        else:
            azi, ele = CAMERAS.get_head_pose(convert=True, average_axis=(1, 2), n_images=1)
            head_pose = np.array(azi, ele)
    elif method.lower() == 'sensor':
        if not SENSOR.device:
            raise ValueError("No sensor connected!")
        else:
            head_pose = SENSOR.get_pose(convention=convention)
    else:
        raise ValueError("Method must be 'camera' or 'sensor'")
    return head_pose


def check_pose(fix=(0, 0), var=10):
    """
    Check if the head pose is directed towards the fixation point

    Args:
        fix: azimuth and elevation of the fixation point
        var: degrees, the pose is allowed to deviate from the fixation point in azimuth and elevations
    Returns:
        bool: True if difference between pose and fix is smaller than var, False otherwise
    """
    azi, ele = get_head_pose(n_images=1)
    if (azi is np.nan) or (azi is None):
        azi = fix[0]
    if (ele is np.nan) or (ele is None):
        ele = fix[1]
    if np.abs(azi - fix[0]) > var or np.abs(ele - fix[1]) > var:
        return False
    else:
        return True

def get_head_response(method='sensor', proc="RP2", tag="response"):
    """
    Get participants localization response by pointing their head towards the perceived
     sound source and pressing a button.
    Args:
        method (string): Method use for headpose estimation. Can be "camera" or "sensor".
        proc (string): Precssor that reads out the button response
        tag (string): Name of the Tag in the RPvdsEX file connected to the button input
    Returns:
        head_pose (numpy.ndarray): Array containing spheric coordinates of
        the current head orientation: (Azimuth, Elevation)
    """
    response = 0
    while not response:
        pose = get_head_pose(method)
        if all(pose):
            print('head pose: azimuth: %.1f, elevation: %.1f' % (pose[0], pose[1]), end="\r", flush=True)
        else:
            print('no head pose detected', end="\r", flush=True)
        response = read(tag, proc)
    if all(pose):
        print('Response| azimuth: %.1f, elevation: %.1f' % (pose[0], pose[1]))
    return pose

def calibrate_sensor(led_feedback=True, button_control='processor'):
    """
    Calibrate the motion sensor offset to 0° Azimuth and 0° Elevation. A LED will light up to guide head orientation
    towards the center speaker. After a button is pressed, head orientation will be measured until it remains stable.
    The average is then used as an offset for pose estimation.
        Args:
        led_feedback: whether to turn on the central led to assist gaze control during calibration
        button_control (str): whether to initialize calibration by button response; may be 'processor' if a button is
         connected to the RP2 or 'keyboard', if usb keyboard input is to be used.
    Returns:
        bool: True if difference between pose and fix is smaller than var, False otherwise
    """
    log_size = 100
    limit = 0.2
    if led_feedback:
        [led_speaker] = pick_speakers(23)  # get object for center speaker LED
        write(tag='bitmask', value=led_speaker.digital_channel,
              processors=led_speaker.digital_proc)  # illuminate LED
    if button_control == 'processor':
        logging.debug('rest at center speaker and press button to start calibration...')
        wait_for_button()  # start calibration after button press
    elif button_control == 'keyboard':
        input('Rest at center speaker and press button to start calibration...')
    logging.debug('calibrating')
    log = np.zeros(2)
    while True:  # wait in loop for sensor to stabilize
        pose = SENSOR.get_pose(calibrate=False)
        log = np.vstack((log, pose))
        # check if orientation is stable for at least 30 data points
        if len(log) > log_size:
            diff = np.mean(np.abs(np.diff(log[-log_size:], axis=0)), axis=0).astype('float16')
            logging.debug('az diff: %f,  ele diff: %f' % (diff[0], diff[1]))
            if diff[0] < limit and diff[1] < limit:  # limit in degree
                break
    if led_feedback:
        write(tag='bitmask', value=0, processors=led_speaker.digital_proc)  # turn off LED
    SENSOR.pose_offset = np.around(np.mean(log[-int(log_size / 2):].astype('float16'), axis=0), decimals=2)
    logging.debug('Sensor calibration complete.')


def calibrate_camera(speakers, n_reps=1, n_images=5, show=True):
    """
    Calibrate all cameras by lighting up a series of LEDs and estimate the pose when the head is pointed
    towards the currently lit LED. This results in a list of world and camera coordinates which is used to
    calibrate the cameras.

    Args:
        speakers (): rows from the speaker table. The speakers must have a LED attached
        n_reps(int): number of repetitions for each target
        n_images(int): number of images taken for each head pose estimate
    Returns:
        pandas DataFrame: camera and world coordinates acquired (calibration is performed automatically)
    """
    # TODO: save the camera calibration in a temporary directory
    if not PROCESSORS.mode == "cam_calibration":  # initialize setup in camera calibration mode
        PROCESSORS.initialize_default(mode="cam_calibration")
    speakers = pick_speakers(speakers)
    if not all([s.digital_channel for s in speakers]):
        raise ValueError("All speakers must have a LED attached for a test with visual cues")
    seq = slab.Trialsequence(n_reps=n_reps, conditions=speakers)
    world_coordinates = [(seq.conditions[t - 1].azimuth, seq.conditions[t - 1].elevation) for t in seq.trials]
    camera_coordinates = []
    for speaker in seq:
        write(tag="bitmask", value=int(speaker.digital_channel), processors=speaker.digital_proc)
        wait_for_button()
        camera_coordinates.append(CAMERAS.get_head_pose(average_axis=1, convert=False, n_images=n_images))
        write(tag="bitmask", value=0, processors=speaker.digital_proc)
    CAMERAS.calibrate(world_coordinates, camera_coordinates, plot=show)


def calibrate_camera_no_visual(speakers, n_reps=1, n_images=5):
    """
    This is an alteration of calibrate_camera for cases in which LEDs are
    not available. The list of targets is repeated n_reps times in the
    exact same order without any randomization. When the whole setup is
    equipped with LEDs this function should be removed
    """
    if not PROCESSORS.mode == "cam_calibration":
        PROCESSORS.initialize_default(mode="cam_calibration")
    speakers = pick_speakers(speakers)
    camera_coordinates = []
    speakers = speakers * n_reps
    world_coordinates = [(s.azimuth, s.elevation) for s in speakers]
    for _ in speakers:
        wait_for_button()
        camera_coordinates.append(CAMERAS.get_head_pose(average_axis=1, convert=False, n_images=n_images))
    CAMERAS.calibrate(world_coordinates, camera_coordinates, plot=True)


def localization_test_freefield(speakers, duration=0.5, n_reps=1, n_images=5, visual=False):
    """
    Run a basic localization test where the same sound is played from different
    speakers in randomized order, without playing the same position twice in
    a row. After every trial the presentation is paused and the listener has
    to localize the sound source by pointing the head towards the source and
    pressing the response button. The cameras need to be calibrated before the
    test! After every trial the listener has to point to the middle speaker at
    0 elevation and azimuth and press the button to indicate the next trial.

    Args:
        speakers : rows from the speaker table or index numbers of the speakers.
        duration (float): duration of the noise played from the target positions in seconds
        n_reps(int): number of repetitions for each target
        n_images(int): number of images taken for each head pose estimate
        visual(bool): If True, light a LED at the target position - the speakers must have a LED attached
    Returns:
        instance of slab.Trialsequence: the response is stored in the data attribute as tuples with (azimuth, elevation)
    """
    speakers = pick_speakers(speakers)
    if not PROCESSORS.mode == "loctest_freefield":
        PROCESSORS.initialize_default(mode="loctest_freefield")
    if visual is True:
        if not all([s.digital_channel for s in speakers]):
            raise ValueError("All speakers must have a LED attached for a test with visual cues")
    seq = slab.Trialsequence(speakers, n_reps, kind="non_repeating")
    play_start_sound()
    for speaker in seq:
        wait_for_button()
        while check_pose(fix=[0, 0]) is None:  # check if head is in position
            play_warning_sound()
            wait_for_button()
        sound = slab.Sound.pinknoise(duration=duration)
        write(tag="playbuflen", value=sound.n_samples, processors=["RX81", "RX82"])
        if visual is True:  # turn LED on
            write(tag="bitmask", value=speaker.digital_channel, processors=speaker.digital_proc)
        set_signal_and_speaker(signal=sound.data.flatten(), speaker=speaker)
        play()
        wait_to_finish_playing()
        wait_for_button()
        pose = get_head_pose(n_images=n_images)
        if visual is True:  # turn LED off
            write(tag="bitmask", value=0, processors=speaker.digital_proc)
        seq.add_response(pose)
    play_start_sound()
    # change conditions property so it contains the only azimuth and elevation of the source
    seq.conditions = np.array([(s.azimuth, s.elevation) for s in seq.conditions])
    return seq


def localization_test_headphones(speakers, signals, n_reps=1, n_images=5, visual=False):
    """
    Run a basic localization test where previously recorded/generated binaural sound are played via headphones.
    The procedure is the same as in localization_test_freefield().

    Args:
        speakers : rows from the speaker table or index numbers of the speakers.
        signals (array-like) : binaural sounds that are played. Must be ordered corresponding to the targets (first
            element of signals is played for the first row of targets etc.). If the elements of signals are
            instances of slab.Precomputed, a random one is drawn in each trial (useful if you don't want to repeat
            the exact same sound in each trial)
        n_reps(int): number of repetitions for each target
        n_images(int): number of images taken for each head pose estimate
        visual(bool): If True, light a LED at the target position - the speakers must have a LED attached
    Returns:
        instance of slab.Trialsequence: the response is stored in the data attribute as tuples with (azimuth, elevation)
    """
    if not PROCESSORS.mode == "loctest_headphones":
        PROCESSORS.initialize_default(mode="loctest_headphones")
    if not len(signals) == len(speakers):
        raise ValueError("There must be one signal for each target!")
    if not all(isinstance(sig, (slab.Binaural, slab.Precomputed)) for sig in signals):
        raise ValueError("Signal argument must be an instance of slab.Binaural or slab.Precomputed.")
    if visual is True:
        if not all([s.digital_channel for s in speakers]):
            raise ValueError("All speakers must have a LED attached for a test with visual cues")
    seq = slab.Trialsequence(speakers, n_reps, kind="non_repeating")
    play_start_sound()
    for speaker in seq:
        signal = signals[seq.trials[seq.this_n] - 1]  # get the signal corresponding to the target
        if isinstance(signal, slab.Precomputed):  # if signal is precomputed, pick a random one
            signal = signal[np.random.randint(len(signal))]
            try:
                signal = slab.Binaural(signal)
            except IndexError:
                logging.warning("Binaural sounds must have exactly two channels!")
        wait_for_button()
        while check_pose(fix=[0, 0]) is None:  # check if head is in position
            play_warning_sound()
            wait_for_button()
        write(tag="playbuflen", value=signal.n_samples, processors="RP2")
        write(tag="data_l", value=signal.left.data.flatten(), processors="RP2")
        write(tag="data_r", value=signal.right.data.flatten(), processors="RP2")
        if visual is True:  # turn LED on
            write(tag="bitmask", value=speaker.digital_channel, processors=speaker.digital_proc)
        play()
        wait_to_finish_playing()
        wait_for_button()
        pose = get_head_pose(n_images=n_images)
        if visual is True:  # turn LED off
            write(tag="bitmask", value=0, processors=speaker.digital_proc)
        seq.add_response(pose)
    play_start_sound()
    # change conditions property so it contains the only azimuth and elevation of the source
    seq.conditions = np.array([(s.azimuth, s.elevation) for s in seq.conditions])
    return seq


# functions implementing complete procedures:
def play_start_sound(speaker=23):
    """
    Load and play the sound that signals the start and end of an experiment/block
    """
    start = slab.Sound.read(DIR / "data" / "sounds" / "start.wav")
    set_signal_and_speaker(signal=start, speaker=speaker)
    play()


def play_warning_sound(duration=.5, speaker=23):
    """
    Load and play the sound that signals a warning (for example if the listener is in the wrong position)
    """
    warning = slab.Sound.clicktrain(duration=duration)
    set_signal_and_speaker(signal=warning, speaker=speaker)
    play()


def set_logger(level, report=True):
    """
    Set the logger to a specific level.
    Parameters:
        level: logging level. Only events of this level and above will be tracked. Can be 'DEBUG', 'INFO', 'WARNING',
         'ERROR' or 'CRITICAL'. Set level to '
    """
    try:
        logger = logging.getLogger()
        eval('logger.setLevel(logging.%s)' %level.upper())
        if report:
            logging.info('Logger set to %s.' %level.upper())
    except AttributeError:
        raise AttributeError("Choose from 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'")
