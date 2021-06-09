import time
from pathlib import Path
import datetime
from copy import deepcopy
import pickle
from dataclasses import dataclass
import logging
import numpy
import numpy as np
import slab
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from freefield import DIR, Processors, camera
logging.basicConfig(level=logging.INFO)
slab.Signal.set_default_samplerate(48828)  # default samplerate for generating sounds, filters etc.
# Initialize global variables:
CAMERAS = camera.Cameras()
PROCESSORS = Processors()
EQUALIZATIONFILE = Path()
EQUALIZATIONDICT = {}  # calibration to equalize levels
SPEAKERS = []  # list of all the loudspeakers in the active setup


def initialize_setup(setup, default_mode=None, proc_list=None, zbus=True, connection="GB", camera_type=None):
    """
    Initialize the processors and load table and calibration for setup.

    We are using two different 48-channel setups. A 3-dimensional 'dome' and
    a horizontal 'arc'. For each setup there is a table describing the position
    and channel of each loudspeaker as well as calibration files. This function
    loads those files and stores them in global variables. This is necessary
    for most of the other functions to work.

    Args:
        setup: determines which files to load, can be 'dome' or 'arc'
        default_mode: initialize the setup using one of the defaults, see Processors.initialize_default
        proc_list: if not using a default, specify the processors in a list, see processors.initialize_processors
        zbus: whether or not to initialize the zbus interface
        connection: type of connection to processors, can be "GB" (optical) or "USB"
        camera_type: kind of camera that is initialized. Can be "webcam", "flir" or None
    """

    # TODO: put level and frequency equalization in one common file
    global EQUALIZATIONDICT, EQUALIZATIONFILE, PROCESSORS, CAMERAS
    # initialize processors
    if bool(proc_list) == bool(default_mode):
        raise ValueError("You have to specify a proc_list OR a default_mode")
    if proc_list is not None:
        PROCESSORS.initialize(proc_list, zbus, connection)
    elif default_mode is not None:
        PROCESSORS.initialize_default(default_mode)
    if camera_type is not None:
        CAMERAS = camera.initialize_cameras(camera_type)
    read_speaker_table(setup)  # load the table containing the information about the loudspeakers
    logging.info(f'Speaker configuration set to {setup}.')
    EQUALIZATIONFILE = DIR / 'data' / Path(f'calibration_{setup}.pkl')
    if EQUALIZATIONFILE.exists():
        with open(EQUALIZATIONFILE, 'rb') as f:
            EQUALIZATIONDICT = pickle.load(f)
        logging.info('Frequency-calibration filters loaded.')
    else:
        logging.warning('Setup not calibrated...')


@dataclass(frozen=True)
class Speaker:
    index: int  # the index number of the speaker
    analog_channel: int  # the number of the analog channel to which the speaker is attached
    analog_proc: str  # the processor to whose analog I/O the speaker is attached
    digital_proc: str  # the processor to whose digital I/O the speaker's LED is attached
    azimuth: float  # the azimuth angle of the speaker
    elevation: float  # the azimuth angle of the speaker
    digital_channel: int  # the int value of the bitmask for the digital channel to which the speakers LED is attached


def read_speaker_table(setup="dome"):
    global SPEAKERS
    SPEAKERS = []
    if setup not in ["dome", "arc"]:
        raise ValueError("Setup must be 'dome' or 'arc'!")
    table_file = DIR / 'data' / 'tables' / Path(f'speakertable_{setup}.txt')
    table = numpy.loadtxt(table_file, skiprows=1, delimiter=",", dtype=str)
    for row in table:
        SPEAKERS.append(Speaker(index=int(row[0]), analog_channel=int(row[1]), analog_proc=row[2],
                                azimuth=float(row[3]), digital_channel=int(row[5]) if row[5] else None,
                                elevation=float(row[4]), digital_proc=row[6] if row[6] else None))


# Wrappers for Processor operations read, write, trigger and halt:
def write(tag, value, procs):
    PROCESSORS.write(tag=tag, value=value, procs=procs)


def read(tag, proc, n_samples=1):
    value = PROCESSORS.read(tag=tag, proc=proc, n_samples=n_samples)
    return value


def play(kind='zBusA', proc=None):
    PROCESSORS.trigger(kind=kind, proc=proc)


def halt():
    PROCESSORS.halt()
    CAMERAS.halt()


def wait_to_finish_playing(proc="all", tag="playback"):
    """
    Busy wait until the processors finished playing.

    For this function to work, the rcx-circuit must have a tag that is 1
    while output is generated and 0 otherwise. The default name for this
    kind of tag is "playback". "playback" is read repeatedly for each processors
    followed by a short sleep if the value is 1.

    Args:
        proc (str, list of str): name(s) of the processor(s) to wait for.
        tag (str): name of the tag that signals if something is played
    """
    if proc == "all":
        proc = list(PROCESSORS.procs.keys())
    elif isinstance(proc, str):
        proc = [proc]
    logging.info(f'Waiting for {tag} on {proc}.')
    while any(PROCESSORS.read(tag, n_samples=1, proc=p) for p in proc):
        time.sleep(0.01)
    logging.info('Done waiting.')


def wait_for_button() -> None:
    while not PROCESSORS.read(tag="response", proc="RP2"):
        time.sleep(0.1)  # wait until button is pressed


def play_and_wait() -> None:
    PROCESSORS.trigger()
    wait_to_finish_playing()


def play_and_wait_for_button() -> None:
    play_and_wait()
    wait_for_button()


def pick_speakers(picks):
    """
    Either return the speaker at given coordinates (azimuth, elevation) or the
    speaker with a specific index number.

    Args:
        picks (list of lists, list, int): index number of the speaker

    Returns:
    """
    if isinstance(picks, (list, numpy.ndarray)):
        if all(isinstance(p, (int, numpy.int64, numpy.int32)) for p in picks):
            speakers = [s for s in SPEAKERS if s.index in picks]
        else:
            speakers = [s for s in SPEAKERS if (s.azimuth, s.elevation) in picks]
    elif isinstance(picks, (int, numpy.int)):
        speakers = [s for s in SPEAKERS if s.index == picks]
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
    # TODO: this is not how it works
    global SPEAKERS
    for speaker in SPEAKERS:
        speaker.azimuth += delta_azi  # azimuth
        speaker.elevation += delta_ele  # elevation
    print(f"shifting the loudspeaker array by {delta_azi} in azimuth and {delta_ele} in elevation")


def set_signal_and_speaker(signal, speaker, calibrate=True):
    """
    Load a signal into the processor buffer and set the output channel to match the speaker.
    The processor is chosen automatically depending on the speaker.

        Args:
            signal (array-like): signal to load to the buffer, must be one-dimensional
            speaker (Speaker, int) : speaker to play the signal from, can be index number or [azimuth, elevation]
            calibrate (bool): if True (=default) apply loudspeaker equalization
    """
    signal = slab.Sound(signal)
    if not isinstance(speaker, Speaker):
        speaker = pick_speakers(speaker)[0]
    if calibrate:
        logging.info('Applying calibration.')  # apply level and frequency calibration
        to_play = apply_equalization(signal, speaker)
    else:
        to_play = signal
    PROCESSORS.write(tag='chan', value=speaker.analog_channel, procs=speaker.analog_proc)
    PROCESSORS.write(tag='data', value=to_play.data, procs=speaker.analog_proc)
    other_procs = set([s.analog_proc for s in SPEAKERS])
    other_procs.remove(speaker.analog_proc)  # set the analog output of other procs to non existent number 99
    PROCESSORS.write(tag='chan', value=99, procs=other_procs)


def apply_equalization(signal, speaker, level=True, frequency=True):
    """
    Apply level correction and frequency equalization to a signal

    Args:
        signal: signal to calibrate
        speaker: index number, coordinates or row from the speaker table. Determines which calibration is used
    Returns:
        slab.Sound: calibrated copy of signal
    """
    if not bool(EQUALIZATIONDICT):
        logging.warning("Setup is not calibrated! Returning the signal unchanged...")
        return signal
    else:
        signal = slab.Sound(signal)
        if not isinstance(speaker, Speaker):
            speaker = pick_speakers(speaker)[0]
        speaker_calibration = EQUALIZATIONDICT[str(speaker.index_number.iloc[0])]
        calibrated_signal = deepcopy(signal)
        if level:
            calibrated_signal.level *= speaker_calibration["level"]
        if frequency:
            calibrated_signal = speaker_calibration["filter"].apply(calibrated_signal)
        return calibrated_signal


def get_recording_delay(distance=1.6, sample_rate=48828, play_from=None, rec_from=None):
    """
        Calculate the delay it takes for played sound to be recorded. Depends
        on the distance of the microphone from the speaker and on the processors
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


def get_head_pose(n_images=1):
    """Wrapper for the get headpose method of the camera class"""
    if not CAMERAS.n_cams:
        raise ValueError("No cameras initialized!")
    else:
        azi, ele = CAMERAS.get_head_pose(convert=True, average_axis=(1, 2), n_images=n_images)
    return azi, ele


def check_pose(fix=(0, 0), var=10):
    """
    Check if the head pose is directed towards the fixation point

    Args:
        fix: azimuth and elevation of the fixation point
        var: degrees, the pose is allowed to deviate from the fixation point in azimuth and elevations
    Returns:
        bool: True if difference between pose and fix is smaller than var, False otherwise
    """
    # TODO: what happens if no image is obtained?
    azi, ele = get_head_pose(n_images=1)
    if (azi is np.nan) or (azi is None):
        azi = fix[0]
    if (ele is np.nan) or (ele is None):
        ele = fix[1]
    if np.abs(azi - fix[0]) > var or np.abs(ele - fix[1]) > var:
        return False
    else:
        return True


# functions implementing complete procedures:
def play_start_sound(speaker=23):
    """
    Load and play the sound that signals the start and end of an experiment/block
    """
    start = slab.Sound.read(DIR/"data"/"sounds"/"start.wav")
    set_signal_and_speaker(signal=start, speaker=speaker)
    play_and_wait()


def play_warning_sound(duration=.5, speaker=23):
    """
    Load and play the sound that signals a warning (for example if the listener is in the wrong position)
    """
    warning = slab.Sound.clicktrain(duration=duration)
    set_signal_and_speaker(signal=warning, speaker=speaker)
    play_and_wait()


def calibrate_camera(speakers, n_reps=1, n_images=5):
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
    if not all(isinstance(s, Speaker) for s in speakers):
        speakers = pick_speakers(speakers)
    if not all([s.digital_channel for s in speakers]):
        raise ValueError("All speakers must have a LED attached for a test with visual cues")
    seq = slab.Trialsequence(n_reps=n_reps, conditions=speakers)
    world_coordinates = [(seq.conditions[t-1].azimuth, seq.conditions[t-1].elevation) for t in seq.trials]
    camera_coordinates = []
    for speaker in seq:
        write(tag="bitmask", value=int(speaker.digital_channel), procs=speaker.digital_proc)
        wait_for_button()
        camera_coordinates.append(CAMERAS.get_head_pose(average_axis=1, convert=False, n_images=n_images))
        write(tag="bitmask", value=0, procs=speaker.digital_proc)
    CAMERAS.calibrate(world_coordinates, camera_coordinates, plot=True)


def calibrate_camera_no_visual(speakers, n_reps=1, n_images=5):
    """
    This is an alteration of calibrate_camera for cases in which LEDs are
    not available. The list of targets is repeated n_reps times in the
    exact same order without any randomization. When the whole setup is
    equipped with LEDs this function should be removed
    """
    if not PROCESSORS.mode == "cam_calibration":
        PROCESSORS.initialize_default(mode="cam_calibration")
    if not all(isinstance(s, Speaker) for s in speakers):
        speakers = pick_speakers(speakers)
    camera_coordinates = []
    speakers = speakers * n_reps
    world_coordinates = [(s.azimuth, s.elevation) for s in speakers]
    for speaker in speakers:
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
    if not all(isinstance(s, Speaker) for s in speakers):
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
        write(tag="playbuflen", value=sound.n_samples, procs=["RX81", "RX82"])
        if visual is True:  # turn LED on
            write(tag="bitmask", value=speaker.digital_channel, procs=speaker.digital_proc)
        set_signal_and_speaker(signal=sound.data.flatten(), speaker=speaker)
        play_and_wait_for_button()
        pose = get_head_pose(n_images=n_images)
        if visual is True:  # turn LED off
            write(tag="bitmask", value=0, procs=speaker.digital_proc)
        seq.add_response(pose)
    play_start_sound()
    # change conditions property so it contains the only azimuth and elevation of the source
    seq.conditions = numpy.array([(s.azimuth, s.elevation) for s in seq.conditions])
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
        signal = signals[seq.trials[seq.this_n]-1]  # get the signal corresponding to the target
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
        write(tag="playbuflen", value=signal.n_samples, procs="RP2")
        write(tag="data_l", value=signal.left.data.flatten(), procs="RP2")
        write(tag="data_r", value=signal.right.data.flatten(), procs="RP2")
        if visual is True:  # turn LED on
            write(tag="bitmask", value=speaker.digital_channel, procs=speaker.digital_proc)
        play_and_wait_for_button()
        pose = get_head_pose(n_images=n_images)
        if visual is True:  # turn LED off
            write(tag="bitmask", value=0, procs=speaker.digital_proc)
        seq.add_response(pose)
    play_start_sound()
    # change conditions property so it contains the only azimuth and elevation of the source
    seq.conditions = numpy.array([(s.azimuth, s.elevation) for s in seq.conditions])
    return seq


def equalize_speakers(speakers="all", target_speaker=23, bandwidth=1/10, db_tresh=80,
                      low_cutoff=200, high_cutoff=16000, alpha=1.0, plot=False, test=True):
    """
    Equalize the loudspeaker array in two steps. First: equalize over all
    level differences by a constant for each speaker. Second: remove spectral
    difference by inverse filtering. For more details on how the
    inverse filters are computed see the documentation of slab.Filter.equalizing_filterbank
    """
    global EQUALIZATIONDICT
    logging.info('Starting calibration.')
    if not PROCESSORS.mode == "play_rec":
        PROCESSORS.initialize_default(mode="play_and_record")
    sig = slab.Sound.chirp(duration=0.05, from_frequency=low_cutoff, to_frequency=high_cutoff)
    if speakers == "all":  # use the whole speaker table
        speaker_list = TABLE
    elif isinstance(speakers, list):  # use a subset of speakers
        speaker_list = get_speaker_list(speakers)
    else:
        raise ValueError("Argument speakers must be a list of interers or 'all'!")
    calibration_lvls = _level_equalization(sig, speaker_list, target_speaker, db_tresh)
    filter_bank, rec = _frequency_equalization(sig, speaker_list, target_speaker, calibration_lvls,
                                               bandwidth, low_cutoff, high_cutoff, alpha, db_tresh)
    # if plot:  # save plot for each speaker
    #     for i in range(rec.nchannels):
    #         _plot_equalization(target_speaker, rec.channel(i),
    #                            fbank.channel(i), i)
    for i in range(TABLE.shape[0]):  # write level and frequency equalization into one dictionary
        EQUALIZATIONDICT[str(i)] = {"level": calibration_lvls[i], "filter": filter_bank.channel(i)}
    if EQUALIZATIONFILE.exists():  # move the old calibration to the log folder
        date = datetime.datetime.now().strftime("_%Y-%m-%d-%H-%M-%S")
        rename_previous = DIR / 'data' / Path("log/" + EQUALIZATIONFILE.stem + date + EQUALIZATIONFILE.suffix)
        EQUALIZATIONFILE.rename(rename_previous)
    with open(EQUALIZATIONFILE, 'wb') as f:  # save the newly recorded calibration
        pickle.dump(EQUALIZATIONDICT, f, pickle.HIGHEST_PROTOCOL)
    logging.info('Calibration completed.')


def _level_equalization(sig, speaker_list, target_speaker, db_thresh):
    """
    Record the signal from each speaker in the list and return the level of each
    speaker relative to the target speaker(target speaker must be in the list)
    """
    rec = []
    for i in range(speaker_list.shape[0]):
        row = speaker_list.loc[i]
        rec.append(play_and_record(row.index_number, sig, apply_calibration=False))
        if row.index_number == target_speaker:
            target = rec[-1]
    rec = slab.Sound(rec)
    rec.data[:, rec.level < db_thresh] = target.data  # thresholding
    return target.level / rec.level


def _frequency_equalization(sig, speaker_list, target_speaker, calibration_lvls, bandwidth,
                            low_cutoff, high_cutoff, alpha, db_thresh):
    """
    play the level-equalized signal, record and compute and a bank of inverse filter
    to equalize each speaker relative to the target one. Return filterbank and recordings
    """
    rec = []
    for i in range(speaker_list.shape[0]):
        row = speaker_list.loc[i]
        modulated_sig = deepcopy(sig)  # copy signal and correct for lvl difference
        modulated_sig.level *= calibration_lvls[row.index_number]
        rec.append(play_and_record(row.index_number, modulated_sig, apply_calibration=False))
        if row.index_number == target_speaker:
            target = rec[-1]
    rec = slab.Sound(rec)
    # set recordings which are below the threshold or which are from exluded speaker
    # equal to the target so that the resulting frequency filter will be flat
    rec.data[:, rec.level < db_thresh] = target.data

    filter_bank = slab.Filter.equalizing_filterbank(target=target, signal=rec, low_cutoff=low_cutoff,
                                                    high_cutoff=high_cutoff, bandwidth=bandwidth, alpha=alpha)
    # check for notches in the filter:
    transfer_function = filter_bank.tf(show=False)[1][0:900, :]
    if (transfer_function < -30).sum() > 0:
        logging.warning(f"The filter for speaker {row.index_number} at azimuth {row.azi} and elevation {row.ele} /n"
                        "contains deep notches - adjust the equalization parameters!")

    return filter_bank, rec


def check_equalization(sig, speakers="all", max_diff=5, db_thresh=80):
    """
    Test the effectiveness of the speaker equalization
    """
    fig, ax = plt.subplots(3, 2, sharex=True)
    # recordings without, with level and with complete (level+frequency) equalization
    rec_raw, rec_lvl_eq, rec_freq_eq = [], [], []
    if speakers == "all":  # use the whole speaker table
        speaker_list = TABLE
    elif isinstance(speakers, list):  # use a subset of speakers
        speaker_list = get_speaker_list(speakers)
    else:
        raise ValueError("Speakers must be 'all' or a list of indices/coordinates!")
    for i in range(speaker_list.shape[0]):
        row = speaker_list.loc[i]
        sig2 = apply_equalization(sig, speaker=row.index_number, level=True, frequency=False)  # only level equalization
        sig3 = apply_equalization(sig, speaker=row.index_number, level=True, frequency=True)  # level and frequency
        rec_raw.append(play_and_record(row.index_number, sig, calibrate=False))
        rec_lvl_eq.append(play_and_record(row.index_number, sig2, calibrate=False))
        rec_freq_eq.append(play_and_record(row.index_number, sig3, calibrate=False))
    for i, rec in enumerate([rec_raw, rec_lvl_eq, rec_freq_eq]):
        rec = slab.Sound(rec)
        rec.data = rec.data[:, rec.level > db_thresh]
        rec.spectrum(axes=ax[i, 0], show=False)
        spectral_range(rec, plot=ax[i, 1], thresh=max_diff, log=False)
    plt.show()

    return slab.Sound(rec_raw), slab.Sound(rec_lvl_eq), slab.Sound(rec_freq_eq)


def spectral_range(signal, bandwidth=1 / 5, low_cutoff=50, high_cutoff=20000, thresh=3,
                   plot=True, log=True):
    """
    Compute the range of differences in power spectrum for all channels in
    the signal. The signal is devided into bands of equivalent rectangular
    bandwidth (ERB - see More&Glasberg 1982) and the level is computed for
    each frequency band and each channel in the recording. To show the range
    of spectral difference across channels the minimum and maximum levels
    across channels are computed. Can be used for example to check the
    effect of loud speaker equalization.
    """
    # TODO: this really should be part of the slab.Sound file
    # generate ERB-spaced filterbank:
    fbank = slab.Filter.cos_filterbank(length=1000, bandwidth=bandwidth,
                                       low_cutoff=low_cutoff, high_cutoff=high_cutoff,
                                       samplerate=signal.samplerate)
    center_freqs, _, _ = slab.Filter._center_freqs(low_cutoff, high_cutoff, bandwidth)
    center_freqs = slab.Filter._erb2freq(center_freqs)
    # create arrays to write data into:
    levels = np.zeros((signal.nchannels, fbank.nchannels))
    max_level, min_level = np.zeros(fbank.nchannels), np.zeros(fbank.nchannels)
    for i in range(signal.nchannels):  # compute ERB levels for each channel
        levels[i] = fbank.apply(signal.channel(i)).level
    for i in range(fbank.nchannels):  # find max and min for each frequency
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


def play_and_record(speaker_nr, sig, compensate_delay=True, compensate_level=True, calibrate=False):
    """
    Play the signal from a speaker and return the recording. Delay compensation
    means making the buffer of the recording processor n samples longer and then
    throwing the first n samples away when returning the recording so sig and
    rec still have the same legth. For this to work, the circuits rec_buf.rcx
    and play_buf.rcx have to be initialized on RP2 and RX8s and the mic must
    be plugged in.
    Parameters:
        speaker_nr: integer between 1 and 48, index number of the speaker
        sig: instance of slab.Sound, signal that is played from the speaker
        compensate_delay: bool, compensate the delay between play and record
    Returns:
        rec: 1-D array, recorded signal
    """
    if PROCESSORS.mode == "play_birec":
        binaural = True  # 2 channel recording
    elif PROCESSORS.mode == "play_rec":
        binaural = False  # record single channel
    else:
        raise ValueError("Setup must be initialized in mode 'play_rec' or 'play_birec'!")
    PROCESSORS.write(tag="playbuflen", value=sig.nsamples, procs=["RX81", "RX82"])
    if compensate_delay:
        n_delay = get_recording_delay(play_from="RX8", rec_from="RP2")
        n_delay += 50  # make the delay a bit larger, just to be sure
    else:
        n_delay = 0
    PROCESSORS.write(tag="playbuflen", value=sig.nsamples, procs=["RX81", "RX82"])
    PROCESSORS.write(tag="playbuflen", value=sig.nsamples + n_delay, procs="RP2")
    set_signal_and_speaker(sig, speaker_nr, calibrate)
    play_and_wait()
    if binaural is False:  # read the data from buffer and skip the first n_delay samples
        rec = PROCESSORS.read(tag='data', proc='RP2', n_samples=sig.nsamples + n_delay)[n_delay:]
        rec = slab.Sound(rec)
    else:  # read data for left and right ear from buffer
        rec_l = PROCESSORS.read(tag='datal', proc='RP2', n_samples=sig.nsamples + n_delay)[n_delay:]
        rec_r = PROCESSORS.read(tag='datar', proc='RP2', n_samples=sig.nsamples + n_delay)[n_delay:]
        rec = slab.Binaural([rec_l, rec_r])
    if compensate_level:
        if binaural:
            iid = rec.left.level - rec.right.level
            rec.level = sig.level
            rec.left.level += iid
        else:
            rec.level = sig.level
    return rec
