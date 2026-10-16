import numpy as np
import pytest
import slab
import pickle

import freefield.freefield as ff
import freefield.setups as setups

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# =============================================================================
# test apply_equalization
# =============================================================================

@pytest.fixture
def equalization_speaker(monkeypatch):
    speaker = ff.Speaker(
        index=2,
        analog_channel=0,
        analog_proc="RX8",
        digital_proc="RX8",
        azimuth=30,
        elevation=12.5,
        distance=1.4,
        digital_channel=0,
    )

    monkeypatch.setattr(
        ff,
        "SPEAKERS",
        [speaker],
    )

    return speaker


def test_apply_equalization_applies_level_to_copy(
    equalization_speaker,
):
    # Arrange
    signal = slab.Sound.whitenoise(
        duration=0.01,
        samplerate=48828,
    )
    signal.level = 40
    equalization_speaker.level = 65

    # Act
    result = ff.apply_equalization(
        signal=signal,
        speaker=equalization_speaker,
        level=True,
        frequency=False,
    )

    # Assert
    assert isinstance(result, slab.Sound)
    assert result is not signal
    assert result.level == pytest.approx(65)
    assert signal.level == pytest.approx(40)


def test_apply_equalization_raises_error_when_level_is_missing(
    equalization_speaker,
):
    # Arrange
    signal = slab.Sound.whitenoise(
        duration=0.01,
        samplerate=48828,
    )
    equalization_speaker.level = None

    # Act and Assert
    with pytest.raises(
        ValueError,
        match="speaker not level-equalized",
    ):
        ff.apply_equalization(
            signal=signal,
            speaker=equalization_speaker,
            level=True,
            frequency=False,
        )


def test_apply_equalization_applies_real_frequency_filter(
    equalization_speaker,
):
    # Arrange
    samplerate = 48828

    signal_data = np.array(
        [1.0, -0.5, 0.25, -0.125],
        dtype=float,
    )

    signal = slab.Sound(
        signal_data,
        samplerate=samplerate,
    )

    equalization_speaker.filter = slab.Filter(
        data=np.array([0.5]),
        samplerate=samplerate,
        fir="IR",
    )

    original_data = signal.data.copy()

    # Act
    result = ff.apply_equalization(
        signal=signal,
        speaker=equalization_speaker,
        level=False,
        frequency=True,
    )

    # Assert
    assert isinstance(result, slab.Sound)
    assert result is not signal
    assert result.samplerate == samplerate

    np.testing.assert_allclose(
        result.data,
        original_data * 0.5,
    )

    np.testing.assert_allclose(
        signal.data,
        original_data,
    )


def test_apply_equalization_raises_error_when_filter_is_missing(
    equalization_speaker,
):
    # Arrange
    signal = slab.Sound.whitenoise(
        duration=0.01,
        samplerate=48828,
    )

    equalization_speaker.filter = None

    # Act and Assert
    with pytest.raises(
        ValueError,
        match="speaker not frequency-equalized",
    ):
        ff.apply_equalization(
            signal=signal,
            speaker=equalization_speaker,
            level=False,
            frequency=True,
        )


def test_apply_equalization_returns_copy_when_corrections_are_disabled(
    equalization_speaker,
):
    # Arrange
    signal = slab.Sound.whitenoise(
        duration=0.01,
        samplerate=48828,
    )

    original_data = signal.data.copy()

    # Act
    result = ff.apply_equalization(
        signal=signal,
        speaker=equalization_speaker,
        level=False,
        frequency=False,
    )

    # Assert
    assert isinstance(result, slab.Sound)
    assert result is not signal

    np.testing.assert_allclose(
        result.data,
        original_data,
    )

    np.testing.assert_allclose(
        signal.data,
        original_data,
    )


def test_apply_equalization_applies_level_and_frequency_filter(
    equalization_speaker,
):
    # Arrange
    samplerate = 48828

    signal = slab.Sound.whitenoise(
        duration=0.01,
        samplerate=samplerate,
    )

    signal.level = 40

    equalization_speaker.level = 65

    equalization_speaker.filter = slab.Filter(
        data=np.array([0.5]),
        samplerate=samplerate,
        fir="IR",
    )

    # Act
    result = ff.apply_equalization(
        signal=signal,
        speaker=equalization_speaker,
        level=True,
        frequency=True,
    )

    # Assert
    assert isinstance(result, slab.Sound)
    assert result is not signal
    assert signal.level == pytest.approx(40)

    expected_level = 65 + 20 * np.log10(0.5)

    assert result.level == pytest.approx(expected_level)


# =============================================================================
# test get_recording_delay
# =============================================================================

def test_get_recording_delay_returns_sound_travel_time_without_processors():
    # Arrange
    distance = 343
    sample_rate = 1000

    # Act
    result = ff.get_recording_delay(
        distance=distance,
        sample_rate=sample_rate,
    )

    # Assert
    assert result == 1000


# =============================================================================
# test load_equalization
# =============================================================================

@pytest.fixture
def load_equalization_speakers(monkeypatch):
    speaker_2 = ff.Speaker(
        index=2,
        analog_channel=0,
        analog_proc="RX8",
        digital_proc="RX8",
        azimuth=30,
        elevation=12.5,
        distance=1.4,
        digital_channel=0,
    )

    speaker_3 = ff.Speaker(
        index=3,
        analog_channel=1,
        analog_proc="RX8",
        digital_proc="RX8",
        azimuth=-30,
        elevation=12.5,
        distance=1.4,
        digital_channel=1,
    )

    speaker_2.level = None
    speaker_2.filter = None

    speaker_3.level = None
    speaker_3.filter = None

    monkeypatch.setattr(
        ff,
        "SPEAKERS",
        [speaker_2, speaker_3],
    )

    return {
        2: speaker_2,
        3: speaker_3,
    }


def test_load_equalization_assigns_level_and_filter(
    tmp_path,
    load_equalization_speakers,
):
    # Arrange
    equalization_file = tmp_path / "equalization.pkl"

    equalization = {
        2: {
            "level": 62.5,
            "filter": "filter-for-speaker-2",
        },
        3: {
            "level": 64.0,
            "filter": "filter-for-speaker-3",
        },
    }

    with open(equalization_file, "wb") as file:
        pickle.dump(equalization, file)

    # Act
    ff.load_equalization(
        file=equalization_file,
    )

    # Assert
    assert load_equalization_speakers[2].level == 62.5
    assert (
        load_equalization_speakers[2].filter
        == "filter-for-speaker-2"
    )

    assert load_equalization_speakers[3].level == 64.0
    assert (
        load_equalization_speakers[3].filter
        == "filter-for-speaker-3"
    )


def test_load_equalization_ignores_level_if_false(
    tmp_path,
    load_equalization_speakers,
):
    # Arrange
    equalization_file = tmp_path / "equalization.pkl"

    equalization = {
        2: {
            "level": 62.5,
            "filter": "filter-for-speaker-2",
        },
        3: {
            "level": 64.0,
            "filter": "filter-for-speaker-3",
        },
    }

    with open(equalization_file, "wb") as file:
        pickle.dump(equalization, file)

    # Act
    ff.load_equalization(
        file=equalization_file,
        level=False,
    )

    # Assert
    assert load_equalization_speakers[2].level is None
    assert (
        load_equalization_speakers[2].filter
        == "filter-for-speaker-2"
    )

    assert load_equalization_speakers[3].level is None
    assert (
        load_equalization_speakers[3].filter
        == "filter-for-speaker-3"
    )


def test_load_equalization_ignores_filter_if_false(
    tmp_path,
    load_equalization_speakers,
):
    # Arrange
    equalization_file = tmp_path / "equalization.pkl"

    equalization = {
        2: {
            "level": 62.5,
            "filter": "filter-for-speaker-2",
        },
        3: {
            "level": 64.0,
            "filter": "filter-for-speaker-3",
        },
    }

    with open(equalization_file, "wb") as file:
        pickle.dump(equalization, file)

    # Act
    ff.load_equalization(
        file=equalization_file,
        frequency=False,
    )

    # Assert
    assert load_equalization_speakers[2].level == 62.5
    assert load_equalization_speakers[2].filter is None

    assert load_equalization_speakers[3].level == 64.0
    assert load_equalization_speakers[3].filter is None

@pytest.fixture
def test_setup():
    return setups.Setup(
        name="test_setup",
        speaker_table="tables/speakertable_test_setup.txt",
        calibration_file="calibration_test_setup.pkl",
        playback_processors=("RX81", "RX82"),
        recording_processor="RP2",
    )
def test_load_equalization_uses_default_file(
    tmp_path,
    load_equalization_speakers,
    monkeypatch,
    test_setup
):
    # Arrange

    monkeypatch.setattr(
        ff,
        "DIR",
        tmp_path,
    )

    monkeypatch.setattr(
        ff,
        "SETUP",
        test_setup,
    )

    data_directory = tmp_path / "data"
    data_directory.mkdir()

    equalization_file = (
        data_directory
        / "calibration_test_setup.pkl"
    )

    equalization = {
        2: {
            "level": 62.5,
            "filter": "filter-for-speaker-2",
        },
        3: {
            "level": 64.0,
            "filter": "filter-for-speaker-3",
        },
    }

    with open(equalization_file, "wb") as file:
        pickle.dump(equalization, file)

    # Act
    # bewusst kein file-Argument übergeben
    ff.load_equalization()

    # Assert
    assert load_equalization_speakers[2].level == 62.5
    assert (
        load_equalization_speakers[2].filter
        == "filter-for-speaker-2"
    )

    assert load_equalization_speakers[3].level == 64.0
    assert (
        load_equalization_speakers[3].filter
        == "filter-for-speaker-3"
    )


# =============================================================================
# test pick_speakers
# =============================================================================

@pytest.fixture
def pick_speakers_setup(monkeypatch):
    speaker_index_2 = ff.Speaker(
        index=2,
        analog_channel=0,
        analog_proc="RX8",
        digital_proc="RX8",
        azimuth=30,
        elevation=12.5,
        distance=1.4,
        digital_channel=0,
    )

    speaker_index_3 = ff.Speaker(
        index=3,
        analog_channel=0,
        analog_proc="RX8",
        digital_proc="RX8",
        azimuth=20,
        elevation=12.5,
        distance=1.4,
        digital_channel=0,
    )

    speaker_index_4 = ff.Speaker(
        index=4,
        analog_channel=0,
        analog_proc="RX8",
        digital_proc="RX8",
        azimuth=35,
        elevation=12.5,
        distance=1.4,
        digital_channel=0,
    )

    speakers_by_index = {
        2: speaker_index_2,
        3: speaker_index_3,
        4: speaker_index_4,
    }

    monkeypatch.setattr(
        ff,
        "SPEAKERS",
        list(speakers_by_index.values()),
    )

    return speakers_by_index


def test_pick_speakers_returns_speaker_for_existing_index(
    pick_speakers_setup,
):
    # Diagnose der Testvoraussetzungen
    assert [
        speaker.index
        for speaker in ff.SPEAKERS
    ] == [2, 3, 4]

    assert (
        ff.SPEAKERS[0]
        is pick_speakers_setup[2]
    )

    # Act
    result = ff.pick_speakers(2)

    # Assert
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0] is pick_speakers_setup[2]


def test_pick_speakers_returns_empty_list_for_non_existing_index(
    pick_speakers_setup,
):
    # Act
    result = ff.pick_speakers(99)

    # Assert
    assert result == []


def test_pick_speakers_returns_speakers_for_existing_indices(
    pick_speakers_setup,
):
    # Act
    result = ff.pick_speakers(
        [4, 2]
    )

    # Assert
    assert isinstance(result, list)
    assert len(result) == 2

    assert result[0] is pick_speakers_setup[2]
    assert result[1] is pick_speakers_setup[4]


def test_pick_speakers_returns_each_speaker_only_once_for_duplicate_indices(
    pick_speakers_setup,
):
    # Act
    result = ff.pick_speakers(
        [2, 2]
    )

    # Assert
    assert result == [
        pick_speakers_setup[2]
    ]


def test_pick_speakers_returns_speaker_for_existing_coordinates(
    pick_speakers_setup,
):
    # Act
    result = ff.pick_speakers(
        (30, 12.5, 1.4)
    )

    # Assert
    assert result == [
        pick_speakers_setup[2]
    ]


def test_pick_speakers_returns_empty_list_for_non_existing_coordinates(
    pick_speakers_setup,
):
    # Act
    result = ff.pick_speakers(
        (90, 17.5, 0.4)
    )

    # Assert
    assert result == []


def test_pick_speakers_returns_speakers_for_coordinate_list(
    pick_speakers_setup,
):
    # Arrange
    coordinates = [
        (35, 12.5, 1.4),
        (30, 12.5, 1.4),
    ]

    # Act
    result = ff.pick_speakers(
        coordinates
    )

    # Assert
    assert len(result) == 2
    assert result[0] is pick_speakers_setup[2]
    assert result[1] is pick_speakers_setup[4]


def test_pick_speakers_returns_single_speaker_object(
    pick_speakers_setup,
):
    # Arrange
    selected_speaker = pick_speakers_setup[3]

    # Act
    result = ff.pick_speakers(
        selected_speaker
    )

    # Assert
    assert len(result) == 1
    assert result[0] is selected_speaker


def test_pick_speakers_returns_list_of_speaker_objects(
    pick_speakers_setup,
):
    # Arrange
    selected_speakers = [
        pick_speakers_setup[4],
        pick_speakers_setup[2],
    ]

    # Act
    result = ff.pick_speakers(
        selected_speakers
    )

    # Assert
    assert len(result) == 2
    assert result[0] is pick_speakers_setup[4]
    assert result[1] is pick_speakers_setup[2]


@pytest.mark.parametrize(
    "numpy_index",
    [
        np.int32(3),
        np.int64(3),
    ],
)
def test_pick_speakers_accepts_numpy_integer_index(
    pick_speakers_setup,
    numpy_index,
):
    # Act
    result = ff.pick_speakers(
        numpy_index
    )

    # Assert
    assert len(result) == 1
    assert result[0] is pick_speakers_setup[3]


def test_pick_speakers_accepts_numpy_array_of_indices(
    pick_speakers_setup,
):
    # Arrange
    indices = np.array(
        [4, 2],
        dtype=np.int64,
    )

    # Act
    result = ff.pick_speakers(
        indices
    )

    # Assert
    assert len(result) == 2
    assert result[0] is pick_speakers_setup[2]
    assert result[1] is pick_speakers_setup[4]


def test_pick_speakers_returns_empty_list_for_empty_selection(
    pick_speakers_setup,
):
    # Act
    result = ff.pick_speakers([])

    # Assert
    assert result == []


# =============================================================================
# test read_speaker_table
# =============================================================================

def test_read_speaker_table_assigns_columns_correctly(
    monkeypatch,
    test_setup
):
    # Arrange
    table = np.array(
        [
            [
                "0",
                "22",
                "RX82",
                "-52.5",
                "25",
                "1.4",
                "",
                "",
            ],
            [
                "1",
                "23",
                "RX82",
                "52.5",
                "25",
                "1.6",
                "7",
                "RX81",
            ],
        ],
        dtype=str,
    )

    monkeypatch.setattr(
        ff,
        "SETUP",
        test_setup,
    )

    monkeypatch.setattr(
        ff.np,
        "loadtxt",
        lambda *args, **kwargs: table,
    )

    # Act
    result = ff.read_speaker_table()

    # Assert
    assert len(result) == 2

    speaker_0 = result[0]

    assert speaker_0.index == 0
    assert speaker_0.analog_channel == 22
    assert speaker_0.analog_proc == "RX82"
    assert speaker_0.azimuth == -52.5
    assert speaker_0.elevation == 25.0
    assert speaker_0.distance == 1.4
    assert speaker_0.digital_channel is None
    assert speaker_0.digital_proc is None

    speaker_1 = result[1]

    assert speaker_1.index == 1
    assert speaker_1.analog_channel == 23
    assert speaker_1.analog_proc == "RX82"
    assert speaker_1.azimuth == 52.5
    assert speaker_1.elevation == 25.0
    assert speaker_1.distance == 1.6
    assert speaker_1.digital_channel == 7.0
    assert speaker_1.digital_proc == "RX81"


# =============================================================================
# test spectral_range
# =============================================================================

class DummySignal:
    def __init__(self):
        self.n_channels = 3
        self.samplerate = 48828

    def channel(self, index):
        return index


class DummyFilteredSignal:
    def __init__(self, level):
        self.level = np.asarray(
            level,
            dtype=float,
        )


class DummyFilterBank:
    def __init__(self):
        self.n_channels = 4

        self.channel_levels = {
            0: [10, 20, 30, 40],
            1: [13, 18, 35, 40],
            2: [11, 25, 32, 38],
        }

    def apply(self, channel):
        return DummyFilteredSignal(
            level=self.channel_levels[channel],
        )


def prepare_spectral_range(
    monkeypatch,
):
    filter_bank = DummyFilterBank()
    received_filterbank_arguments = {}

    def create_filterbank(**kwargs):
        received_filterbank_arguments.update(
            kwargs
        )
        return filter_bank

    def create_center_frequencies(
        low_cutoff,
        high_cutoff,
        bandwidth,
    ):
        return (
            np.array(
                [1, 2, 3, 4],
                dtype=float,
            ),
            None,
            None,
        )

    def convert_erb_to_frequency(
        center_frequencies,
    ):
        return np.array(
            [
                100,
                500,
                1000,
                5000,
            ],
            dtype=float,
        )

    monkeypatch.setattr(
        ff.slab.Filter,
        "cos_filterbank",
        create_filterbank,
    )

    monkeypatch.setattr(
        ff.slab.Filter,
        "_center_freqs",
        create_center_frequencies,
    )

    monkeypatch.setattr(
        ff.slab.Filter,
        "_erb2freq",
        convert_erb_to_frequency,
    )

    return received_filterbank_arguments


def test_spectral_range_returns_level_range_for_each_frequency_band(
    monkeypatch,
):
    # Arrange
    signal = DummySignal()

    received_arguments = prepare_spectral_range(
        monkeypatch
    )

    # Act
    result = ff.spectral_range(
        signal=signal,
        bandwidth=0.2,
        low_cutoff=100,
        high_cutoff=5000,
        plot=False,
    )

    # Assert
    expected = np.array(
        [3, 7, 5, 2],
        dtype=float,
    )

    np.testing.assert_allclose(
        result,
        expected,
    )

    assert received_arguments == {
        "length": 1000,
        "bandwidth": 0.2,
        "low_cutoff": 100,
        "high_cutoff": 5000,
        "samplerate": 48828,
    }


def test_spectral_range_plots_on_given_axes(
    monkeypatch,
):
    # Arrange
    signal = DummySignal()

    prepare_spectral_range(
        monkeypatch
    )

    figure, axes = plt.subplots()

    try:
        # Act
        result = ff.spectral_range(
            signal=signal,
            bandwidth=0.2,
            low_cutoff=100,
            high_cutoff=5000,
            thresh=3,
            plot=axes,
            log=True,
        )

        # Assert
        # numerisches Ergebnis bleibt unverändert
        np.testing.assert_allclose(
            result,
            np.array(
                [3, 7, 5, 2],
                dtype=float,
            ),
        )

        # Maximum und Minimum werden
        # als zwei Linien dargestellt
        assert len(axes.lines) == 2

        max_line, min_line = axes.lines

        np.testing.assert_allclose(
            max_line.get_xdata(),
            [100, 500, 1000, 5000],
        )

        np.testing.assert_allclose(
            max_line.get_ydata(),
            [13, 25, 35, 40],
        )

        np.testing.assert_allclose(
            min_line.get_ydata(),
            [10, 18, 30, 38],
        )

        assert max_line.get_color() == "black"
        assert min_line.get_color() == "black"

        assert (
            max_line.get_linestyle()
            == "--"
        )

        assert (
            min_line.get_linestyle()
            == "--"
        )

        # semilogx() muss eine
        # logarithmische x-Achse erzeugen
        assert axes.get_xscale() == "log"

        # Differenzen über 3 liegen
        # in Band 2 und 3:
        # [3, 7, 5, 2] > 3
        assert len(axes.collections) == 2

        for collection in axes.collections:
            red, green, blue, alpha = (
                collection
                .get_facecolor()[0]
            )

            assert red == 1
            assert green == 0
            assert blue == 0
            assert alpha == 0.6

    finally:
        plt.close(figure)