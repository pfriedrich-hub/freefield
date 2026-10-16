from __future__ import annotations
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProcessorConfig:
    name: str
    model: str
    circuit: str


@dataclass(frozen=True)
class ModeConfig:
    processors: Tuple[ProcessorConfig, ...]
    zbus: bool = True
    connection: str = "GB"


@dataclass(frozen=True)
class Setup:
    name: str
    speaker_table: str
    calibration_file: str

    playback_processors: Tuple[str, ...]
    recording_processor: Optional[str]

    center_speaker: Optional[int] = None
    reverb_wait: float = 0.0

    default_modes: Dict[str, ModeConfig] = field(default_factory=dict)


# -------------------------------------------------------------------------
# Shared mode definitions
# -------------------------------------------------------------------------

FREEFIELD_STANDARD_MODES = {
    "play_rec": ModeConfig(
        processors=(
            ProcessorConfig("RP2", "RP2", "rec_buf.rcx"),
            ProcessorConfig("RX81", "RX8", "play_buf.rcx"),
            ProcessorConfig("RX82", "RX8", "play_buf.rcx"),
        ),
        zbus=True,
        connection="GB",
    ),

    "play_birec": ModeConfig(
        processors=(
            ProcessorConfig("RP2", "RP2", "bi_rec_buf.rcx"),
            ProcessorConfig("RX81", "RX8", "play_buf.rcx"),
            ProcessorConfig("RX82", "RX8", "play_buf.rcx"),
        ),
        zbus=True,
        connection="GB",
    ),

    "bi_play_rec": ModeConfig(
        processors=(
            ProcessorConfig("RP2", "RP2", "bi_play_rec_buf.rcx"),
            ProcessorConfig("RX81", "RX8", "bits.rcx"),
            ProcessorConfig("RX82", "RX8", "bits.rcx"),
        ),
        zbus=True,
        connection="GB",
    ),

    "loctest_freefield": ModeConfig(
        processors=(
            ProcessorConfig("RP2", "RP2", "button.rcx"),
            ProcessorConfig("RX81", "RX8", "play_buf.rcx"),
            ProcessorConfig("RX82", "RX8", "play_buf.rcx"),
        ),
        zbus=True,
        connection="GB",
    ),

    "loctest_headphones": ModeConfig(
        processors=(
            ProcessorConfig("RP2", "RP2", "bi_play_buf.rcx"),
            ProcessorConfig("RX81", "RX8", "bits.rcx"),
            ProcessorConfig("RX82", "RX8", "bits.rcx"),
        ),
        zbus=True,
        connection="GB",
    ),

    "cam_calibration": ModeConfig(
        processors=(
            ProcessorConfig("RP2", "RP2", "button.rcx"),
            ProcessorConfig("RX81", "RX8", "bits.rcx"),
            ProcessorConfig("RX82", "RX8", "bits.rcx"),
        ),
        zbus=True,
        connection="GB",
    ),
}


# -------------------------------------------------------------------------
# Setup definitions
# -------------------------------------------------------------------------

DOME = Setup(
    name="dome",
    speaker_table="tables/speakertable_dome.txt",
    calibration_file="calibration_dome.pkl",
    playback_processors=("RX81", "RX82"),
    recording_processor="RP2",
    center_speaker=23,
    reverb_wait=0.0,
    default_modes=FREEFIELD_STANDARD_MODES,
)


ARC = Setup(
    name="arc",
    speaker_table="tables/speakertable_arc.txt",
    calibration_file="calibration_arc.pkl",
    playback_processors=("RX81", "RX82"),
    recording_processor="RP2",
    center_speaker=23,
    reverb_wait=0.0,
    default_modes=FREEFIELD_STANDARD_MODES,
)

HEADPHONES = Setup(
    name="arc",
    speaker_table="tables/speakertable_arc.txt",
    calibration_file="calibration_arc.pkl",
    playback_processors=("RX81", "RX82"),
    recording_processor="RP2",
    center_speaker=23,
    reverb_wait=0.0,
    default_modes=FREEFIELD_STANDARD_MODES,
)


DISTANCE_ARRAY = Setup(
    name="distance_array",
    speaker_table="tables/speakertable_distance_array.txt",
    calibration_file="calibration_cathedral.pkl",
    playback_processors=("RX81",),
    recording_processor="RP2",
    center_speaker=23,
    reverb_wait=2.7,
    default_modes={
    "play_rec": ModeConfig(
        processors=(
            ProcessorConfig("RP2", "RP2", "rec_buf.rcx"),
            ProcessorConfig("RX81", "RX8", "play_buf.rcx"),
        ),
        zbus=False,
        connection="USB",
    ),

    "play_birec": ModeConfig(
        processors=(
            ProcessorConfig("RP2", "RP2", "bi_rec_buf.rcx"),
            ProcessorConfig("RX81", "RX8", "play_buf.rcx"),
        ),
        zbus=False,
        connection="USB",
    ),

    "bi_play_rec": ModeConfig(
        processors=(
            ProcessorConfig("RP2", "RP2", "bi_play_rec_buf.rcx"),
            ProcessorConfig("RX81", "RX8", "bits.rcx"),
        ),
        zbus=False,
        connection="USB",
    ),

    "loctest_freefield": ModeConfig(
        processors=(
            ProcessorConfig("RP2", "RP2", "button.rcx"),
            ProcessorConfig("RX81", "RX8", "play_buf.rcx"),
        ),
        zbus=False,
        connection="USB",
    ),

    "loctest_headphones": ModeConfig(
        processors=(
            ProcessorConfig("RP2", "RP2", "bi_play_buf.rcx"),
            ProcessorConfig("RX81", "RX8", "bits.rcx"),
        ),
        zbus=False,
        connection="USB",
    ),

    "cam_calibration": ModeConfig(
        processors=(
            ProcessorConfig("RP2", "RP2", "button.rcx"),
            ProcessorConfig("RX81", "RX8", "bits.rcx"),
        ),
        zbus=False,
        connection="USB",
    ),
}
)


SETUPS = {
    "dome": DOME,
    "arc": ARC,
    "distance_array": DISTANCE_ARRAY,
    "headphones": HEADPHONES
}