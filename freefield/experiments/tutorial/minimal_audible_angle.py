import random
import time

import freefield
import slab


# Initialize the loudspeaker array and response button.
freefield.initialize(
    setup="arc",
    device=[
        ["RX81", "RX8", "play_buf.rcx"],
        ["RX82", "RX8", "play_buf.rcx"],
        ["RP2", "RP2", "button.rcx"],
    ],
)

# Experimental parameters.
reference_azimuth = 0
speaker_spacing = 4.28
elevation = 0
distance = 1.4
isi = 0.3

# Use the same stimulus in every interval.
stimulus = slab.Sound.pinknoise(duration=0.2)

freefield.write(
    tag="playbuflen",
    value=stimulus.n_samples,
    processors=["RX81", "RX82"],
)

# Find the loudspeaker at the reference position.
reference_speaker = freefield.pick_speakers(
    (reference_azimuth, elevation, distance)
)[0]

# The staircase controls the angular separation between
# the reference and deviant loudspeaker.
staircase = slab.Staircase(
    start_val=7 * speaker_spacing,
    n_reversals=10,
    step_sizes=[
        2 * speaker_spacing,
        speaker_spacing,
    ],
    n_up=1,
    n_down=2,
    min_val=speaker_spacing,
    max_val=7 * speaker_spacing,
)

for separation in staircase:

    # Randomly place the deviant to the left or right
    # of the reference position.
    direction = random.choice([-1, 1])

    deviant_azimuth = (
        reference_azimuth
        + direction * separation
    )

    deviant_speaker = freefield.pick_speakers(
        (deviant_azimuth, elevation, distance)
    )[0]

    # Randomly select which of the three intervals contains the deviant.
    target_interval = random.randrange(3)

    trial_speakers = [
        reference_speaker,
        reference_speaker,
        reference_speaker,
    ]

    trial_speakers[target_interval] = deviant_speaker

    # Present the three intervals.
    for speaker in trial_speakers:

        freefield.set_signal_and_speaker(
            signal=stimulus.data.flatten(),
            speaker=speaker,
        )

        freefield.play()

        time.sleep(isi)

    # Ask the participant which interval contained the deviant.
    freefield.wait_for_button(
        proc="RP2",
        tag="response",
    )

    response = freefield.read(
        tag="response",
        processor="RP2",
    )

    # This assumes that the response buttons are encoded as 1, 2, and 3.
    correct_response = target_interval + 1

    correct = response == correct_response

    staircase.add_response(correct)

print(
    f"Estimated MAA: {staircase.threshold():.2f} degrees"
)






