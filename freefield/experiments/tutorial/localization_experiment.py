import time

import freefield
import slab


# Initialize the free-field setup.
# Sensor tracking is used to record the perceived sound direction.
freefield.initialize(
    setup="dome",
    default="loctest_freefield",
    sensor_tracking=True,
)

# Select loudspeakers located in the horizontal plane.
speaker_indices = [1, 2, 3, 4, 5, 6, 7]

# Create a randomized trial sequence.
n_reps = 4
sequence = slab.Trialsequence(
    conditions=speaker_indices,
    n_reps=n_reps,
    kind="non_repeating",
)

# Use the same broadband stimulus for every trial.
stimulus = slab.Sound.pinknoise(duration=0.5)

# The buffer length is identical for every trial and only needs to be set once.
freefield.write(
    tag="playbuflen",
    value=stimulus.n_samples,
    processors=["RX81", "RX82"],
)

freefield.play_start_sound()

for speaker_index in sequence:

    # The participant presses the button when the head is returned to center position
    freefield.wait_for_button(
        proc="RP2",
        tag="response",
    )

    speaker = freefield.pick_speakers(speaker_index)[0]

    # Present the stimulus from the target loudspeaker.
    freefield.set_signal_and_speaker(
        signal=stimulus.data.flatten(),
        speaker=speaker,
    )
    freefield.play()

    # The participant turns their head towards the perceived sound location
    # and presses the response button.
    freefield.wait_for_button(
        proc="RP2",
        tag="response",
    )

    response = freefield.get_head_pose(
        method="sensor",
        convention="psychoacoustics",
    )

    sequence.add_response(response)