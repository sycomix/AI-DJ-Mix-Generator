import numpy as np

# Creating the Mix


def overlap_tracks(master, slave, master_cue, slave_cue, sr):
    # Convert the cue points to sample indices
    master_cue_samples = int(master_cue * sr)
    slave_cue_samples = int(slave_cue * sr)

    # Extract portions of each track
    master_before_cue = master[:master_cue_samples]
    slave_from_cue = slave[slave_cue_samples:]

    # Calculate required padding for the slave track
    padding_length = master_cue_samples - slave_cue_samples
    padding = np.zeros(padding_length)

    # Pad the slave track to align with the master's cue point
    slave_padded = np.concatenate([padding, slave_from_cue])

    # Combine the master track and padded slave track
    combined_length = max(len(master), len(slave_padded))
    combined = np.zeros(combined_length)

    combined[:len(master)] += master
    combined[:len(slave_padded)] += slave_padded

    return combined


def combine_multiple_tracks(tracks, cues, sr):
    combined_mix = tracks[0]  # Start with the first track

    # Calculate where in the mix each track should start
    cumulative_cue = cues[0]  # Start with Track 1 cue out

    # Loop until the second last track
    for i in range(1, len(tracks) - 1):
        master_cue = cumulative_cue
        slave_cue = cues[2*(i-1) + 1]

        combined_mix = overlap_tracks(combined_mix, tracks[i], master_cue, slave_cue, sr)

        # Update cumulative_cue for the next iteration
        cumulative_cue += cues[2*i] - slave_cue

    # Handle the last track separately
    last_track_cue_in = cues[-1]
    combined_mix = overlap_tracks(combined_mix, tracks[-1], cumulative_cue + last_track_cue_in, last_track_cue_in, sr)

    return combined_mix


def calculate_transition_points(cues):
    transition_points = []

    # Start with the first track's cue out
    cumulative_cue = cues[0]
    transition_points.append(cumulative_cue)

    # Loop to calculate subsequent transition points
    for i in range(1, len(cues) // 2):
        cumulative_cue += cues[2*i] - cues[2*(i-1) + 1]
        transition_points.append(cumulative_cue)

    # Convert transition points into the desired format (minutes:seconds)
    formatted_transitions = [f"{int(tp // 60)}:{int(tp % 60):02}" for tp in transition_points]
    nonform = [tp for tp in transition_points]

    return formatted_transitions, nonform

