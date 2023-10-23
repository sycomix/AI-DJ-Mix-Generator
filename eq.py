# Import libraries

import yodel.filter
import numpy as np
import math
import random


def lin2db(linval):
    return 20.0 * math.log10(linval) if linval > 1e-5 else -100.0


def db2lin(dbval):
    return math.pow(10, dbval / 20.0)


def adjust_highs(audio, sr, gain_db):
    # High shelf filter starting from 2kHz
    filter_obj = yodel.filter.Biquad()
    filter_obj.high_shelf(sr, 2000, 1, gain_db)  # 2000 Hz
    output = np.zeros_like(audio)
    filter_obj.process(audio, output)
    return output


def adjust_lows(audio, sr, gain_db):
    # Low shelf filter ending at 500Hz
    filter_obj = yodel.filter.Biquad()
    filter_obj.low_shelf(sr, 500, 1, gain_db)  # 500 Hz
    output = np.zeros_like(audio)
    filter_obj.process(audio, output)
    return output


def adjust_audio(audio, sr, gains):
    # gains should be a list of two values: [gain_lows, gain_highs]
    # Each gain value should be between -1 and 1, where -1 corresponds to -∞ dB (muted) and 1 corresponds to 0 dB (normal level)

    if len(audio) == 0:

        return audio

    # Convert gains to dB
    gain_lows_db = 20 * np.log10(gains[0]) if gains[0] > 0 else -np.inf
    gain_highs_db = 20 * np.log10(gains[1]) if gains[1] > 0 else -np.inf

    # Adjust audio using the defined gains
    audio = adjust_lows(audio, sr, gain_lows_db)
    audio = adjust_highs(audio, sr, gain_highs_db)

    # Rescale audio to be within [-1, 1]
    max_val = np.max(np.abs(audio))
    if max_val != 0:
        audio = audio / max_val

    return audio


# Rapid EQ

def rapid_eq(audio, sr, gains, cue_times, chunk_size=44100):
    # Calculate the cue samples
    cue_samples = [int(t * sr) for t in cue_times]

    # Create a new array to hold the processed audio
    processed_audio = np.zeros_like(audio)

    # Apply the EQ changes in chunks
    for i in range(len(cue_samples) - 1):
        # Calculate the slope of the EQ change for this period
        slope_lows = (gains[i + 1][0] - gains[i][0]) / (cue_samples[i + 1] - cue_samples[i])
        slope_highs = (gains[i + 1][1] - gains[i][1]) / (cue_samples[i + 1] - cue_samples[i])

        for j in range(cue_samples[i], cue_samples[i + 1], chunk_size):
            # Calculate the current time in seconds
            current_time = j / sr

            # Calculate the current gains
            current_gain_lows = gains[i][0] + slope_lows * (current_time - cue_times[i])
            current_gain_highs = gains[i][1] + slope_highs * (current_time - cue_times[i])

            # Ensure the gains never drop below a small positive value
            current_gain_lows = max(current_gain_lows, 0.01)
            current_gain_highs = max(current_gain_highs, 0.01)

            # Calculate the chunk end sample
            end_j = min(j + chunk_size, len(audio))

            # Apply the EQ changes
            processed_audio[j:end_j] = adjust_audio(audio[j:end_j], sr, [current_gain_lows, current_gain_highs])

    # After the last cue time, keep the EQ changes at the last gains
    if cue_samples[-1] < len(audio):
        processed_audio[cue_samples[-1]:] = adjust_audio(audio[cue_samples[-1]:], sr, gains[-1])

    return processed_audio


# Smooth EQ

def sigmoid(x, slope=10):
    return 1 / (1 + np.exp(-slope * x))


def smooth_eq(audio, sr, gains, cue_times, chunk_size=44100):
    # Calculate the cue samples
    cue_samples = [int(t * sr) for t in cue_times]

    # Create a new array to hold the processed audio
    processed_audio = np.zeros_like(audio)

    # Use the sigmoid function with the slope to calculate the current gains
    slope = 5  # adjust this value as desired
    # Apply the EQ changes in chunks
    for i in range(len(cue_samples) - 1):

        # Calculate the length of the transition in samples
        transition_length = cue_samples[i + 1] - cue_samples[i]

        for j in range(cue_samples[i], cue_samples[i + 1], chunk_size):
            # Calculate the proportion of the transition that has elapsed,
            # adjusted to go from -2 to 2 rather than from 0 to 1
            transition_progress = 4 * (j - cue_samples[i]) / transition_length - 2

            gain_transition = sigmoid(transition_progress, slope)
            current_gain_lows = gains[i][0] + (gains[i + 1][0] - gains[i][0]) * gain_transition
            current_gain_highs = gains[i][1] + (gains[i + 1][1] - gains[i][1]) * gain_transition

            # Ensure the gains never drop below a small positive value
            current_gain_lows = max(current_gain_lows, 0.01)
            current_gain_highs = max(current_gain_highs, 0.01)

            # Calculate the chunk end sample
            end_j = min(j + chunk_size, len(audio))

            # Apply the EQ changes
            processed_audio[j:end_j] = adjust_audio(audio[j:end_j], sr, [current_gain_lows, current_gain_highs])

    #print(f"smooth_eq - iteration {i}, processed_audio length after: {len(processed_audio)}")
    # After the last cue time, keep the EQ changes at the last gains
    if cue_samples[-1] < len(audio):
        processed_audio[cue_samples[-1]:] = adjust_audio(audio[cue_samples[-1]:], sr, gains[-1])

    return processed_audio


# Creating EQ-ied tracks

def beats_to_seconds(bpm, beats):
    beats_per_second = bpm / 60
    return beats / beats_per_second


def generate_track_cue_in_out(tracks):
    """Generate cue in and cue out points for tracks."""
    # For the first track (only cue out)
    aout_index = random.randint(6, 8)
    acue_out = tracks[0].cue_points[aout_index]
    cue_in_out_points = [(None, acue_out)]
    # For the middle tracks (both cue in and cue out)
    for i in range(1, len(tracks) - 1):
        bin_index = random.randint(0, 2)
        bout_index = random.randint(6, 8)
        bcue_in = tracks[i].cue_points[bin_index]
        bcue_out = tracks[i].cue_points[bout_index]
        cue_in_out_points.append((bcue_in, bcue_out))

    # For the last track (only cue in)
    cin_index = random.randint(0, 2)
    ccue_in = tracks[-1].cue_points[cin_index]
    cue_in_out_points.append((ccue_in, None))

    return cue_in_out_points


def generate_bass_cue_points(cue_in_out_points, tracks):
    """Generate bass up and down points for tracks."""
    # Initial values
    #y = random.choice([1, 2, 3])
    y = random.choice([2, 3])

    #z = random.choice([4, 8, 16, 32])
    z = random.choice([4, 8])

    x = beats_to_seconds(tracks[0].tempo, z)

    # First track has only bass down
    acue_out = cue_in_out_points[0][1]
    bin_index = tracks[1].cue_points.index(cue_in_out_points[1][0])
    bbass_up_index = bin_index + y
    bbass_up = tracks[1].cue_points[bbass_up_index]
    abass_down = acue_out + (bbass_up - tracks[1].cue_points[bin_index]) - x
    bass_cue_points = [(None, abass_down)]
    # Middle tracks
    for i in range(1, len(cue_in_out_points) - 1):
        bcue_in = cue_in_out_points[i][0]
        bcue_out = cue_in_out_points[i][1]

        y = random.choice([2, 3])
        z = random.choice([4, 8])
        x = beats_to_seconds(tracks[i].tempo, z)

        bin_index = tracks[i].cue_points.index(bcue_in)
        bbass_up_index = bin_index + y
        bbass_up = tracks[i].cue_points[bbass_up_index]

        cin_index = tracks[i+1].cue_points.index(cue_in_out_points[i+1][0])
        cbass_up_index = cin_index + y
        cbass_up = tracks[i+1].cue_points[cbass_up_index]

        bbass_down = bcue_out + (cbass_up - tracks[i+1].cue_points[cin_index]) - x

        bass_cue_points.append((bbass_up, bbass_down))

    # Last track has only bass up
    ccue_in = cue_in_out_points[-1][0]
    cin_index = tracks[-1].cue_points.index(ccue_in)
    cbass_up_index = cin_index + y
    cbass_up = tracks[-1].cue_points[cbass_up_index]
    bass_cue_points.append((cbass_up, None))

    return bass_cue_points


def generate_all_cue_points(tracks):
    """Generate all cue and bass points for tracks."""

    # Step 1: Get the cue in/out points
    cue_in_out_points = generate_track_cue_in_out(tracks)

    # Step 2: Get the bass cue points
    bass_cue_points = generate_bass_cue_points(cue_in_out_points, tracks)

    # Step 3: Combine and return the results
    combined_cue_points = [(cue[0], cue[1], bass[0], bass[1]) for cue, bass in zip(cue_in_out_points, bass_cue_points)]

    return cue_in_out_points, bass_cue_points, combined_cue_points


def generate_cue_points_matrix(tracks):
    """Generate a matrix of cue and bass points for tracks."""

    # Get all cue points using the existing function
    cue_in_out_points, bass_cue_points, combined_cue_points = generate_all_cue_points(tracks)

    # Create a matrix with header and placeholder rows
    matrix = [
        ["Track", "Cue-In", "Cue-Out", "Bass-Up", "Bass-Down"],
        *[
            ['None', 'None', 'None', 'None', 'None']
            for _ in range(len(tracks))
        ],
    ]
    for i in range(len(cue_in_out_points)):
        track_name = f"Track {i+1}"
        cue_in, cue_out = cue_in_out_points[i]
        bass_up, bass_down = bass_cue_points[i]

        # Update the row instead of appending
        matrix[i+1] = [track_name, cue_in, cue_out, bass_up, bass_down]

    return cue_in_out_points, bass_cue_points, combined_cue_points, matrix


# Bass EQ


def create_eq_adjusted_tracks(track1, track2, t1_bass, t2_bass, sr):

    # Define the gains and cue times for track1 (the master)
    gains1_bass = [[1, 1], [0.1, 1]]
    cue_times1_bass = [0, t1_bass]

    # Define the gains and cue times for track2 (the slave)
    gains2_bass = [[0.1, 1], [1, 1]]
    cue_times2_bass = [0, t2_bass]

    # Apply the EQ changes to the tracks
    track1_eq_bass = rapid_eq(track1, sr, gains1_bass, cue_times1_bass)
    track2_eq_bass = rapid_eq(track2, sr, gains2_bass, cue_times2_bass)

    return track1_eq_bass, track2_eq_bass


def generate_eq_adjusted_tracks(tracks, bass_cues, sr):
    # For the first track (only bass down)
    first_track_eq, _ = create_eq_adjusted_tracks(tracks[0], tracks[1], bass_cues[0], bass_cues[1], sr)
    eq_adjusted_tracks = [first_track_eq]
    # For the middle tracks (both bass up and bass down)
    for i in range(1, len(tracks) - 1):

        # Adjust bass up
        _, track_with_bass_up = create_eq_adjusted_tracks(tracks[i-1], tracks[i], bass_cues[2*i-2], bass_cues[2*i-1], sr)

        # Adjust bass down for the same track
        track_with_bass_down, _ = create_eq_adjusted_tracks(track_with_bass_up, tracks[i+1], bass_cues[2*i], bass_cues[2*i+1], sr)

        eq_adjusted_tracks.append(track_with_bass_down)

    # For the last track (only bass up)
    _, last_track_eq = create_eq_adjusted_tracks(tracks[-2], tracks[-1], bass_cues[-2], bass_cues[-1], sr)
    eq_adjusted_tracks.append(last_track_eq)

    return eq_adjusted_tracks


# Treble EQ


def track_fade_out(track, t_treble, duration, sr):
    """Adjust treble for the track fading out."""
    gains = [[1, 1],[1, 1] ,[1, 0.8], [1, 0.6], [1, 0.4], [1, 0.2], [0.01, 0.01]]
    cue_times = [0, t_treble, t_treble +0.1*duration, t_treble + 0.2*duration, t_treble + 0.4*duration, t_treble + 0.8*duration, t_treble + duration+10]
    print(f"track_fade_out duration: {duration}")
    return smooth_eq(track, sr, gains, cue_times)


def track_fade_in(track, t_treble, duration, sr):
    """Adjust treble for the track fading in."""
    gains = [[0.01, 0.01], [0.01, 0.01], [1, 0.8], [1, 1], [1, 1], [1, 1]]
    cue_times = [0, t_treble, t_treble + 0.2*duration, t_treble + 0.4*duration, t_treble + 0.8*duration, t_treble + duration+10]
    print(f"track_fade_in duration: {duration}")
    return smooth_eq(track, sr, gains, cue_times)


def generate_treble_adjusted_tracks(tracks, cue_matrix, sr):
    # For the first track (only fade out)

    # First duration is difference between
    duration_out_first = float(cue_matrix[2][3] if cue_matrix[2][3] else 0) - float(cue_matrix[2][1] if cue_matrix[2][1] else 0)

    # it starts at first tracks cue out
    treble_out_first = float(cue_matrix[1][2] if cue_matrix[1][2] else 0)

    #print(f"First track - duration_out: {duration_out_first}")

    first_track_treble = track_fade_out(tracks[0], treble_out_first, duration_out_first, sr)
    treble_adjusted_tracks = [first_track_treble]
    for i in range(1, len(tracks) - 1):  # This will loop over the 2nd and 3rd tracks in the tracks list
        # For fade-in of the track
        duration_in = float(cue_matrix[i+1][2] if cue_matrix[i+1][2] else 0) - float(cue_matrix[i+1][1] if cue_matrix[i+1][1] else 0)
        treble_in = float(cue_matrix[i+1][1] if cue_matrix[i+1][1] else 0)
        treble_adjusted_track_in = track_fade_in(tracks[i], treble_in, duration_in, sr)

        # For fade-out of the track
        duration_out = float(cue_matrix[i+2][3] if cue_matrix[i+2][3] else 0) - float(cue_matrix[i+2][1] if cue_matrix[i+2][1] else 0)
        treble_out = float(cue_matrix[i+1][2] if cue_matrix[i+1][2] else 0)
        treble_adjusted_track_out = track_fade_out(treble_adjusted_track_in, treble_out, duration_out, sr)

        treble_adjusted_tracks.append(treble_adjusted_track_out)

        # That ends the iteration


    # For the last track (only fade in)
    # Difference between its bass up and cue in - 5 seconds) so its loud 5 seconds before it drops
    duration_in_last = float(cue_matrix[-1][3] if cue_matrix[-1][3] else 0) - float(cue_matrix[-1][1] if cue_matrix[-1][1] else 0)
    # tracks cue in
    treble_in_last = float(cue_matrix[-1][1] if cue_matrix[-1][1] else 0)

    last_track_treble = track_fade_in(tracks[-1], treble_in_last, duration_in_last, sr)
    treble_adjusted_tracks.append(last_track_treble)

    return treble_adjusted_tracks
