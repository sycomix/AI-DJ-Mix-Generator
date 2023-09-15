# Import libraries

import librosa
import librosa.display
import pyrubberband as pyrb
import soundfile as sf
import random
# Import Functions
from preprocessing import *
from track import *


def adjust_tempo_pyrb(input_file, output_file, tempo_ratio):
    # Load the audio
    audio, sr = librosa.load(input_file, sr=44100)

    # Time stretch the audio
    audio_adjusted = pyrb.time_stretch(audio, sr, tempo_ratio)

    # Normalize the amplitude
    #audio_adjusted = audio_adjusted / np.max(np.abs(audio_adjusted))

    # Write the adjusted audio to a .wav file
    sf.write(output_file, audio_adjusted, sr)

    return audio_adjusted, output_file


def calculate_tempo_ratio(master, slave):
    ratio = master.tempo/slave.tempo
    return ratio


def adjust_tempo_and_analyze(master_key, slave_key, tracks_dict):
    master = tracks_dict[master_key]
    slave = tracks_dict[slave_key]

    # Calculate the tempo ratio
    tempo_ratio = calculate_tempo_ratio(master, slave)

    # Adjust the tempo of the slave track using pyrubberband
    adjusted_audio, output_file = adjust_tempo_pyrb(slave.wav_file, f"{slave_key}_AT_{master.tempo}bpm.wav", tempo_ratio)

    # Create a new Track instance for the adjusted audio
    adjusted_slave = Track(f"{slave_key}_AT_{master.tempo}bpm", output_file)

    # Add the adjusted track to the tracks dictionary
    tracks_dict[f"{slave_key}_AT_{master.tempo}bpm"] = adjusted_slave

    # Preprocess the adjusted track
    preprocess_track(adjusted_slave)
    #detect_cue_points_for_track(adjusted_slave, num_cue_points=12)

    return adjusted_slave


def get_mode_bpm(tracks):
    """
    Get the mode BPM value from a dictionary of Track objects.

    Parameters:
    - tracks: Dictionary of Track objects.

    Returns:
    - mode_bpm: The mode BPM value.
    """
    bpm_values = [track.tempo for track in tracks.values()]
    mode_value = mode(bpm_values).mode
    if np.isscalar(mode_value):
        return mode_value
    else:
        return mode_value[0]


def adjust_tracks_to_mode_bpm(tracks):
    """
    Adjusts the BPM of tracks to match the mode BPM.

    Parameters:
    - tracks: Dictionary of Track objects.

    Returns:
    - same_bpm_tracks: Dictionary containing tracks that have been adjusted to have the mode BPM.
    """
    mode_bpm_value = get_mode_bpm(tracks)

    # Get tracks with mode BPM and choose a random one as the master
    tracks_with_mode_bpm = [name for name, track in tracks.items() if track.tempo == mode_bpm_value]
    master_key = random.choice(tracks_with_mode_bpm)

    same_bpm_tracks = {}  # Dictionary to store tracks with the same BPM

    # Iterate over a copy of the tracks dictionary to avoid RuntimeError
    for track_name, track_obj in tracks.copy().items():
        if track_obj.tempo != mode_bpm_value:
            adjusted_track = adjust_tempo_and_analyze(master_key, track_name, tracks)
            same_bpm_tracks[adjusted_track.name] = adjusted_track
        else:
            same_bpm_tracks[track_name] = track_obj

    return same_bpm_tracks
