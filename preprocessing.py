# Import libraries

import librosa
import madmom
import os
from scipy.stats import mode
from madmom.features.downbeats import *


def load_track(wav_file):
    audio, sr = librosa.load(wav_file,sr = 44100)
    return audio, sr


def get_audio_files_from_path(path):
    """
    Returns a list of all .wav and .aif files from the specified path.

    Parameters:
    - path (str): The directory path to search for .wav and .aif files.

    Returns:
    - List of full paths to the .wav and .aif files in the directory.
    """
    return [
        os.path.join(path, file_name)
        for file_name in os.listdir(path)
        if file_name.endswith('.wav') or file_name.endswith('.aif')
    ]


def detect_beats_and_downbeats(audio_file, sr=44100):
    try:
        # Load the audio signal with the specified sample rate
        signal = madmom.audio.signal.Signal(audio_file, sample_rate=sr, num_channels=1)

        # Consider only 4/4 time signature
        proc = madmom.features.DBNDownBeatTrackingProcessor(beats_per_bar=[4], fps=100)
        act = madmom.features.RNNDownBeatProcessor()(signal)
        proc_res = proc(act)

        # Extract the beats and downbeats
        beats = proc_res[:, 0]

        # Filter downbeats (those with beat position 1)
        downbeats = [beat for beat in proc_res if beat[1] == 1]

        return beats, np.array(downbeats)
    except Exception as e:
        print(e)
        return np.array([]), np.array([])  # Return empty arrays for both beats and downbeats


def estimate_tempo_from_downbeats(downbeats):
    # Calculate the time difference between consecutive downbeats
    downbeat_differences = np.around(np.diff(downbeats[:, 0]), decimals=6)

    # Get the mode of the differences
    mod_diff = mode(downbeat_differences, keepdims=False).mode

    # Calculate the tempo: 60 seconds divided by the average difference
    # Since downbeat_differences are in seconds, this gives beats per minute
    tempo = 4 * (60 / mod_diff)

    try:
        # Attempt to round tempo normally
        tempo = round(tempo)
    except TypeError:
        # If that fails, try converting tempo to a scalar and then rounding
        if tempo.size == 1:
            tempo = round(tempo.item())
        else:
            print("Warning: Unexpected tempo value:", tempo)
            return None, mod_diff, downbeat_differences  # Return None if tempo is not a scalar

    return tempo, mod_diff, downbeat_differences


def extract_features(audio, sr, timestamp, window_size=1.0):
    """
    Extract multiple audio features for a given timestamp using librosa.

    Parameters:
    - audio: The audio data as a numpy array.
    - sr: Sample rate.
    - timestamp: The given timestamp (in seconds).
    - window_size: The size of the window around the timestamp (in seconds, default is 1 second).

    Returns:
    A dictionary with feature names as keys and extracted values as values.
    """

    # Define the start and end samples for the window
    center_sample = int(timestamp * sr)
    start_sample = center_sample - int(sr * window_size / 2)
    end_sample = center_sample + int(sr * window_size / 2)

    # Check for edge cases
    start_sample = max(start_sample, 0)
    end_sample = min(end_sample, len(audio))
    windowed_audio = audio[start_sample:end_sample]

    # If windowed_audio is too small, return an empty dictionary
    if len(windowed_audio) <= 1:
        return {}

    # Ensure even size for windowed_audio by padding with zero if necessary
    if len(windowed_audio) % 2 != 0:
        windowed_audio = np.append(windowed_audio, 0)

    # Extract MFCC
    mfcc_coeffs = librosa.feature.mfcc(y=windowed_audio, sr=sr, n_mfcc=13)

    # Extract Spectral Centroid
    spec_centroid = librosa.feature.spectral_centroid(y=windowed_audio, sr=sr)

    # Extract Spectral Contrast
    spec_contrast = librosa.feature.spectral_contrast(y=windowed_audio, sr=sr)

    # Extract Spectral Rolloff
    spec_rolloff = librosa.feature.spectral_rolloff(y=windowed_audio, sr=sr)

    # Extract Spectral Flux - this is the difference between consecutive spectral frames
    spectrum = librosa.stft(windowed_audio)
    flux = np.diff(np.abs(spectrum), axis=1)

    # Calculate RMS for the beat
    rms = librosa.feature.rms(y=windowed_audio)

    return {
        "MFCC": mfcc_coeffs.mean(axis=1),  # Averaging over time frames
        "Spectral Centroid": np.mean(spec_centroid),
        "Spectral Contrast": np.mean(
            spec_contrast, axis=1
        ),  # Averaging over frequency bands
        "Spectral Rolloff": np.mean(spec_rolloff),
        "Spectral Flux": np.mean(flux),
        "RMS": np.mean(rms),
    }
