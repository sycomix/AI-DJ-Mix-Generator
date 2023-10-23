import torch
from joblib import load

from preprocessing import *


class Track:

    _all_tracks = {}  # Class level dictionary to store all instances

    def __init__(self, name, wav_file):
        self.name = name
        self.wav_file = wav_file
        self.audio, self.sr = librosa.load(wav_file, sr=44100)
        self.downbeats = None
        self.beats = None
        self.tempo = None
        self.features = []  # For raw extracted features
        self.flat_features = []  # For flattened feature vectors
        self.x_tensor = None  # For the processed tensor
        self.cue_points_indeces = None
        self.cue_points = None

        # Store the instance in the class-level dictionary
        Track._all_tracks[self.name] = self

    def detect_beats_and_downbeats(self):
        self.beats, self.downbeats = detect_beats_and_downbeats(self.wav_file)

    def estimate_tempo_from_downbeats(self):
        self.tempo, _, _ = estimate_tempo_from_downbeats(self.downbeats)

    def extract_features_for_beats(self):
        for beat_time in self.beats:
            feature = extract_features(self.audio, self.sr, beat_time)
            self.features.append(feature)  # Appending to the list
            flat_feature = self.convert_to_feature_vector(feature)
            self.flat_features.append(flat_feature)

    @staticmethod
    def convert_to_feature_vector(beat_features):
        """Convert beat features into a flattened feature vector."""
        return (
            list(beat_features['MFCC'])
            + list(beat_features['Spectral Contrast'])
            + [beat_features['Spectral Centroid']]
            + [beat_features['Spectral Rolloff']]
            + [beat_features['Spectral Flux']]
            + [beat_features['RMS']]
        )

    def prepare_features_for_prediction(self):
        # Load the scaler and model
        scaler = load('scaler.joblib')
        X_new = [self.convert_to_feature_vector(f) for f in self.features]
        X_new = scaler.transform(X_new)  # Normalize
        self.x_tensor = torch.FloatTensor(X_new).unsqueeze(0)

    @classmethod
    def rename_tracks(cls):
        new_tracks = {}
        counter = {}

        for old_key, track in cls._all_tracks.items():
            new_key = old_key.split()[0]
            if new_key in counter:
                counter[new_key] += 1
                new_key = f"{new_key}{counter[new_key]}"
            else:
                counter[new_key] = 1
            new_tracks[new_key] = track

        cls._all_tracks = new_tracks

    @classmethod
    def get_all_tracks(cls):
        return cls._all_tracks

    @classmethod
    def clear_tracks(cls):
        """

        Clear all the tracks stored in the _all_tracks dictionary.
        """
        cls._all_tracks = {}


def instantiate_and_rename_tracks(file_list):
    """
    Instantiates Track objects for each .wav file in the provided list and renames them.

    Parameters:
    - file_list (list): List of full paths to the .wav files.

    Returns:
    - List of Track objects.
    """
    # Instantiation
    for wav_file in file_list:
        # Extract the base name of the file without extension
        track_name = os.path.splitext(os.path.basename(wav_file))[0]
        Track(track_name, wav_file)  # This will automatically add the track to _all_tracks

    # Rename tracks
    Track.rename_tracks()

    # Access and return the tracks
    return Track.get_all_tracks()


def preprocess_track(track):
    # Detect beats and downbeats
    track.detect_beats_and_downbeats()

    # Estimate tempo from downbeats
    track.estimate_tempo_from_downbeats()

    print(f"Estimated Tempo for {track.name}: {track.tempo:.2f} BPM")



