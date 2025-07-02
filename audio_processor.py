import torch
import torch.nn as nn
from joblib import load
import os

import bpm
from cuepoints import *
import eq
from mix import *
import preprocessing
import track as track_module
import soundfile as sf

# Define the LSTM model (moved from cuepoints.py)
class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hc):
        h0, c0 = hc
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out, (hn, cn)

# Load the scaler and model (from main.py and cuepoints.py)
scaler = load('scaler.joblib')
input_dim = 24  # Corrected based on original model
hidden_dim = 10 # Corrected based on original model
output_dim = 1 # Corrected based on original model
num_layers = 1 # Corrected based on original model

model = LSTMNet(input_dim, hidden_dim, output_dim, num_layers)
model.load_state_dict(torch.load("lstm_model.pth"))
model.eval() # Set the model to evaluation mode

def process_audio_files(file_paths):
    # Get WAV files
    # In the GUI context, file_paths will be directly passed from the QFileDialog
    # So, we don't need preprocessing.get_audio_files_from_path(path) here.
    # We'll assume file_paths is already a list of absolute paths.
    track_files = file_paths
    print(f"Processing audio files: {track_files}")

    # Instantiate and preprocess tracks
    tracks = track_module.instantiate_and_rename_tracks(track_files)
    for track_name, track_obj in tracks.items():
        track_module.preprocess_track(track_obj)

    # Print all BPM values
    for track_name, track_obj in tracks.items():
        print(f"{track_name}: {track_obj.tempo} BPM")
    print(f"Mode BPM: {bpm.get_mode_bpm(tracks)} BPM")

    # Adjust tracks to mode BPM
    tracks = bpm.adjust_tracks_to_mode_bpm(tracks)

    # Extract features for beats
    for track_name, track in tracks.items():
        track.extract_features_for_beats()

    # Cue point detection
    for track_name, track in tracks.items():
        track.prepare_features_for_prediction()
        # First run to determine the dominant array
        detect_cue_points_for_track(track, model, hidden_dim, num_layers, num_cue_points=64)
        print(f"Length of cue_points after first detection: {len(track.cue_points)}")
        dominant_key = categorize_cue_points(track.cue_points_indeces)

        # Second run with filtering
        detect_cue_points_for_track(track, model, hidden_dim, num_layers, num_cue_points=64, filter_by=dominant_key)

        print(f"Track: {track_name}")
        print(f"Cue Points (indices): {track.cue_points_indeces}")
        print(f"Cue Points: {track.cue_points}")
        print("------")

    # Generate the cue points matrix
    track_list = [tracks[key] for key in tracks]

    # Generate the cue points matrix
    cue_in_out_points, bass_cue_points, combined_cue_points, matrix = eq.generate_cue_points_matrix(track_list)

    flattened_cue_in_out = [item for sublist in cue_in_out_points for item in sublist if item is not None]
    flattened_bass_cues = [item for sublist in bass_cue_points for item in sublist if item is not None]

    print(matrix)

    # EQ adjustments
    audio_list = [track.audio for track in track_list]
    eqbass = eq.generate_eq_adjusted_tracks(audio_list, flattened_bass_cues, track_list[0].sr)

    # Treble adjustments
    treble_tracks = eq.generate_treble_adjusted_tracks(eqbass, matrix, 44100)

    # Combine tracks
    combined = combine_multiple_tracks(treble_tracks, flattened_cue_in_out, track_list[0].sr)
    transition_points, nonform = calculate_transition_points(flattened_cue_in_out)

    print(transition_points)  # This will print the transition points in the format minute:seconds
    print(nonform)

    sf.write('output.wav', combined, 44100)

    return combined, treble_tracks, eqbass, flattened_cue_in_out, flattened_bass_cues, matrix, tracks