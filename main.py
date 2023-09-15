
import soundfile as sf

from preprocessing import *
from track import *
from bpm import *
from cuepoints import *
from eq import *
from mix import *


scaler = load('scaler.joblib')
model = LSTMNet(input_dim, hidden_dim, output_dim, num_layers)
model.load_state_dict(torch.load("lstm_model.pth"))
model.eval()  # Set the model to evaluation mode


def main_processing_function(path):
    # Get WAV files
    track_files = get_audio_files_from_path(path)
    print(track_files)

    # Instantiate and preprocess tracks
    tracks = instantiate_and_rename_tracks(track_files)
    for track_name, track_obj in tracks.items():
        preprocess_track(track_obj)

    # Print all BPM values
    for track_name, track_obj in tracks.items():
        print(f"{track_name}: {track_obj.tempo} BPM")
    print(f"Mode BPM: {get_mode_bpm(tracks)} BPM")

    # Adjust tracks to mode BPM
    tracks = adjust_tracks_to_mode_bpm(tracks)

    # Extract features for beats
    for track_name, track in tracks.items():
        track.extract_features_for_beats()

    # Cue point detection
    for track_name, track in tracks.items():
        track.prepare_features_for_prediction()
        # First run to determine the dominant array
        detect_cue_points_for_track(track, num_cue_points=64)
        dominant_key = categorize_cue_points(track.cue_points_indeces)

        # Second run with filtering
        detect_cue_points_for_track(track, num_cue_points=12, filter_by=dominant_key)

        print(f"Track: {track_name}")
        print(f"Cue Points (indices): {track.cue_points_indeces}")
        print(f"Cue Points: {track.cue_points}")
        print("------")

    # Generate the cue points matrix
    track_list = [tracks[key] for key in tracks]

    #cue_in_out_points, bass_cue_points, _, _ = generate_cue_points_matrix(track_list)

    # Generate the cue points matrix
    cue_in_out_points, bass_cue_points, combined_cue_points, matrix = generate_cue_points_matrix(track_list)

    flattened_cue_in_out = [item for sublist in cue_in_out_points for item in sublist if item is not None]
    flattened_bass_cues = [item for sublist in bass_cue_points for item in sublist if item is not None]

    print(matrix)

    # EQ adjustments
    audio_list = [track.audio for track in track_list]
    eqbass = generate_eq_adjusted_tracks(audio_list, flattened_bass_cues, track_list[0].sr)

    # Treble adjustments
    treble_tracks = generate_treble_adjusted_tracks(eqbass, matrix, 44100)

    # Combine tracks
    combined = combine_multiple_tracks(treble_tracks, flattened_cue_in_out, track_list[0].sr)
    transition_points,nonform = calculate_transition_points(flattened_cue_in_out)

    print(transition_points)  # This will print the transition points in the format minute:seconds
    print(nonform)

    sf.write('output.wav', combined, 44100)

    return combined, treble_tracks, eqbass, flattened_cue_in_out, flattened_bass_cues, matrix, tracks


if __name__ == "__main__":
    # Get the input path from the user
    while True:
        path = input("Enter the path to the folder of music: ")
        if os.path.exists(path) and os.path.isdir(path):
            break
        else:
            print("The provided path does not exist or is not a directory. Please try again.")

    # Call the main processing function with the user-provided path
    combined, treble_tracks, eqbass, flattened_cue_in_out, flattened_bass_cues, matrix, tracks = main_processing_function(path)

    # Optionally, you can provide some feedback or additional actions here
    print("Mixing complete!")

    # Save the mix to the output.wav file in the current directory
    sf.write('output.wav', combined, 44100)

