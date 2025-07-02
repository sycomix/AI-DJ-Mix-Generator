# Import libraries

from bisect import bisect_left

import numpy as np
import torch
from bisect import bisect_left


def get_cue_point_timestamps(track, cue_point_indices):
    return [track.beats[i] for i in cue_point_indices]


def init_hidden(batch_size, hidden_dim, num_layers):
    return (torch.zeros(num_layers, batch_size, hidden_dim),
            torch.zeros(num_layers, batch_size, hidden_dim))


def categorize_cue_points(cue_points):
    # Define the four arrays
    arrays = {
        0: [],
        1: [],
        2: [],
        3: []
    }

    for point in cue_points:
        remainder = point % 4
        arrays[remainder].append(point)

    # Find the array with the most cue points
    max_key = max(arrays, key=lambda k: len(arrays[k]))

    return max_key


def detect_cue_points_for_track(track, model, hidden_dim, num_layers, num_cue_points=12, filter_by=None):
    with torch.no_grad():
        h0, c0 = init_hidden(1, hidden_dim, num_layers)  # Batch size of 1
        outputs, _ = model(track.x_tensor, (h0, c0))
        predicted_cues = outputs.flatten().tolist()

    # If filtering is applied
    if filter_by is not None:
        filtered_indices = [i for i in range(len(predicted_cues)) if i % 4 == filter_by]
        filtered_cues = [predicted_cues[i] for i in filtered_indices]
    else:
        filtered_cues = predicted_cues
        filtered_indices = list(range(len(predicted_cues)))

    # Split the track into n intervals and get the top cue point in each interval
    interval_length = len(predicted_cues) // num_cue_points  # We use the full track's length
    cue_indices = []

    for i in range(num_cue_points):
        start = i * interval_length
        end = (i + 1) * interval_length

        # Get the corresponding start and end indices for the filtered cues
        filtered_start = bisect_left(filtered_indices, start)
        filtered_end = bisect_left(filtered_indices, end)

        # Clip the end if it exceeds the filtered cues length
        filtered_end = min(filtered_end, len(filtered_cues) - 1)

        interval_values = filtered_cues[filtered_start:filtered_end]

        # Skip if interval_values is empty
        if not interval_values:
            continue

        max_index_in_interval = np.argmax(interval_values)
        original_index = filtered_indices[filtered_start + max_index_in_interval]
        cue_indices.append(original_index)

    track.cue_points_indeces = np.array(cue_indices)
    track.cue_points = get_cue_point_timestamps(track, track.cue_points_indeces)
