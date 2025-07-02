import bpm
import os
from cuepoints import *
import eq
from mix import *
import preprocessing
import track as track_module
import soundfile as sf

# These are now loaded in audio_processor.py
# scaler = load('scaler.joblib')
# model = LSTMNet(input_dim, hidden_dim, output_dim, num_layers)
# model.load_state_dict(torch.load("lstm_model.pth"))
# model.eval()  # Set the model to evaluation mode