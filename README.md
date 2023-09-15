# AI DJ Mix Generator

## Project Description

The AI DJ Mix Generator is an innovative tool that seeks to revolutionize the music mixing domain by integrating artificial intelligence and expert insights from professional DJing. Trained on XML metadata from Rekordbox - a professional DJ software by Pioneer DJ, and fine-tuned with a private music collection, this project not only aims to advance existing technologies but also to preserve the artistry in DJing by incorporating nuanced elements from a professional DJ's perspective. 

Part of the project, the LSTM model (`lstm_model.pth`), which is crucial for generating cue points, is extracted from the AI Cue Point Generator and will be updated consistently to enhance its performance and accuracy as it undergoes retraining on larger datasets. Additionally, the `scaler.joblib` is the tool utilized for preprocessing new data, ensuring the seamless integration and functioning of the model.

## Table of Contents

1. [Main Objective](#main-objective)
2. [AI-Generated Cue Points](#ai-generated-cue-points)
3. [AI DJ Mix Generator](#ai-dj-mix-generator)
4. [Techniques Employed](#techniques-employed)
5. [Getting Started](#getting-started)
6. [Contributing](#contributing)
7. [License](#license)
8. [Contact](#contact)

## Main Objective

The core objective of this project is to develop a tool that brings a fresh and automated approach to the DJing process while retaining the artistry and expertise that a professional DJ would incorporate in a live set. This tool aims to create seamless, unique, and high-quality mixes by automating several aspects that were previously done manually.

## AI-Generated Cue Points

This section outlines the development of the AI model responsible for generating cue points in tracks. The notebook involves the following steps:

1. **Importing Data from Rekordbox**: Data from 24 tracks are imported from Rekordbox, with cue point labels and timestamps retrieved for supervised training.
2. **Labels**: Timestamps are labeled as cue points or not, providing a supervised dataset for training the model.
3. **LSTM Model for AI Cue Point Generator**: An LSTM Neural Network is implemented using PyTorch to identify recurring patterns signifying cue points.

## AI DJ Mix Generator

This section encapsulates the functionalities and techniques incorporated in the AI DJ Mix Generator, including the following components:

1. **Preprocessing Step**: Involves initializing Track class objects with necessary attributes, including beats, BPM, audio file, and sample rate.
2. **Mod and BPM Detection**: Identifies the prevalent BPM and adjusts other tracks accordingly to match this tempo, ensuring a harmonized BPM across all tracks.
3. **Cue Point Detection**: Utilizes the model and scaler from the AI Cue Point Generator to identify significant cue points in each track.
4. **Randomness via Relevant Cue Point Selection & Randomness**: Incorporates a semi-random approach to select cue points for creating a unique and dynamic mix.
5. **Frequency Equalization of the Songs (EQing)**: Implements functions to adjust the frequency components of the songs to facilitate smoother transitions.
6. **Final Step: Generating a Mix**: Combines the adjusted audio files based on the determined cue points to generate a seamless mix.

## Techniques Employed

This section details the techniques utilized in both the AI Cue Point Generator and the AI DJ Mix Generator, encompassing:

### 4.3.1 Techniques for both AI Cue Point Generator and AI DJ Mix Generator

1. **Reading from WAV File**: Audio files are read from the "wav" file into an audio file, initiating the preprocessing of tracks.
2. **Beat Detection**: Essential beat information is retrieved using various libraries, with Madmom providing the most accurate results.
3. **BPM Estimation**: The BPM of songs is estimated, a crucial step for aligning tracks in the mixing process.
4. **Feature Extraction for Each Beat Timestamp**: Features are extracted for each beat timestamp using the Librosa library, including MFCC, Spectral Centroid, Spectral Contrast, Spectral Rolloff, Spectral Flux, and RMS.

## Getting Started

Ready to jump in? Here's how you can get started with the AI DJ Mix Generator!

1. **Clone the Repository**: Clone the project repository to your local machine.
   
2. **Setting Up Your Environment**: Before you begin, make sure to set up a virtual environment to manage dependencies. You can create a virtual environment using the following command:


python3 -m venv env

3. **Installing Dependencies**: Navigate to the project directory and install the necessary dependencies using the `requirements.txt` file. Run the following command in your terminal:

pip install -r requirements.txt

4. **Adding Your Music**: Create a folder with all the tracks you want to mix and add it to the current working directory.

5. **Running the Script**: Now, it's time to generate your mix! Run the `main.py` script and provide the path to your music folder when prompted. Like so:

python main.py

"Enter the path to the folder of music: " -> < path to folder goes here >  

## License

The project is licensed under the MIT license. You can find the license details in the [MITLicense.txt](MITLicense.txt) file in the project repository.

## Contact

ruszkowskifranciszek@gmail.com
