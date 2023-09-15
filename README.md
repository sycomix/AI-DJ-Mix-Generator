# AI DJ Mix Generator

## Project Description

The AI DJ Mix Generator is a groundbreaking tool that combines artificial intelligence and expert insights from the DJing realm. Leveraging XML metadata from Rekordbox, a renowned DJ software by Pioneer DJ, and honed with a private music collection, this project is not just advancing existing technologies but also safeguarding the artistry intrinsic to DJing.

Part of this project, the `lstm_model.pth`, a critical component for generating cue points, is derived from the AI Cue Point Generator. This model undergoes regular updates to enhance its precision and performance, drawing from continuous retraining on larger datasets. The `scaler.joblib` plays a vital role in preprocessing new data, ensuring the model's seamless operation and integration.

## Table of Contents

1. [Main Objective](#main-objective)
2. [AI-Generated Cue Points](#ai-generated-cue-points)
3. [AI DJ Mix Generator](#ai-dj-mix-generator)
4. [Techniques Employed](#techniques-employed)
5. [Getting Started](#getting-started)
6. [Contributing](#contributing)
7. [License](#license)
8. [Contact](#contact)
9. [Demo Links](#demo-links)

## Main Objective

The main goal here is to craft a tool that injects a fresh, automated twist into the DJing process, all while honoring the craftsmanship and expertise a seasoned DJ brings to a live set. This tool is designed to automate the creation of unique, high-quality, and seamless mixes that were traditionally crafted manually.

## AI-Generated Cue Points

This part of the project delineates the creation of the AI model that generates cue points in tracks. The steps involved in this notebook are as follows:

1. **Importing Data from Rekordbox**: This step involves importing data from 24 tracks via Rekordbox, where cue point labels and timestamps are extracted for supervised training.
2. **Labels**: Here, timestamps are labeled, indicating whether they represent cue points, thus creating a supervised dataset for training the model.
3. **LSTM Model for AI Cue Point Generator**: This step entails implementing an LSTM Neural Network using PyTorch, which identifies patterns that signal cue points.

## AI DJ Mix Generator

This part encapsulates the functionalities and techniques embedded in the AI DJ Mix Generator, which includes the following elements:

1. **Preprocessing Step**: This step is about initializing Track class objects with essential attributes like beats, BPM, audio file, and sample rate.
2. **Mod and BPM Detection**: This process identifies the dominant BPM and tweaks other tracks to align with this tempo, ensuring uniform BPM across all tracks.
3. **Cue Point Detection**: This part utilizes the model and scaler from the AI Cue Point Generator to pinpoint significant cue points in each track.
4. **Randomness via Relevant Cue Point Selection & Randomness**: This aspect incorporates a semi-random strategy to choose cue points, creating a vibrant and unique mix.
5. **Frequency Equalization of the Songs (EQing)**: This process involves adjusting the frequency components of the songs, facilitating smoother transitions.
6. **Final Step: Generating a Mix**: This final step merges the adjusted audio files based on the determined cue points to produce a seamless mix.

## Techniques Employed 

This section elaborates on the techniques utilized in both the AI Cue Point Generator and the AI DJ Mix Generator, encompassing:

### Techniques for both AI Cue Point Generator and AI DJ Mix Generator

1. **Reading from WAV File**: This step initiates the preprocessing of tracks by reading audio files from the "wav" format.
2. **Beat Detection**: This process retrieves critical beat information using various libraries, with Madmom offering the most precise results.
3. **BPM Estimation**: This essential step estimates the songs' BPM, a critical component in aligning tracks during the mix process.
4. **Feature Extraction for Each Beat Timestamp**: This step extracts features for each beat timestamp using the Librosa library, encompassing aspects like MFCC, Spectral Centroid, Spectral Contrast, Spectral Rolloff, Spectral Flux, and RMS.

## Getting Started

Ready to dive in? Here's how you can kick start your journey with the AI DJ Mix Generator!

1. **Clone the Repository**: First off, clone the project repository onto your local machine.
   
2. **Setting Up Your Environment**: Before you start, ensure to establish a virtual environment to manage dependencies. You can do this with the command:


```python3 -m venv env```

3. **Installing Dependencies**: Navigate to the project directory and install the necessary dependencies using the `requirements.txt` file. Run the following command in your terminal:

```pip install -r requirements.txt```

4. **Adding Your Music**: Create a folder with all the tracks you want to mix and add it to the current working directory.

5. **Running the Script**: Now, it's time to generate your mix! Run the `main.py` script and provide the path to your music folder when prompted. Like so:

```python main.py```

"Enter the path to the folder of music: " -> < path to folder goes here >  

## License

The project is available under the MIT license. You can find the license details in the [MITLicense.txt](MITLicense.txt) file in the project repository.

## Contact

Feel free to get in touch at ruszkowskifranciszek@gmail.com.

## Demo Links

1. [Perpetual Groove EP by Alec Dienaar](https://soundcloud.com/franas_oven/demo1/s-IsV4HEA5Yz9?si=cba3a18e72ea4370af040972a7518700&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)

2. [Impulsive Behaviour Unreleased Tracks](https://soundcloud.com/franas_oven/demo2/s-1TFiSgjzuqh?si=7edcb49667814dc898a7243b49bf7cd7&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)
