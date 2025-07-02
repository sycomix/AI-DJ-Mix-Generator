from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QListWidget, QFileDialog, QTextEdit, QGridLayout
from PyQt6.QtCore import Qt, QThread, QObject, pyqtSignal
import sys
import sounddevice as sd
import numpy as np

# Import the audio processing function
from audio_processor import process_audio_files

class Worker(QObject):
    finished = pyqtSignal(object) # Emit processed data
    progress = pyqtSignal(str)

    def __init__(self, file_paths):
        super().__init__()
        self.file_paths = file_paths

    def run(self):
        # Redirect stdout to capture print statements
        old_stdout = sys.stdout
        sys.stdout = self

        processed_data = None
        try:
            # Call the main audio processing function
            processed_data = process_audio_files(self.file_paths)
        finally:
            sys.stdout = old_stdout # Restore stdout

        self.finished.emit(processed_data)

    def write(self, text):
        self.progress.emit(text)

    def flush(self):
        pass

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI DJ DAW")
        self.setGeometry(100, 100, 1000, 700) # Increased size for better layout

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Status Label
        self.status_label = QLabel("Welcome to AI DJ DAW!")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.status_label)

        # Main content layout (Grid)
        content_layout = QGridLayout()
        main_layout.addLayout(content_layout)

        # --- Left Panel: Track Management ---
        track_management_layout = QVBoxLayout()
        content_layout.addLayout(track_management_layout, 0, 0, 2, 1) # Row 0, Col 0, Span 2 rows, 1 col

        self.load_button = QPushButton("Load Tracks")
        self.load_button.clicked.connect(self._load_tracks)
        track_management_layout.addWidget(self.load_button)

        self.track_list_widget = QListWidget()
        track_management_layout.addWidget(self.track_list_widget)

        # --- Right Panel: Playback Controls ---
        playback_controls_layout = QVBoxLayout() # Use QVBoxLayout for vertical arrangement of controls
        content_layout.addLayout(playback_controls_layout, 0, 1, 1, 1) # Row 0, Col 1, Span 1 row, 1 col

        playback_label = QLabel("Playback Controls")
        playback_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        playback_controls_layout.addWidget(playback_label)

        # Playback buttons
        button_row_layout = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self._play_audio)
        self.play_button.setEnabled(False) # Disable until audio is processed
        button_row_layout.addWidget(self.play_button)

        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self._pause_audio)
        self.pause_button.setEnabled(False)
        button_row_layout.addWidget(self.pause_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self._stop_audio)
        self.stop_button.setEnabled(False)
        button_row_layout.addWidget(self.stop_button)

        playback_controls_layout.addLayout(button_row_layout)

        # --- Bottom Panel: Log Output ---
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        content_layout.addWidget(self.log_output, 1, 1, 1, 1) # Row 1, Col 1, Span 1 row, 1 col

        self.thread = None
        self.worker = None
        self.processed_audio_data = None # To store the combined audio
        self.audio_stream = None
        self.current_frame = 0

    def _load_tracks(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        file_dialog.setNameFilter("Audio Files (*.wav *.aif *.mp3)")

        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            if not selected_files:
                self.status_label.setText("No tracks selected.")
                return

            self.track_list_widget.clear()
            for file_path in selected_files:
                self.track_list_widget.addItem(file_path)
            self.status_label.setText(f"Loaded {len(selected_files)} tracks. Processing...")
            self.log_output.clear()

            # Disable buttons during processing
            self.play_button.setEnabled(False)
            self.pause_button.setEnabled(False)
            self.stop_button.setEnabled(False)

            # Start the audio processing in a new thread
            self.thread = QThread()
            self.worker = Worker(selected_files)
            self.worker.moveToThread(self.thread)

            self.thread.started.connect(self.worker.run)
            self.worker.finished.connect(self._on_processing_finished)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.quit)
            self.thread.finished.connect(self.thread.deleteLater)
            self.worker.progress.connect(self.update_log)

            self.thread.start()

    def _on_processing_finished(self, processed_data):
        self.status_label.setText("Processing complete!")
        if processed_data and 'combined' in processed_data:
            self.processed_audio_data = processed_data['combined']
            self.play_button.setEnabled(True) # Enable play button
            self.stop_button.setEnabled(True) # Enable stop button
        else:
            self.status_label.setText("Processing failed or no audio generated.")

    def _play_audio(self):
        if self.processed_audio_data is None:
            self.status_label.setText("No processed audio to play.")
            return

        if self.audio_stream is None or not self.audio_stream.active:
            # Assuming a sample rate of 44100 from your audio_processor.py
            samplerate = 44100
            self.audio_stream = sd.OutputStream(samplerate=samplerate, channels=self.processed_audio_data.ndim, callback=self._audio_callback)
            self.audio_stream.start()
            self.status_label.setText("Playing audio...")
            self.play_button.setEnabled(False)
            self.pause_button.setEnabled(True)
            self.stop_button.setEnabled(True)
        elif self.audio_stream.stopped:
            self.audio_stream.start()
            self.status_label.setText("Resuming audio...")
            self.play_button.setEnabled(False)
            self.pause_button.setEnabled(True)
            self.stop_button.setEnabled(True)

    def _pause_audio(self):
        if self.audio_stream and self.audio_stream.active:
            self.audio_stream.stop()
            self.status_label.setText("Audio paused.")
            self.play_button.setEnabled(True)
            self.pause_button.setEnabled(False)
            self.stop_button.setEnabled(True)

    def _stop_audio(self):
        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
            self.audio_stream = None
            self.current_frame = 0 # Reset playback position
            self.status_label.setText("Audio stopped.")
            self.play_button.setEnabled(True)
            self.pause_button.setEnabled(False)
            self.stop_button.setEnabled(False)

    def _audio_callback(self, outdata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        chunksize = frames
        remaining_frames = len(self.processed_audio_data) - self.current_frame

        if remaining_frames == 0:
            # End of audio, stop stream
            raise sd.CallbackStop

        if remaining_frames < chunksize:
            chunksize = remaining_frames

        outdata[:chunksize] = self.processed_audio_data[self.current_frame:self.current_frame + chunksize]
        outdata[chunksize:] = 0 # Fill remaining buffer with zeros
        self.current_frame += chunksize

    def update_log(self, text):
        self.log_output.append(text.strip())

if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()