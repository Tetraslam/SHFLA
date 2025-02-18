import os
import shutil
import subprocess
import time
import warnings
from typing import Any, Dict, Optional, Tuple

import librosa
import numpy as np
import yt_dlp
from rich.console import Console

import config
from utils import normalize_audio, suppress_output

console = Console()

def find_ffmpeg() -> Optional[str]:
    """Find ffmpeg executable in system PATH."""
    try:
        # Try using where on Windows or which on Unix
        if os.name == 'nt':  # Windows
            result = subprocess.run(['where', 'ffmpeg'], capture_output=True, text=True)
        else:  # Unix-like
            result = subprocess.run(['which', 'ffmpeg'], capture_output=True, text=True)
        
        if result.returncode == 0:
            return result.stdout.strip().split('\n')[0]
    except Exception:
        pass

    # Check common installation paths
    common_paths = [
        r'C:\ffmpeg\bin\ffmpeg.exe',  # Windows common path
        '/usr/bin/ffmpeg',            # Linux common path
        '/usr/local/bin/ffmpeg',      # macOS common path
    ]
    
    for path in common_paths:
        if os.path.isfile(path):
            return path
    
    return None

class AudioProcessor:
    """Class to handle audio processing and feature extraction."""
    def __init__(self):
        self.y = None
        self.sr = None
        self.duration = 0
        self.chunk_samples = 0
        self.total_samples = 0
        self.total_chunks = 0
        
        # Find ffmpeg
        ffmpeg_path = find_ffmpeg()
        if ffmpeg_path:
            config.YDL_OPTS['ffmpeg_location'] = ffmpeg_path
        else:
            console.print("[yellow]Warning: ffmpeg not found in PATH. Audio processing may fail.[/yellow]")

    def load_from_youtube(self, url: str) -> bool:
        """Download and load audio from YouTube URL."""
        try:
            # Make sure temp directory exists
            os.makedirs(config.TEMP_DIR, exist_ok=True)

            # Clean up any existing files
            for file in os.listdir(config.TEMP_DIR):
                if file.startswith('downloaded_audio'):
                    try:
                        os.remove(os.path.join(config.TEMP_DIR, file))
                    except Exception as e:
                        console.print(f"[yellow]Warning: Could not remove existing file {file}: {e}[/yellow]")

            # Configure yt-dlp with progress hook
            ydl_opts = config.YDL_OPTS.copy()
            
            def progress_hook(d):
                if d['status'] == 'downloading':
                    console.print(f"[yellow]Downloading: {d.get('_percent_str', '0%')}[/yellow]", end='\r')
                elif d['status'] == 'finished':
                    console.print("[green]Download complete, processing audio...[/green]")
                elif d['status'] == 'error':
                    console.print(f"[red]Error during download: {d.get('error', 'Unknown error')}[/red]")
            
            ydl_opts['progress_hooks'] = [progress_hook]

            # Download the audio
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                console.print("[yellow]Starting download...[/yellow]")
                try:
                    # First, try to extract info to verify the URL is valid
                    info = ydl.extract_info(url, download=False)
                    if not info:
                        console.print("[red]Error: Could not get video information[/red]")
                        return False
                    
                    # If it's a search query, use the first result
                    if info.get('_type') == 'playlist':
                        if not info.get('entries'):
                            console.print("[red]Error: No results found for search query[/red]")
                            return False
                        info = info['entries'][0]
                    
                    # Now download the audio
                    ydl.download([info['webpage_url']])
                except Exception as e:
                    console.print(f"[red]Error during download: {str(e)}[/red]")
                    return False

            # Look for the downloaded file
            wav_files = [f for f in os.listdir(config.TEMP_DIR) if f.endswith('.wav')]
            if not wav_files:
                console.print("[red]Error: No WAV file found after download[/red]")
                return False

            # Use the most recently modified WAV file
            latest_wav = max(wav_files, key=lambda f: os.path.getmtime(os.path.join(config.TEMP_DIR, f)))
            downloaded_path = os.path.join(config.TEMP_DIR, latest_wav)

            # Rename to expected filename if necessary
            if downloaded_path != config.DOWNLOAD_PATH:
                try:
                    if os.path.exists(config.DOWNLOAD_PATH):
                        os.remove(config.DOWNLOAD_PATH)
                    os.rename(downloaded_path, config.DOWNLOAD_PATH)
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not rename audio file: {e}[/yellow]")
                    # Try to use the file as-is
                    return self.load_from_file(downloaded_path)

            # Wait for file to be fully written (max 10 seconds)
            max_wait = 10
            wait_time = 0
            last_size = -1
            while wait_time < max_wait:
                try:
                    current_size = os.path.getsize(config.DOWNLOAD_PATH)
                    if current_size > 0 and current_size == last_size:
                        break
                    last_size = current_size
                    time.sleep(0.5)
                    wait_time += 0.5
                    console.print(f"[yellow]Processing audio... {current_size} bytes[/yellow]", end='\r')
                except Exception:
                    time.sleep(0.5)
                    wait_time += 0.5

            console.print("\n[green]Audio processing complete[/green]")
            return self.load_from_file(config.DOWNLOAD_PATH)

        except Exception as e:
            console.print(f"[red]Error downloading audio: {str(e)}[/red]")
            return False

    def load_from_file(self, file_path: str) -> bool:
        """Load audio from local file."""
        try:
            if not os.path.exists(file_path):
                console.print(f"[red]Error: Audio file not found: {file_path}[/red]")
                return False

            if os.path.getsize(file_path) == 0:
                console.print("[red]Error: Audio file is empty[/red]")
                return False

            with suppress_output(), warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                self.y, self.sr = librosa.load(file_path, sr=config.SAMPLE_RATE, mono=config.MONO)
            
            if self.y is None or self.sr is None:
                console.print("[red]Error: Failed to load audio data[/red]")
                return False

            if self.y.ndim == 1:
                self.y = self.y[np.newaxis, :]
            
            # Normalize audio
            self.y = normalize_audio(self.y)
            
            # Calculate chunk parameters
            self.duration = librosa.get_duration(y=self.y, sr=self.sr)
            self.chunk_samples = int(config.CHUNK_DURATION * self.sr)
            self.total_samples = self.y.shape[1]
            self.total_chunks = int(np.ceil(self.total_samples / self.chunk_samples))
            
            console.print(f"[green]Successfully loaded audio: {self.duration:.1f} seconds[/green]")
            return True

        except Exception as e:
            console.print(f"[red]Error loading audio file: {e}[/red]")
            return False

    def get_chunk_features(self, chunk_idx: int) -> Optional[Dict[str, Any]]:
        """Extract audio features from a specific chunk."""
        try:
            current_sample = chunk_idx * self.chunk_samples
            if current_sample >= self.total_samples:
                return None

            # Get chunk data
            if current_sample + self.chunk_samples > self.total_samples:
                y_chunk = self.y[:, current_sample:]
            else:
                y_chunk = self.y[:, current_sample:current_sample + self.chunk_samples]

            if y_chunk.shape[1] == 0:
                return None

            # Convert to mono for feature extraction
            y_mono = np.mean(y_chunk, axis=0)

            # Initialize default values
            features = {
                'mean_f0': 440,
                'mean_centroid': config.MIN_CENTROID,
                'mean_chroma': np.zeros(12),
                'rms': 0.0,
                'onset_strength': 0.0
            }

            # Extract features if audio signal is strong enough
            if np.max(np.abs(y_mono)) >= 1e-3:
                # Pitch detection
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    f0 = librosa.yin(
                        y_mono,
                        fmin=librosa.note_to_hz("C2"),
                        fmax=librosa.note_to_hz("C7"),
                        sr=self.sr
                    )
                    valid_f0 = f0[f0 > 0]
                    if valid_f0.size > 0:
                        features['mean_f0'] = np.mean(valid_f0)

                # Spectral centroid
                spec_centroid = librosa.feature.spectral_centroid(y=y_mono, sr=self.sr)
                features['mean_centroid'] = max(np.mean(spec_centroid), config.MIN_CENTROID)

                # Chroma features
                chroma = librosa.feature.chroma_stft(y=y_mono, sr=self.sr)
                features['mean_chroma'] = np.mean(chroma, axis=1)

                # RMS energy
                features['rms'] = np.sqrt(np.mean(y_mono**2))

                # Onset strength
                onset_env = librosa.onset.onset_strength(y=y_mono, sr=self.sr)
                features['onset_strength'] = np.mean(onset_env)

            return features

        except Exception as e:
            console.print(f"[yellow]Warning: Error extracting features from chunk {chunk_idx}: {e}[/yellow]")
            return None

    def get_beat_times(self) -> np.ndarray:
        """Detect beat times in the audio."""
        try:
            y_mono = np.mean(self.y, axis=0)
            tempo, beat_frames = librosa.beat.beat_track(y=y_mono, sr=self.sr)
            return librosa.frames_to_time(beat_frames, sr=self.sr)
        except Exception as e:
            console.print(f"[yellow]Warning: Error detecting beats: {e}[/yellow]")
            return np.array([])

    @property
    def total_time(self) -> float:
        """Get total duration of audio in seconds."""
        return self.duration

    def cleanup(self) -> None:
        """Clean up resources."""
        self.y = None
        self.sr = None 