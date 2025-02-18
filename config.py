import atexit
import os
import shutil

import pygame

# Default song settings
DEFAULT_SONG_NAME = 'Only Shallow by My Bloody Valentine'
DEFAULT_YOUTUBE_LINK = 'https://www.youtube.com/watch?v=FyYMzEplnfU'

# Visualization parameters
CHUNK_DURATION = 0.05          # Duration of each audio chunk in seconds (20Hz update rate)
ALPHA = 0.9                   # Smoothing factor for transitions
MAX_ITER_DEFAULT = 1000       # Reduced maximum iterations for better performance
FPS = 60                      # Reduced target FPS for better stability
ZOOM = 0.8                    # Zoom level for the fractal

# Initial smoothing variables
SMOOTHED_C_REAL_INIT = -0.4   # Initial real part of complex parameter c
SMOOTHED_C_IMAG_INIT = 0.6    # Initial imaginary part of complex parameter c
SMOOTHED_COLOR_PHASE_INIT = 0.0

# Fractal parameter constraints
C_REAL_MIN = -0.7
C_REAL_MAX = 0.7
C_IMAG_MIN = -0.7
C_IMAG_MAX = 0.7

# Rotation and color speeds
ROTATION_SPEED = 0.0005
COLOR_SPEED = 0.0001      # Reduced from 0.001 to 0.0001 for much slower color changes

# HSV color parameters
SATURATION = 0.8          # Slightly reduced from 1.0 for less intense colors
VALUE = 1.0

# Spectral centroid minimum to avoid zero values
MIN_CENTROID = 10000

# Brightness adjustment
BRIGHTNESS_EXPONENT = 0.25    # Exponent for brightness scaling

# Number of threads for CPU processing
NUM_THREADS = 4               # Adjust based on CPU capabilities

# Audio settings
SAMPLE_RATE = None  # Use native sample rate
MONO = True

# Window buffer settings
BUFFER_SIZE = 100  # Number of frames to keep in memory

# Video export settings
VIDEO_EXPORT_FPS = 30
VIDEO_EXPORT_QUALITY = 8  # 0-10, higher is better quality

# Keyboard control settings
SKIP_SECONDS = 5
ZOOM_STEP = 0.1
MENU_KEY = pygame.K_m  # Key to toggle menu
LEFT_KEY = pygame.K_LEFT  # Skip backward
RIGHT_KEY = pygame.K_RIGHT  # Skip forward
SPACE_KEY = pygame.K_SPACE  # Pause/Resume
ESC_KEY = pygame.K_ESCAPE  # Exit
UP_KEY = pygame.K_UP  # Zoom in
DOWN_KEY = pygame.K_DOWN  # Zoom out
R_KEY = pygame.K_r  # Record

# Menu settings
MENU_WIDTH = 300
MENU_PADDING = 10
MENU_ITEM_HEIGHT = 30
MENU_BACKGROUND_COLOR = (0, 0, 0, 200)  # RGBA
MENU_TEXT_COLOR = (255, 255, 255)
MENU_HIGHLIGHT_COLOR = (100, 100, 255, 200)
MENU_FONT_SIZE = 20

# Parameter adjustment steps
BRIGHTNESS_STEP = 0.05
ROTATION_SPEED_STEP = 0.0001
COLOR_SPEED_STEP = 0.0001
ALPHA_STEP = 0.05

# Parameter limits
BRIGHTNESS_RANGE = (0.05, 1.0)
ROTATION_SPEED_RANGE = (0.0, 0.002)
COLOR_SPEED_RANGE = (0.0, 0.005)
ALPHA_RANGE = (0.05, 1.0)

# File paths
def setup_directories():
    """Set up necessary directories and return paths."""
    base_dir = os.getcwd()
    temp_dir = os.path.join(base_dir, 'temp')
    video_dir = os.path.join(base_dir, 'exports')
    
    # Create directories if they don't exist
    for directory in [temp_dir, video_dir]:
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except Exception as e:
            print(f"Error creating directory {directory}: {e}")
            raise
    
    return temp_dir, video_dir

# Set up directories
TEMP_DIR, VIDEO_OUTPUT_DIR = setup_directories()
DOWNLOAD_PATH = os.path.join(TEMP_DIR, 'downloaded_audio.wav')

# Clean up function
def cleanup_directories():
    """Clean up temporary directories on exit."""
    try:
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
    except Exception as e:
        print(f"Error cleaning up temp directory: {e}")

# Register cleanup
atexit.register(cleanup_directories)

# YouTube download options
YDL_OPTS = {
    "format": "bestaudio/best",
    "outtmpl": os.path.splitext(DOWNLOAD_PATH)[0],  # Remove extension from output template
    "noplaylist": True,
    "quiet": False,
    "no_warnings": False,
    "extract_audio": True,
    "postprocessors": [{
        "key": "FFmpegExtractAudio",
        "preferredcodec": "wav",
        "preferredquality": "192",
        "nopostoverwrites": True,  # Changed to True to ensure overwrite
    }],
    "ffmpeg_location": None,
    "verbose": True,
} 