import numpy as np
import pygame
import librosa
import sys
import colorsys
import os
import yt_dlp
from rich import print
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.prompt import Prompt
from numba import cuda, njit, prange
import math
import warnings
import contextlib
from concurrent.futures import ThreadPoolExecutor

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module='librosa')
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message='Trying to estimate tuning from empty frequency set.'
)
warnings.filterwarnings("ignore", category=UserWarning, module='numba')
warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    from numba.core.errors import NumbaPerformanceWarning
    warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
except ImportError:
    warnings.filterwarnings(
        "ignore",
        message='Host array used in CUDA kernel will incur copy overhead'
    )

# ===========================
#        Modifiable Variables
# ===========================

# Default song settings
DEFAULT_SONG_NAME = 'Only Shallow by My Bloody Valentine'
DEFAULT_YOUTUBE_LINK = 'https://www.youtube.com/watch?v=FyYMzEplnfU'

# Visualization parameters
CHUNK_DURATION = 0.1          # Duration of each audio chunk in seconds
ALPHA = 0.1                   # Smoothing factor for transitions
MAX_ITER_DEFAULT = 3000       # Default maximum iterations for fractal computation
FPS = 60                      # Frames per second for the visualization
ZOOM = 1.0                    # Zoom level for the fractal

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
COLOR_SPEED = 0.001

# HSV color parameters
SATURATION = 1.0
VALUE = 1.0

# Spectral centroid minimum to avoid zero values
MIN_CENTROID = 10000

# Brightness adjustment
BRIGHTNESS_EXPONENT = 0.35    # Exponent for brightness scaling (adjust between 0.5 and 1.0)

# Number of threads for CPU processing
NUM_THREADS = 4               # Adjust based on your CPU's capabilities

# ===========================
#        Function Definitions
# ===========================

@cuda.jit
def julia_kernel(width, height, max_iter, zoom, c_real, c_imag, color, rotation_angle, image):
    x, y = cuda.grid(2)
    if x >= width or y >= height:
        return

    # Precompute constants
    zx_factor = 1.5 / (0.5 * zoom * width)
    zy_factor = 1.0 / (0.5 * zoom * height)
    cos_theta = math.cos(rotation_angle)
    sin_theta = math.sin(rotation_angle)

    zx = (x - width / 2) * zx_factor
    zy = (y - height / 2) * zy_factor

    # Apply rotation
    zx_rot = zx * cos_theta - zy * sin_theta
    zy_rot = zx * sin_theta + zy * cos_theta

    zx_temp = zx_rot
    zy_temp = zy_rot
    iteration = 0

    while (zx_temp * zx_temp + zy_temp * zy_temp < 4.0) and (iteration < max_iter):
        xtemp = zx_temp * zx_temp - zy_temp * zy_temp + c_real
        zy_temp = 2.0 * zx_temp * zy_temp + c_imag
        zx_temp = xtemp
        iteration += 1

    if iteration < max_iter:
        # Smooth coloring
        log_zn = math.log(zx_temp * zx_temp + zy_temp * zy_temp) / 2
        nu = math.log(log_zn / math.log(2)) / math.log(2)
        iteration = iteration + 1 - nu
        ratio = iteration / max_iter
        brightness = math.pow(ratio, BRIGHTNESS_EXPONENT)
        # Compute color
        col_r = color[0] * brightness
        col_g = color[1] * brightness
        col_b = color[2] * brightness
        # Clip values
        col_r = min(255, max(0, col_r))
        col_g = min(255, max(0, col_g))
        col_b = min(255, max(0, col_b))
        # Assign to image
        image[y, x, 0] = np.uint8(col_r)
        image[y, x, 1] = np.uint8(col_g)
        image[y, x, 2] = np.uint8(col_b)
    else:
        # Assign black color
        image[y, x, 0] = 0
        image[y, x, 1] = 0
        image[y, x, 2] = 0

@njit(parallel=True)
def julia_cpu_kernel(width, height, max_iter, zoom, c_real, c_imag, color, rotation_angle):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    zx_factor = 1.5 / (0.5 * zoom * width)
    zy_factor = 1.0 / (0.5 * zoom * height)
    cos_theta = math.cos(rotation_angle)
    sin_theta = math.sin(rotation_angle)

    for y in prange(height):
        for x in range(width):
            zx = (x - width / 2) * zx_factor
            zy = (y - height / 2) * zy_factor

            # Apply rotation
            zx_rot = zx * cos_theta - zy * sin_theta
            zy_rot = zx * sin_theta + zy * cos_theta

            zx_temp = zx_rot
            zy_temp = zy_rot
            iteration = 0

            while (zx_temp * zx_temp + zy_temp * zy_temp < 4.0) and (iteration < max_iter):
                xtemp = zx_temp * zx_temp - zy_temp * zy_temp + c_real
                zy_temp = 2.0 * zx_temp * zy_temp + c_imag
                zx_temp = xtemp
                iteration += 1

            if iteration < max_iter:
                # Smooth coloring
                log_zn = math.log(zx_temp * zx_temp + zy_temp * zy_temp) / 2
                nu = math.log(log_zn / math.log(2)) / math.log(2)
                iteration = iteration + 1 - nu
                ratio = iteration / max_iter
                brightness = math.pow(ratio, BRIGHTNESS_EXPONENT)
                # Compute color
                col_r = color[0] * brightness
                col_g = color[1] * brightness
                col_b = color[2] * brightness
                # Clip values
                col_r = min(255, max(0, col_r))
                col_g = min(255, max(0, col_g))
                col_b = min(255, max(0, col_b))
                # Assign to image
                image[y, x, 0] = np.uint8(col_r)
                image[y, x, 1] = np.uint8(col_g)
                image[y, x, 2] = np.uint8(col_b)
            else:
                # Assign black color
                image[y, x, 0] = 0
                image[y, x, 1] = 0
                image[y, x, 2] = 0
    return image

def generate_fractal(width, height, max_iter, zoom, c_real, c_imag, color, rotation_angle, use_cuda=True):
    if use_cuda:
        image_host = np.zeros((height, width, 3), dtype=np.uint8)
        image_device = cuda.to_device(image_host)

        threads_per_block = (16, 16)
        blocks_per_grid_x = (width + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_per_grid_y = (height + threads_per_block[1] - 1) // threads_per_block[1]
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        julia_kernel[blocks_per_grid, threads_per_block](
            width, height, max_iter, zoom, c_real, c_imag, color, rotation_angle, image_device
        )

        image_device.copy_to_host(image_host)
        return image_host
    else:
        # CPU-based fractal generation
        return julia_cpu_kernel(width, height, max_iter, zoom, c_real, c_imag, color, rotation_angle)

# ===========================
#             Main Function
# ===========================

def main():
    console = Console()
    with contextlib.redirect_stdout(open(os.devnull, 'w')), contextlib.redirect_stderr(open(os.devnull, 'w')):
        pygame.init()

    console.rule("[bold magenta]Shoegaze Hierarchical Fractal Language Architecture[/bold magenta]")
    console.print(
        "[green]Press Enter to use the default song or enter a song name or YouTube link.[/green]\n"
    )
    song_input = Prompt.ask(
        f"[bold cyan]Enter song name or YouTube link[/bold cyan]", default=DEFAULT_YOUTUBE_LINK
    )

    resolution_input = Prompt.ask(
        "[bold cyan]Enter the resolution as width height (e.g., '1920 1080') or press Enter for default[/bold cyan]",
        default=""
    )
    if resolution_input.strip():
        try:
            width, height = map(int, resolution_input.strip().split())
        except ValueError:
            console.print("[red]Invalid resolution input. Using default resolution 1920x1080.[/red]")
            width, height = 1920, 1080
    else:
        width, height = 1920, 1080

    with contextlib.redirect_stdout(open(os.devnull, 'w')), contextlib.redirect_stderr(open(os.devnull, 'w')):
        screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Shoegaze Hierarchical Fractal Language Architecture")
    clock = pygame.time.Clock()

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": "downloaded_audio.%(ext)s",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
                "nopostoverwrites": False,
            }
        ],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            console.print("[yellow]Downloading and processing audio...[/yellow]")
            if song_input.startswith('http'):
                youtube_url = song_input
            else:
                youtube_url = f'ytsearch1:{song_input}'
            ydl.download([youtube_url])
        except Exception as e:
            console.print(f"[red]Error downloading audio: {e}[/red]")
            sys.exit(1)

    audio_file = "downloaded_audio.wav"

    if not os.path.exists(audio_file):
        console.print("[red]Error: Audio file not found after download.[/red]")
        sys.exit(1)

    try:
        y, sr = librosa.load(audio_file, sr=None, mono=False)
    except Exception as e:
        console.print(f"[red]Error loading audio file: {e}[/red]")
        sys.exit(1)
    if y.ndim == 1:
        y = y[np.newaxis, :]

    with contextlib.redirect_stdout(open(os.devnull, 'w')), contextlib.redirect_stderr(open(os.devnull, 'w')):
        pygame.mixer.init(frequency=sr)
    try:
        pygame.mixer.music.load(audio_file)
    except Exception as e:
        console.print(f"[red]Error loading audio for playback: {e}[/red]")
        sys.exit(1)

    duration = librosa.get_duration(y=y, sr=sr)
    chunk_samples = int(CHUNK_DURATION * sr)
    total_samples = y.shape[1]
    total_chunks = int(np.ceil(total_samples / chunk_samples))

    console.print("[yellow]Pre-processing images for visualization...[/yellow]")
    images = [None] * total_chunks

    alpha = ALPHA
    smoothed_c_real = SMOOTHED_C_REAL_INIT
    smoothed_c_imag = SMOOTHED_C_IMAG_INIT
    smoothed_color_phase = SMOOTHED_COLOR_PHASE_INIT
    max_iter = MAX_ITER_DEFAULT
    rotation_angle = 0.0
    zoom = ZOOM

    # Detect if CUDA is available
    use_cuda = cuda.is_available()
    if use_cuda:
        console.print("[green]CUDA-compatible GPU detected. Using GPU acceleration.[/green]")
    else:
        console.print("[yellow]CUDA-compatible GPU not detected. Falling back to CPU computation with multithreading.[/yellow]")

    def process_chunk(chunk_idx):
        nonlocal smoothed_c_real, smoothed_c_imag, smoothed_color_phase, max_iter, rotation_angle

        current_sample = chunk_idx * chunk_samples
        if current_sample + chunk_samples > total_samples:
            y_chunk = y[:, current_sample:]
        else:
            y_chunk = y[:, current_sample:current_sample + chunk_samples]

        if y_chunk.shape[1] == 0:
            return

        y_mono = np.mean(y_chunk, axis=0)
        min_centroid = MIN_CENTROID

        if np.max(np.abs(y_mono)) < 1e-3:
            mean_f0 = 440
            mean_centroid = min_centroid
        else:
            try:
                f0 = librosa.yin(
                    y_mono,
                    fmin=librosa.note_to_hz("C2"),
                    fmax=librosa.note_to_hz("C7"),
                    sr=sr,
                )
                valid_f0 = f0[f0 > 0]
                mean_f0 = np.mean(valid_f0) if valid_f0.size > 0 else 440
            except Exception:
                mean_f0 = 440

            spec_centroid = librosa.feature.spectral_centroid(y=y_mono, sr=sr)
            mean_centroid = np.mean(spec_centroid)
            mean_centroid = max(mean_centroid, min_centroid)

        chroma = librosa.feature.chroma_stft(y=y_mono, sr=sr)
        mean_chroma = np.mean(chroma, axis=1)

        angle = ((mean_f0 - 65) / (1000 - 65)) * 2 * np.pi
        radius = 0.7885
        target_c_real = radius * np.cos(angle)
        target_c_imag = radius * np.sin(angle)

        target_c_real = np.clip(target_c_real, C_REAL_MIN, C_REAL_MAX)
        target_c_imag = np.clip(target_c_imag, C_IMAG_MIN, C_IMAG_MAX)

        smoothed_c_real = alpha * target_c_real + (1 - alpha) * smoothed_c_real
        smoothed_c_imag = alpha * target_c_imag + (1 - alpha) * smoothed_c_imag

        centroid_normalized = (mean_centroid - min_centroid) / (sr / 2 - min_centroid)
        centroid_normalized = np.clip(centroid_normalized, 0, 1)
        target_max_iter = int(500 + (math.exp(centroid_normalized) - 1) * 500)
        target_max_iter = np.clip(target_max_iter, 500, 1000)
        max_iter = int(alpha * target_max_iter + (1 - alpha) * max_iter)

        rotation_angle += ROTATION_SPEED
        rotation_angle %= 2 * np.pi

        color_phase = (smoothed_color_phase + COLOR_SPEED) % 1.0
        dominant_chroma = np.argmax(mean_chroma)
        target_hue = (dominant_chroma / 12.0 + color_phase) % 1.0
        smoothed_color_phase = alpha * target_hue + (1 - alpha) * smoothed_color_phase

        r, g, b = colorsys.hsv_to_rgb(smoothed_color_phase, SATURATION, VALUE)
        color = np.array([r * 255, g * 255, b * 255], dtype=np.float32)

        image = generate_fractal(
            width,
            height,
            max_iter,
            zoom,
            smoothed_c_real,
            smoothed_c_imag,
            color,
            rotation_angle,
            use_cuda
        )

        images[chunk_idx] = image

    if use_cuda:
        # GPU processing
        with Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Processing images with GPU...", total=total_chunks)
            for i in range(total_chunks):
                process_chunk(i)
                progress.advance(task)
    else:
        # CPU processing with multithreading
        console.print("[yellow]Processing images using CPU with multithreading...[/yellow]")
        with Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Processing images with CPU...", total=total_chunks)
            with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
                futures = []
                for i in range(total_chunks):
                    futures.append(executor.submit(process_chunk, i))
                for future in futures:
                    future.result()
                    progress.advance(task)

    console.print("[green]Pre-processing complete! Starting visualization...[/green]\n")

    pygame.mixer.music.play()
    running = True
    start_time = pygame.time.get_ticks()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        current_time = pygame.time.get_ticks() - start_time
        chunk_idx = int((current_time / 1000.0) / CHUNK_DURATION)

        if chunk_idx >= total_chunks:
            running = False
            continue

        image = images[chunk_idx]
        if image is not None:
            surface = pygame.surfarray.make_surface(np.transpose(image, (1, 0, 2)))
            screen.blit(surface, (0, 0))
            pygame.display.flip()

        clock.tick(FPS)

    pygame.quit()

    if os.path.exists(audio_file):
        os.remove(audio_file)

# ===========================
#              Entry Point
# ===========================

if __name__ == "__main__":
    main()
