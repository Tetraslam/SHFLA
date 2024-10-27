import numpy as np
import pygame
import librosa
from numba import njit, prange
import sys
import colorsys
import os
import yt_dlp
import requests
from PIL import Image
from io import BytesIO

def main():
    # Initialize Pygame
    pygame.init()

    # Get song name from user input
    song_name = input("Enter the name of the song to search for: ")

    # Set resolution
    resolution_input = input("Enter the resolution as width height (e.g., '1920 1080') or press Enter for default: ")
    if resolution_input.strip():
        try:
            width, height = map(int, resolution_input.strip().split())
        except ValueError:
            print("Invalid resolution input. Using default resolution 1920x1080.")
            width, height = 1920, 1080
    else:
        width, height = 1920, 1080  # Default to Full HD

    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Dynamic Julia Set Music Visualizer")
    clock = pygame.time.Clock()

    # Fetch album cover art using iTunes Search API
    itunes_api_url = "https://itunes.apple.com/search"
    params = {
        'term': song_name,
        'media': 'music',
        'limit': 1,
    }
    response = requests.get(itunes_api_url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data['resultCount'] > 0:
            artwork_url = data['results'][0]['artworkUrl100']
            # Get higher resolution image
            artwork_url = artwork_url.replace('100x100bb', '1000x1000bb')
            # Download the image
            image_response = requests.get(artwork_url)
            if image_response.status_code == 200:
                image_data = image_response.content
                album_image = Image.open(BytesIO(image_data))
                # Display the image
                #album_image.show()
            else:
                print("Error downloading album artwork.")
        else:
            print("No results found for the song.")
    else:
        print("Error fetching data from iTunes API.")

    # Download and convert audio
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'downloaded_audio.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            print("Downloading and processing audio...")
            ydl.download([f'ytsearch1:"{song_name}"'])
        except Exception as e:
            print(f"Error downloading audio: {e}")
            sys.exit(1)

    # Audio file
    audio_file = 'downloaded_audio.wav'

    # Check if the audio file exists
    if not os.path.exists(audio_file):
        print("Error: Audio file not found after download.")
        sys.exit(1)

    # Load audio file
    try:
        y, sr = librosa.load(audio_file, sr=None, mono=False)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        sys.exit(1)
    if y.ndim == 1:
        y = y[np.newaxis, :]  # Convert to 2D array with one channel

    # Load audio for playback
    pygame.mixer.init(frequency=sr)
    try:
        pygame.mixer.music.load(audio_file)
    except Exception as e:
        print(f"Error loading audio for playback: {e}")
        sys.exit(1)

    # Start playing the audio
    pygame.mixer.music.play()

    # Audio parameters
    duration = librosa.get_duration(y=y, sr=sr)
    chunk_duration = 0.1  # seconds
    chunk_samples = int(chunk_duration * sr)
    total_samples = y.shape[1]
    current_sample = 0

    # Fractal parameters
    zoom = 1.0
    zoom_factor = 1.01
    max_iter = 256

    # Smooth parameter transitions
    alpha = 0.05  # Smoothing factor for gradual transitions
    smoothed_c_real = -0.8
    smoothed_c_imag = 0.156
    smoothed_color_phase = 0.0

    # Rotation parameters
    rotation_angle = 0.0
    rotation_speed = 0.001  # Adjust for desired rotation speed

    # Previous c values for transitions
    target_c_real = smoothed_c_real
    target_c_imag = smoothed_c_imag

    # For smooth color transitions
    color_phase = 0.0
    color_speed = 0.002

    running = True

    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Calculate current position in audio
        current_pos = pygame.mixer.music.get_pos()  # in milliseconds
        if current_pos == -1:
            # Playback has finished
            running = False
            continue
        current_sample = int((current_pos / 1000.0) * sr)
        if current_sample + chunk_samples > total_samples:
            y_chunk = y[:, current_sample:]
        else:
            y_chunk = y[:, current_sample:current_sample + chunk_samples]

        if y_chunk.shape[1] == 0:
            continue  # Skip empty chunks

        y_mono = np.mean(y_chunk, axis=0)

        # Extract features
        # Check for silence
        if np.max(np.abs(y_mono)) < 1e-3:
            mean_f0 = mean_f0 if 'mean_f0' in locals() else 440  # Default to A4
        else:
            # Pitch
            try:
                f0 = librosa.yin(y_mono, fmin=librosa.note_to_hz('C2'),
                                 fmax=librosa.note_to_hz('C7'), sr=sr)
                valid_f0 = f0[f0 > 0]
                if valid_f0.size > 0:
                    mean_f0 = np.mean(valid_f0)
                else:
                    mean_f0 = mean_f0 if 'mean_f0' in locals() else 440  # Default to A4
            except Exception as e:
                mean_f0 = mean_f0 if 'mean_f0' in locals() else 440  # Default to A4

        # Spectral centroid (brightness)
        spec_centroid = librosa.feature.spectral_centroid(y=y_mono, sr=sr)
        mean_centroid = np.mean(spec_centroid)

        # Chroma (key/pitch)
        chroma = librosa.feature.chroma_stft(y=y_mono, sr=sr)
        mean_chroma = np.mean(chroma, axis=1)

        # Map features to Julia set parameters

        # Complex parameter c for Julia set
        angle = ((mean_f0 - 65) / (1000 - 65)) * 2 * np.pi
        radius = 0.7885
        new_c_real = radius * np.cos(angle)
        new_c_imag = radius * np.sin(angle)

        # Update target c values
        target_c_real = new_c_real
        target_c_imag = new_c_imag

        # Apply smoothing to c parameters for transitions
        smoothed_c_real = alpha * target_c_real + (1 - alpha) * smoothed_c_real
        smoothed_c_imag = alpha * target_c_imag + (1 - alpha) * smoothed_c_imag

        # Adjust max_iter based on spectral centroid
        target_max_iter = int(128 + (mean_centroid / (sr / 2)) * 512)
        target_max_iter = np.clip(target_max_iter, 128, 1024)
        # Smooth max_iter
        max_iter = int(alpha * target_max_iter + (1 - alpha) * max_iter)

        # Adjust zoom based on spectral centroid
        target_zoom_factor = 1.0 + (mean_centroid / (sr / 2)) * 0.01
        zoom_factor = alpha * target_zoom_factor + (1 - alpha) * zoom_factor
        zoom *= zoom_factor

        # Adjust rotation angle
        rotation_angle += rotation_speed
        rotation_angle %= 2 * np.pi

        # Adjust color phase for smooth color transitions
        color_phase = (color_phase + color_speed) % 1.0

        # Map chroma to hue
        dominant_chroma = np.argmax(mean_chroma)
        target_hue = (dominant_chroma / 12.0 + color_phase) % 1.0

        # Smooth hue
        smoothed_color_phase = alpha * target_hue + (1 - alpha) * smoothed_color_phase

        saturation = 1.0
        value = 1.0

        r, g, b = colorsys.hsv_to_rgb(smoothed_color_phase, saturation, value)
        color = np.array([r * 255, g * 255, b * 255], dtype=np.float32)

        # Generate Julia set image
        image = julia_set(width, height, max_iter, zoom, smoothed_c_real, smoothed_c_imag, color, rotation_angle)

        # Display image
        surface = pygame.surfarray.make_surface(np.transpose(image, (1, 0, 2)))
        screen.blit(surface, (0, 0))
        pygame.display.flip()

        # Control frame rate
        clock.tick(30)

    pygame.quit()

    # Clean up the downloaded audio file
    if os.path.exists(audio_file):
        os.remove(audio_file)

@njit(parallel=True)
def julia_set(width, height, max_iter, zoom, c_real, c_imag, color, rotation_angle):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    zx_factor = np.float32(1.5 / (0.5 * zoom * width))
    zy_factor = np.float32(1.0 / (0.5 * zoom * height))
    c_real = np.float32(c_real)
    c_imag = np.float32(c_imag)
    cos_theta = np.cos(rotation_angle)
    sin_theta = np.sin(rotation_angle)
    for y in prange(height):
        zy = np.float32((y - height / 2) * zy_factor)
        for x in range(width):
            zx = np.float32((x - width / 2) * zx_factor)
            # Apply rotation
            zx_rot = zx * cos_theta - zy * sin_theta
            zy_rot = zx * sin_theta + zy * cos_theta
            iteration = 0
            zx_temp = zx_rot
            zy_temp = zy_rot
            while (zx_temp * zx_temp + zy_temp * zy_temp < 4.0) and (iteration < max_iter):
                xtemp = zx_temp * zx_temp - zy_temp * zy_temp + c_real
                zy_temp = 2.0 * zx_temp * zy_temp + c_imag
                zx_temp = xtemp
                iteration += 1
            if iteration < max_iter:
                # Smooth coloring
                log_zn = np.log(zx_temp * zx_temp + zy_temp * zy_temp) / 2
                nu = np.log(log_zn / np.log(2)) / np.log(2)
                iteration = iteration + 1 - nu
                ratio = iteration / max_iter
                brightness = np.sqrt(ratio)
                # Compute color
                col_r = color[0] * brightness
                col_g = color[1] * brightness
                col_b = color[2] * brightness
                # Clip values manually
                col_r = min(255, max(0, col_r))
                col_g = min(255, max(0, col_g))
                col_b = min(255, max(0, col_b))
                # Assign to image
                image[y, x, 0] = np.uint8(col_r)
                image[y, x, 1] = np.uint8(col_g)
                image[y, x, 2] = np.uint8(col_b)
            else:
                # Assign black color
                image[y, x, 0] = np.uint8(0)
                image[y, x, 1] = np.uint8(0)
                image[y, x, 2] = np.uint8(0)
    return image

if __name__ == "__main__":
    main()
