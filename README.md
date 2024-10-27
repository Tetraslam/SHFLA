# SHFLA (Shoegaze Hierarchical Fractal Language Architecture)

**Authors:** Shresht Bhowmick, Arnav Dave  
**Date:** October 2024

---

## Table of Contents

- [SHFLA (Shoegaze Hierarchical Fractal Language Architecture)](#shfla-shoegaze-hierarchical-fractal-language-architecture)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Features](#features)
  - [How It Works](#how-it-works)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Required Python Packages](#required-python-packages)
    - [Steps](#steps)
  - [Usage](#usage)
  - [Requirements](#requirements)
  - [Examples](#examples)
    - [Visualization Screenshots](#visualization-screenshots)
      - [Brightness Mapping](#brightness-mapping)
      - [Contrast Mapping](#contrast-mapping)
      - [Edge Smoothness and Complexity](#edge-smoothness-and-complexity)
  - [Contributing](#contributing)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)
  - [Code Overview](#code-overview)
  - [Contact](#contact)

---

## Introduction

**SHFLA (Shoegaze Hierarchical Fractal Language Architecture)** is an interdisciplinary project that integrates Cognitive Musicology, Linguistics, Music Theory, and Computer Science. The core of this project is a dynamic system actualized as a fractal that continually adapts to a musical excerpt, piece, or song provided by the user. The program interprets the music through sequences of changing visual imagery, specifically generating Julia set fractals in real-time synchronized with the audio.

This project explores unconventional computing paradigms by mapping musical features to fractal parameters, creating a unique visual and auditory experience that also demonstrates Turing completeness using sound-based computation. You can read more in our writeup [here](./SHFLA__Shoegaze_Hierarchical_Fractal_Language_Architecture_.pdf).

---

## Features

- **Real-Time Music Visualization:** Generates dynamic Julia set fractals synchronized with any song input by the user.
- **Feature Mapping:**
  - **Brightness:** Corresponds to the spectral centroid (perceived brightness) of the music.
  - **Contrast:** Linked to the complexity of the Fourier transform of the audio chunk.
  - **Color:** Maps the musical key (pitch) to the RGB color palette.
  - **Edge Smoothness and Complexity:** Represents consonance and dissonance in the music.
  - **Sphericality:** Relates to the resonance and spectral characteristics of the audio.
  - **Asymmetry:** Reflects the panning (left-right balance) of the music.
- **Interactive Experience:** Users can input any song name, and the program fetches the audio and album art automatically.
- **Turing Completeness Exploration:** Demonstrates the potential for Turing-complete computation using audio input and fractal generation.

---

## How It Works

1. **Audio Input:**
   - The user inputs the name of a song.
   - The program downloads the audio using YouTube as a source.
   - Loads the audio for both processing and playback.

2. **Feature Extraction:**
   - **Pitch (Mean Fundamental Frequency):** Determines the complex parameter `c` for the Julia set.
   - **Spectral Centroid:** Influences the zoom factor and maximum iterations in the fractal generation.
   - **Chroma (Pitch Class Profile):** Maps to the hue in the HSV color space for coloring the fractal.
   - **Panning Information:** Used for asymmetry.

3. **Fractal Generation:**
   - Generates a Julia set fractal for each chunk of audio data.
   - Parameters like `c`, zoom, rotation, and color are updated in real-time based on the extracted features.
   - Utilizes Numba's JIT compilation for performance optimization in fractal computation.

4. **Visualization:**
   - Displays the fractal images using Pygame.
   - Synchronizes the visual changes with the music playback.

---

## Installation

### Prerequisites

- **Python 3.9 or higher**
- **pip** package manager

### Required Python Packages

- `numpy`
- `pygame`
- `librosa`
- `numba`
- `pillow`
- `yt-dlp`
- `requests`

### Steps

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Tetraslam/SHFLA.git
   cd SHFLA
   ```

2. **Install the Required Packages:**

   ```bash
   pip install -r requirements.txt
   ```

   Alternatively, you can install the packages manually:

   ```bash
   pip install numpy pygame librosa numba pillow yt-dlp requests
   ```

3. **Ensure FFMPEG is Installed:**

   The program requires `ffmpeg` for audio processing.

   - **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html#build-windows) and add to your PATH.
   - **macOS:** Install via Homebrew:

     ```bash
     brew install ffmpeg
     ```

   - **Linux:** Install via package manager:

     ```bash
     sudo apt-get install ffmpeg
     ```

---

## Usage

1. **Run the Program:**

   ```bash
   python main.py
   ```

2. **Input Song Name:**

   When prompted, enter the name of the song you wish to visualize.

   ```
   Enter the name of the song to search for: hades in the dead of winter by my dead girlfriend
   ```

3. **Set Resolution (Optional):**

   You can specify the window resolution or press Enter to use the default (1920x1080).

   ```
   Enter the resolution as width height (e.g., '1920 1080' without quotes) or press Enter for default:
   ```

4. **Enjoy the Visualization:**

   The program will download the audio, process it, and display the dynamic Julia set fractal synchronized with the music.

---

## Requirements

- **Operating System:** Windows, macOS, or Linux
- **Python Version:** 3.9 or higher
- **Internet Connection:** Required for downloading audio and album art
- **Hardware:** A machine capable of running real-time audio and graphics processing

---

## Examples

### Visualization Screenshots

Here are some examples of the fractal visualizations generated by SHFLA:

#### Brightness Mapping

![Brightness Mapping](images/brightness.png)

*Figure 1: Fractal visualization showing brightness corresponding to the spectral centroid.*

#### Contrast Mapping

![Contrast Mapping](images/fourier.png)

*Figure 2: Fractal visualization showing contrast related to Fourier complexity.*

#### Edge Smoothness and Complexity

![Edge Smoothness](images/edge_smoothness.png)

*Figure 3: Fractal edges representing consonance and dissonance.*

---

## Contributing

We welcome contributions from the community! If you'd like to contribute to SHFLA, please follow these steps:

1. **Fork the Repository**
2. **Create a Feature Branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Commit Your Changes**

   ```bash
   git commit -am 'Add a new feature'
   ```

4. **Push to the Branch**

   ```bash
   git push origin feature/your-feature-name
   ```

5. **Open a Pull Request**

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Cognitive Musicology and Music Theory:** For inspiring the integration of musical features into computational models.
- **Fractal Geometry:** Benoît Mandelbrot's work on fractals laid the foundation for this project.
- **Unconventional Computing Paradigms:** Exploring new ways to represent computation through audio and visual mediums.
- **Python Community:** For the development of libraries like NumPy, Librosa, Pygame, and Numba, which made this project possible.

---

**References:**

- Adamatzky, A. (Ed.). (2016). *Advances in Unconventional Computing: Volume 1: Theory*. Springer.
- Devaney, R. L. (1992). *A First Course in Chaotic Dynamical Systems: Theory and Experiment*. Westview Press.
- Hsu, K. J., & Hsu, A. (1990). Fractal geometry of music. *Proceedings of the National Academy of Sciences*, 87(3), 938-941.
- Leman, M. (1995). *Music and Schema Theory: Cognitive Foundations of Systematic Musicology*. Springer.
- MacLennan, B. J. (2003). Transcending Turing computability. *Minds and Machines*, 13(1), 3-22.
- Mandelbrot, B. B. (1983). *The Fractal Geometry of Nature*. W. H. Freeman.
- Purwins, H., Herrera, P., Grachten, M., Hazan, A., Marxer, R., & Serra, X. (2008). Computational models of music perception and cognition I: The perceptual and cognitive processing chain. *Physics of Life Reviews*, 5(3), 151-168.
- Voss, R. F., & Clarke, J. (1975). "1/f noise" in music and speech. *Nature*, 258(5533), 317-318.

---

## Code Overview

Below is a brief overview of the main components of the code:

- **Imports Necessary Libraries:** Including `numpy`, `pygame`, `librosa`, `numba`, and others.
- **Main Functionality:**
  - **User Input:** Prompts for the song name and desired resolution.
  - **Album Art Retrieval:** Fetches album cover art using the iTunes Search API.
  - **Audio Download and Processing:** Downloads the audio using `yt-dlp` and processes it with `librosa`.
  - **Audio Playback:** Uses `pygame.mixer` to play the audio.
  - **Feature Extraction:** Extracts features like pitch, spectral centroid, and chroma.
  - **Parameter Mapping:** Maps extracted features to fractal parameters such as `c`, zoom, rotation, and color.
  - **Fractal Generation:** Generates the Julia set fractal using a Numba-optimized function.
  - **Visualization Loop:** Continuously updates and displays the fractal in sync with the music.

- **Julia Set Function (`julia_set`):**
  - Uses Numba's `@njit` decorator for just-in-time compilation.
  - Computes the fractal for each pixel, applying smooth coloring techniques.
  - Incorporates rotation and zoom transformations.

---

## Contact

- **Shresht Bhowmick:** [Email](mailto:bhowmick.sh@northeastern.edu) | [GitHub](https://github.com/Tetraslam)
- **Arnav Dave:**

---

*Through this project, we aim to propose a new computing paradigm that encompasses multiple senses as part of the computational representation of input and output. We hope you find this exploration as exciting as we do!*