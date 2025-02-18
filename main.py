import os
import sys
import warnings

from rich.console import Console
from rich.prompt import Prompt

import config
from audio import AudioProcessor
from utils import cleanup_temp_files, register_exit_handler
from visualization import Visualizer

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

def main():
    """Main entry point for the application."""
    console = Console()
    
    # Register cleanup handler
    register_exit_handler(cleanup_temp_files)
    
    # Print welcome message
    console.rule("[bold magenta]Shoegaze Hierarchical Fractal Language Architecture[/bold magenta]")
    console.print(
        "[green]Press Enter to use the default song or enter a song name or YouTube link.[/green]\n"
    )
    
    # Get song input
    song_input = Prompt.ask(
        "[bold cyan]Enter song name or YouTube link[/bold cyan]",
        default=config.DEFAULT_YOUTUBE_LINK
    )
    
    # Get resolution input
    resolution_input = Prompt.ask(
        "[bold cyan]Enter the resolution as width height (e.g., '1920 1080') or press Enter for default[/bold cyan]",
        default=""
    )
    
    # Parse resolution
    if resolution_input.strip():
        try:
            width, height = map(int, resolution_input.strip().split())
        except ValueError:
            console.print("[red]Invalid resolution input. Using default resolution 1920x1080.[/red]")
            width, height = 1920, 1080
    else:
        width, height = 1920, 1080
    
    # Initialize audio processor
    audio_processor = AudioProcessor()
    
    # Load audio
    if song_input.startswith('http'):
        if not audio_processor.load_from_youtube(song_input):
            console.print("[red]Failed to load audio from YouTube.[/red]")
            return
    else:
        youtube_url = f'ytsearch1:{song_input}'
        if not audio_processor.load_from_youtube(youtube_url):
            console.print("[red]Failed to load audio from YouTube search.[/red]")
            return
    
    # Initialize visualizer
    visualizer = Visualizer(width, height)
    
    # Print controls
    console.print("\n[yellow]Controls:[/yellow]")
    console.print("  [cyan]Space[/cyan]: Pause/Resume")
    console.print("  [cyan]R[/cyan]: Start/Stop Recording")
    console.print("  [cyan]Up/Down[/cyan]: Zoom In/Out")
    console.print("  [cyan]Esc[/cyan]: Exit")
    console.print("\n[green]Starting visualization...[/green]")
    
    # Run visualization
    try:
        visualizer.run(audio_processor)
    except Exception as e:
        console.print(f"[red]Error during visualization: {e}[/red]")
    finally:
        # Clean up
        audio_processor.cleanup()
        cleanup_temp_files()

if __name__ == "__main__":
    main()
