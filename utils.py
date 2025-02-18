import contextlib
import os
import sys
from typing import Optional, Tuple, Union

import numpy as np
import psutil
from rich.console import Console

from config import TEMP_DIR

console = Console()

class MemoryMonitor:
    @staticmethod
    def get_memory_usage() -> float:
        """Returns current memory usage as a percentage."""
        process = psutil.Process(os.getpid())
        return process.memory_percent()

    @staticmethod
    def check_memory_usage(threshold: float = 80.0) -> bool:
        """
        Check if memory usage is above threshold.
        Returns True if memory usage is OK, False if too high.
        """
        usage = MemoryMonitor.get_memory_usage()
        if usage > threshold:
            console.print(f"[red]Warning: High memory usage ({usage:.1f}%)[/red]")
            return False
        return True

class CircularBuffer:
    """Circular buffer for storing frames with memory monitoring."""
    def __init__(self, size: int):
        self.size = size
        self.buffer = [None] * size
        self.head = 0
        self.tail = 0
        self.full = False

    def push(self, item: np.ndarray) -> None:
        """Add item to buffer."""
        self.buffer[self.head] = item
        self.head = (self.head + 1) % self.size
        if self.head == self.tail:
            self.full = True
            self.tail = (self.tail + 1) % self.size

    def get(self, index: int) -> Optional[np.ndarray]:
        """Get item at relative index from tail."""
        if index >= self.size:
            return None
        if not self.full and index >= self.head:
            return None
        pos = (self.tail + index) % self.size
        return self.buffer[pos]

    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer = [None] * self.size
        self.head = 0
        self.tail = 0
        self.full = False

def cleanup_temp_files() -> None:
    """Clean up temporary files and directories."""
    try:
        for file in os.listdir(TEMP_DIR):
            file_path = os.path.join(TEMP_DIR, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                console.print(f"[yellow]Error deleting {file_path}: {e}[/yellow]")
    except Exception as e:
        console.print(f"[yellow]Error cleaning temp directory: {e}[/yellow]")

def suppress_output() -> contextlib.ExitStack:
    """Context manager to suppress stdout and stderr."""
    stack = contextlib.ExitStack()
    stack.enter_context(contextlib.redirect_stdout(open(os.devnull, 'w')))
    stack.enter_context(contextlib.redirect_stderr(open(os.devnull, 'w')))
    return stack

def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Normalize audio to [-1, 1] range."""
    if audio.size == 0:
        return audio
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        return audio / max_val
    return audio

def format_time(seconds: float) -> str:
    """Format time in seconds to MM:SS format."""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min_val and max_val."""
    return max(min_val, min(max_val, value))

def register_exit_handler(cleanup_func: callable) -> None:
    """Register cleanup function for graceful exit."""
    import atexit
    import signal
    
    def signal_handler(signum, frame):
        cleanup_func()
        sys.exit(0)
    
    atexit.register(cleanup_func)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler) 