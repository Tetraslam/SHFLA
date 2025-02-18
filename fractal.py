import math
from typing import Optional, Tuple

import numpy as np
from numba import cuda, njit, prange

import config
from utils import MemoryMonitor


@cuda.jit(fastmath=True)
def julia_kernel(width: int, height: int, max_iter: int, zoom: float,
                c_real: float, c_imag: float, color: np.ndarray,
                rotation_angle: float, brightness_exp: float, image: np.ndarray) -> None:
    """Optimized CUDA kernel for Julia set computation."""
    x, y = cuda.grid(2)
    if x >= width or y >= height:
        return

    # Precompute constants
    zx_factor = 1.5 / (0.5 * zoom * width)
    zy_factor = 1.0 / (0.5 * zoom * height)
    cos_theta = math.cos(rotation_angle)
    sin_theta = math.sin(rotation_angle)

    # Calculate initial position
    zx = (x - width / 2) * zx_factor
    zy = (y - height / 2) * zy_factor

    # Apply rotation
    zx_rot = zx * cos_theta - zy * sin_theta
    zy_rot = zx * sin_theta + zy * cos_theta

    zx_temp = zx_rot
    zy_temp = zy_rot
    iteration = 0

    # Main iteration loop
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
        brightness = math.pow(ratio, brightness_exp)
        
        # Compute and clip color values
        col_r = min(255, max(0, color[0] * brightness))
        col_g = min(255, max(0, color[1] * brightness))
        col_b = min(255, max(0, color[2] * brightness))
        
        # Assign to image
        image[y, x, 0] = np.uint8(col_r)
        image[y, x, 1] = np.uint8(col_g)
        image[y, x, 2] = np.uint8(col_b)
    else:
        # Black color for points outside set
        image[y, x, 0] = 0
        image[y, x, 1] = 0
        image[y, x, 2] = 0

@njit(parallel=True)
def julia_cpu_kernel(width: int, height: int, max_iter: int, zoom: float,
                    c_real: float, c_imag: float, color: np.ndarray,
                    rotation_angle: float, brightness_exp: float) -> np.ndarray:
    """Optimized CPU kernel for Julia set computation using parallel processing."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Precompute constants
    zx_factor = 1.5 / (0.5 * zoom * width)
    zy_factor = 1.0 / (0.5 * zoom * height)
    cos_theta = math.cos(rotation_angle)
    sin_theta = math.sin(rotation_angle)

    for y in prange(height):
        for x in range(width):
            # Calculate initial position
            zx = (x - width / 2) * zx_factor
            zy = (y - height / 2) * zy_factor

            # Apply rotation
            zx_rot = zx * cos_theta - zy * sin_theta
            zy_rot = zx * sin_theta + zy * cos_theta

            zx_temp = zx_rot
            zy_temp = zy_rot
            iteration = 0

            # Main iteration loop
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
                brightness = math.pow(ratio, brightness_exp)
                
                # Compute and clip color values
                col_r = min(255, max(0, color[0] * brightness))
                col_g = min(255, max(0, color[1] * brightness))
                col_b = min(255, max(0, color[2] * brightness))
                
                # Assign to image
                image[y, x, 0] = np.uint8(col_r)
                image[y, x, 1] = np.uint8(col_g)
                image[y, x, 2] = np.uint8(col_b)
            else:
                # Black color for points outside set
                image[y, x, 0] = 0
                image[y, x, 1] = 0
                image[y, x, 2] = 0

    return image

class FractalGenerator:
    """Class to manage fractal generation with GPU/CPU support."""
    def __init__(self, width: int, height: int, use_cuda: bool = True):
        self.width = width
        self.height = height
        self.use_cuda = use_cuda and cuda.is_available()
        
        if self.use_cuda:
            # Pre-allocate CUDA memory
            self.image_device = cuda.device_array((height, width, 3), dtype=np.uint8)
            
            # Calculate optimal grid and block dimensions
            # Use larger thread blocks for better occupancy
            self.threads_per_block = (32, 32)
            self.blocks_per_grid = (
                (width + self.threads_per_block[0] - 1) // self.threads_per_block[0],
                (height + self.threads_per_block[1] - 1) // self.threads_per_block[1]
            )
            
            # Pre-compile kernel for faster launches
            dummy_color = np.array([0, 0, 0], dtype=np.float32)
            julia_kernel[self.blocks_per_grid, self.threads_per_block](
                self.width, self.height, 100, 1.0,
                0.0, 0.0, dummy_color, 0.0,
                0.15, self.image_device
            )
            cuda.synchronize()

    def generate(self, max_iter: int, zoom: float, c_real: float, c_imag: float,
                color: np.ndarray, rotation_angle: float) -> Optional[np.ndarray]:
        """Generate fractal image using either GPU or CPU."""
        if not MemoryMonitor.check_memory_usage():
            return None

        if self.use_cuda:
            # Launch kernel asynchronously
            julia_kernel[self.blocks_per_grid, self.threads_per_block](
                self.width, self.height, max_iter, zoom,
                c_real, c_imag, color, rotation_angle, 
                config.BRIGHTNESS_EXPONENT,
                self.image_device
            )
            # Only synchronize if we need the result immediately
            cuda.synchronize()
            return self.image_device.copy_to_host()
        else:
            return julia_cpu_kernel(
                self.width, self.height, max_iter, zoom,
                c_real, c_imag, color, rotation_angle,
                config.BRIGHTNESS_EXPONENT
            )

    def cleanup(self) -> None:
        """Clean up CUDA resources."""
        if self.use_cuda:
            # Free CUDA memory
            self.image_device = None
            cuda.close() 