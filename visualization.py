import colorsys
import math
import os
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import pygame
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

import config
from audio import AudioProcessor
from fractal import FractalGenerator
from utils import clamp, cleanup_temp_files, format_time, suppress_output

console = Console()

class Visualizer:
    """Class to handle visualization and user interaction."""
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.running = False
        self.paused = False
        self.recording = False
        self.video_writer = None
        self.audio_started = False
        self.show_menu = False
        self.selected_menu_item = 0
        self.start_time = 0
        self.current_time = 0
        
        # Initialize Pygame
        with suppress_output():
            pygame.init()
            pygame.mixer.init()
            self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE | pygame.DOUBLEBUF)
            pygame.display.set_caption("Shoegaze Hierarchical Fractal Language Architecture")
        
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, config.MENU_FONT_SIZE)
        
        # Initialize fractal generator
        self.fractal_gen = FractalGenerator(width, height)
        
        # Visualization state
        self.zoom = config.ZOOM
        self.rotation_angle = 0.0
        self.color_phase = config.SMOOTHED_COLOR_PHASE_INIT
        self.smoothed_c_real = config.SMOOTHED_C_REAL_INIT
        self.smoothed_c_imag = config.SMOOTHED_C_IMAG_INIT
        
        # Initialize color with default values
        initial_hue = 0.0
        r, g, b = colorsys.hsv_to_rgb(initial_hue, config.SATURATION, config.VALUE)
        self.color = np.array([r * 255, g * 255, b * 255], dtype=np.float32)
        
        # Store base dimensions for UI scaling
        self.base_width = width
        self.base_height = height
        self.is_fullscreen = False
        self.ui_scale = 1.0
        
        # Add pan position
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.pan_speed = 0.01
        
        # Original dimensions for aspect ratio
        self.viewport_width = width
        self.viewport_height = height
        self.viewport_x = 0
        self.viewport_y = 0
        
        # Add frame interpolation buffers
        self.prev_image = None
        self.current_image = None
        self.transition_progress = 0.0
        self.transition_speed = 0.1  # Faster transitions (0.1 seconds)
        
        # Add frame timing
        self.last_frame_time = 0
        self.frame_delta = 0
        
        # Menu items - Initialize before UI surfaces
        self.menu_items = [
            ("Brightness", config.BRIGHTNESS_EXPONENT, config.BRIGHTNESS_RANGE, config.BRIGHTNESS_STEP),
            ("Rotation Speed", config.ROTATION_SPEED, config.ROTATION_SPEED_RANGE, config.ROTATION_SPEED_STEP),
            ("Color Speed", config.COLOR_SPEED, config.COLOR_SPEED_RANGE, config.COLOR_SPEED_STEP),
            ("Smoothing", config.ALPHA, config.ALPHA_RANGE, config.ALPHA_STEP)
        ]
        self.temp_menu_items = self.menu_items.copy()
        
        # Initialize UI elements with relative sizes
        self.progress_height = int(0.01 * height)  # 1% of height
        self.progress_surface = None  # Will be created in update_ui_surfaces
        self.progress_rect = None
        self.preview_font = None  # Will be created in update_ui_surfaces
        self.ui_height = int(0.04 * height)  # 4% of height
        self.ui_surface = None
        self.menu_surface = None
        self.progress_hover = False
        self.progress_hover_time = 0.0
        
        # Initialize display and UI surfaces
        self.update_display_mode()
        self.update_ui_surfaces()
        
        # Enable double buffering for smoother rendering
        pygame.event.set_allowed([pygame.QUIT, pygame.KEYDOWN, pygame.VIDEORESIZE])  # Limit event types for better performance

    def start_recording(self) -> None:
        """Start recording visualization to video file."""
        if not self.recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(config.VIDEO_OUTPUT_DIR, f"fractal_vis_{timestamp}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                filename,
                fourcc,
                config.VIDEO_EXPORT_FPS,
                (self.width, self.height)
            )
            self.recording = True
            console.print(f"[green]Started recording to {filename}[/green]")

    def stop_recording(self) -> None:
        """Stop recording and save video file."""
        if self.recording and self.video_writer is not None:
            self.video_writer.release()
            self.recording = False
            console.print("[green]Recording saved[/green]")

    def start_audio(self) -> bool:
        """Initialize and start audio playback."""
        try:
            if not os.path.exists(config.DOWNLOAD_PATH):
                console.print("[red]Error: Audio file not found for playback[/red]")
                return False

            pygame.mixer.music.load(config.DOWNLOAD_PATH)
            pygame.mixer.music.play()
            self.audio_started = True
            return True
        except Exception as e:
            console.print(f"[red]Error starting audio playback: {e}[/red]")
            return False

    def update_display_mode(self) -> None:
        """Update display mode (fullscreen/windowed) and handle resolution changes."""
        if self.is_fullscreen:
            # Get current display info
            display_info = pygame.display.Info()
            new_width, new_height = display_info.current_w, display_info.current_h
            
            # Set fullscreen mode
            self.screen = pygame.display.set_mode(
                (new_width, new_height),
                pygame.FULLSCREEN | pygame.DOUBLEBUF
            )
        else:
            # Return to windowed mode with original dimensions
            self.screen = pygame.display.set_mode(
                (self.base_width, self.base_height),
                pygame.RESIZABLE | pygame.DOUBLEBUF
            )
        
        # Update current dimensions
        self.width = self.screen.get_width()
        self.height = self.screen.get_height()
        
        # Calculate UI scale based on height ratio
        self.ui_scale = self.height / self.base_height
        
        # Update viewport dimensions
        self.handle_resize(self.width, self.height)

    def update_ui_surfaces(self) -> None:
        """Update UI surface sizes based on current dimensions and scale."""
        # Update progress bar
        self.progress_height = max(10, int(0.01 * self.height))  # At least 10 pixels, 1% of height
        self.progress_surface = pygame.Surface((self.width, self.progress_height))
        self.progress_rect = pygame.Rect(0, self.height - self.progress_height, self.width, self.progress_height)
        
        # Update fonts
        base_font_size = config.MENU_FONT_SIZE
        self.font = pygame.font.Font(None, int(base_font_size * self.ui_scale))
        self.preview_font = pygame.font.Font(None, int(24 * self.ui_scale))
        
        # Update UI surface
        self.ui_height = max(30, int(0.04 * self.height))  # At least 30 pixels, 4% of height
        self.ui_surface = pygame.Surface((self.width, self.ui_height), pygame.SRCALPHA)
        
        # Update menu surface
        menu_width = int(config.MENU_WIDTH * self.ui_scale)
        menu_item_height = int(config.MENU_ITEM_HEIGHT * self.ui_scale)
        menu_padding = int(config.MENU_PADDING * self.ui_scale)
        
        menu_height = (len(self.temp_menu_items) + 1) * menu_item_height + 2 * menu_padding
        self.menu_surface = pygame.Surface((menu_width, menu_height), pygame.SRCALPHA)

    def handle_resize(self, new_width: int, new_height: int) -> None:
        """Handle window resize event maintaining aspect ratio."""
        self.width = new_width
        self.height = new_height
        
        # Calculate viewport dimensions to maintain aspect ratio
        target_ratio = self.base_width / self.base_height
        current_ratio = new_width / new_height
        
        if current_ratio > target_ratio:
            # Window is too wide
            self.viewport_height = new_height
            self.viewport_width = int(new_height * target_ratio)
            self.viewport_x = (new_width - self.viewport_width) // 2
            self.viewport_y = 0
        else:
            # Window is too tall
            self.viewport_width = new_width
            self.viewport_height = int(new_width / target_ratio)
            self.viewport_x = 0
            self.viewport_y = (new_height - self.viewport_height) // 2
        
        # Update UI scale and surfaces
        self.ui_scale = self.height / self.base_height
        self.update_ui_surfaces()

    def handle_menu_input(self, event: pygame.event.Event) -> None:
        """Handle menu input events."""
        if event.key == pygame.K_UP:
            self.selected_menu_item = (self.selected_menu_item - 1) % (len(self.temp_menu_items) + 1)  # +1 for Apply button
        elif event.key == pygame.K_DOWN:
            self.selected_menu_item = (self.selected_menu_item + 1) % (len(self.temp_menu_items) + 1)
        elif event.key == pygame.K_RETURN and self.selected_menu_item == len(self.temp_menu_items):
            # Apply button selected
            self.menu_items = [item for item in self.temp_menu_items]  # Create a new copy
            
            # Update actual parameters and instance variables
            for i, (name, value, _, _) in enumerate(self.menu_items):
                value = float(value)  # Ensure we're working with float values
                if i == 0:  # Brightness
                    config.BRIGHTNESS_EXPONENT = value
                    # Brightness doesn't need an instance variable as it's used directly in the shader
                elif i == 1:  # Rotation Speed
                    config.ROTATION_SPEED = value
                    # No need to update instance variable as it's added to rotation_angle each frame
                elif i == 2:  # Color Speed
                    config.COLOR_SPEED = value
                    # No need to update instance variable as it's added to color_phase each frame
                elif i == 3:  # Smoothing
                    config.ALPHA = value
                    # No need to update instance variable as it's used directly in smoothing calculations
            
            # Visual feedback that changes were applied
            console.print("[green]Menu changes applied[/green]")
            
        elif event.key in (pygame.K_LEFT, pygame.K_RIGHT) and self.selected_menu_item < len(self.temp_menu_items):
            item = self.temp_menu_items[self.selected_menu_item]
            value = float(item[1])  # Ensure we're working with float values
            value_range = item[2]
            step = item[3]
            
            if event.key == pygame.K_LEFT:
                value = max(value_range[0], value - step)
            else:
                value = min(value_range[1], value + step)
            
            # Update temporary menu items list
            self.temp_menu_items[self.selected_menu_item] = (item[0], value, item[2], item[3])

    def draw_menu(self) -> None:
        """Draw the parameter adjustment menu."""
        if not self.show_menu:
            return

        # Scale menu dimensions
        menu_width = int(config.MENU_WIDTH * self.ui_scale)
        menu_item_height = int(config.MENU_ITEM_HEIGHT * self.ui_scale)
        menu_padding = int(config.MENU_PADDING * self.ui_scale)
        
        # Update menu surface size
        menu_height = (len(self.temp_menu_items) + 1) * menu_item_height + 2 * menu_padding
        self.menu_surface = pygame.Surface((menu_width, menu_height), pygame.SRCALPHA)
        self.menu_surface.fill(config.MENU_BACKGROUND_COLOR)
        
        # Draw menu items with scaled positions
        for i, (name, value, _, _) in enumerate(self.temp_menu_items):
            y = menu_padding + i * menu_item_height
            
            if i == self.selected_menu_item:
                pygame.draw.rect(
                    self.menu_surface,
                    config.MENU_HIGHLIGHT_COLOR,
                    (0, y, menu_width, menu_item_height)
                )
            
            # Draw text and show if value is different from current
            current_value = None
            if i == 0:
                current_value = config.BRIGHTNESS_EXPONENT
            elif i == 1:
                current_value = config.ROTATION_SPEED
            elif i == 2:
                current_value = config.COLOR_SPEED
            elif i == 3:
                current_value = config.ALPHA
            
            text = f"{name}: {value:.4f}"
            if current_value is not None and abs(float(value) - current_value) > 1e-6:
                text += " *"  # Mark changed values with an asterisk
            
            text_surface = self.font.render(text, True, config.MENU_TEXT_COLOR)
            self.menu_surface.blit(text_surface, (menu_padding, y + 5))
        
        # Draw Apply button
        apply_y = menu_padding + len(self.temp_menu_items) * menu_item_height
        if self.selected_menu_item == len(self.temp_menu_items):
            pygame.draw.rect(
                self.menu_surface,
                config.MENU_HIGHLIGHT_COLOR,
                (0, apply_y, menu_width, menu_item_height)
            )
        
        # Show Apply button in different color if there are changes
        has_changes = any(
            abs(float(self.temp_menu_items[i][1]) - 
                [config.BRIGHTNESS_EXPONENT, config.ROTATION_SPEED, 
                 config.COLOR_SPEED, config.ALPHA][i]) > 1e-6
            for i in range(len(self.temp_menu_items))
        )
        
        apply_text = "Apply Changes (Enter)"
        if has_changes:
            apply_text += " *"
            text_color = (255, 200, 100)  # Orange-ish for pending changes
        else:
            text_color = config.MENU_TEXT_COLOR
        
        text_surface = self.font.render(apply_text, True, text_color)
        self.menu_surface.blit(text_surface, (menu_padding, apply_y + 5))
        
        # Draw menu at scaled position
        menu_x = self.width - menu_width - int(10 * self.ui_scale)
        menu_y = int(10 * self.ui_scale)
        self.screen.blit(self.menu_surface, (menu_x, menu_y))

    def draw_menu_hint(self) -> None:
        """Draw menu hint in top right corner."""
        if not self.show_menu:
            hint_text = "Press M for menu, F for fullscreen"
            text_surface = self.font.render(hint_text, True, (255, 255, 255, 180))
            text_rect = text_surface.get_rect()
            text_rect.topright = (self.width - int(10 * self.ui_scale), int(10 * self.ui_scale))
            self.screen.blit(text_surface, text_rect)

    def handle_events(self) -> bool:
        """Handle user input events."""
        keys = pygame.key.get_pressed()
        
        # Handle continuous pan input
        if not self.show_menu:  # Only pan when menu is not shown
            # Scale pan speed based on zoom level
            effective_speed = self.pan_speed / (self.zoom ** 0.5)  # Square root scaling for smoother control
            
            if keys[pygame.K_a]:  # Pan left
                self.pan_x += effective_speed  # Reversed direction for more intuitive control
            if keys[pygame.K_d]:  # Pan right
                self.pan_x -= effective_speed  # Reversed direction for more intuitive control
            if keys[pygame.K_w]:  # Pan up
                self.pan_y += effective_speed  # Reversed direction for more intuitive control
            if keys[pygame.K_s]:  # Pan down
                self.pan_y -= effective_speed  # Reversed direction for more intuitive control
            
            # Constrain panning to keep fractal in view
            max_pan = 2.0 / self.zoom  # Increased range of panning
            self.pan_x = clamp(self.pan_x, -max_pan, max_pan)
            self.pan_y = clamp(self.pan_y, -max_pan, max_pan)
        
        mouse_x, mouse_y = pygame.mouse.get_pos()
        self.progress_hover = self.progress_rect.collidepoint(mouse_x, mouse_y)
        if self.progress_hover:
            self.progress_hover_time = mouse_x / self.width * self.duration
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.VIDEORESIZE:
                self.handle_resize(event.w, event.h)
            elif event.type == pygame.KEYDOWN:
                if event.key == config.ESC_KEY:
                    if self.is_fullscreen:
                        # Exit fullscreen instead of quitting
                        self.is_fullscreen = False
                        self.update_display_mode()
                        return True
                    return False
                elif event.key == config.MENU_KEY:
                    self.show_menu = not self.show_menu
                    if self.show_menu:
                        # Reset temporary menu items when opening menu
                        self.temp_menu_items = self.menu_items.copy()
                elif self.show_menu:
                    self.handle_menu_input(event)
                elif event.key == config.SPACE_KEY:
                    self.paused = not self.paused
                    if self.audio_started:
                        if self.paused:
                            pygame.mixer.music.pause()
                        else:
                            pygame.mixer.music.unpause()
                elif event.key == config.R_KEY:
                    if not self.recording:
                        self.start_recording()
                    else:
                        self.stop_recording()
                elif event.key == config.UP_KEY:
                    # Zoom in towards mouse position
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    rel_x = (mouse_x - self.viewport_x) / self.viewport_width - 0.5
                    rel_y = (mouse_y - self.viewport_y) / self.viewport_height - 0.5
                    
                    old_zoom = self.zoom
                    self.zoom *= (1 + config.ZOOM_STEP)
                    
                    # Adjust pan to keep mouse position fixed
                    zoom_factor = self.zoom / old_zoom - 1
                    self.pan_x -= rel_x * zoom_factor
                    self.pan_y -= rel_y * zoom_factor
                elif event.key == config.DOWN_KEY:
                    # Zoom out from mouse position
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    rel_x = (mouse_x - self.viewport_x) / self.viewport_width - 0.5
                    rel_y = (mouse_y - self.viewport_y) / self.viewport_height - 0.5
                    
                    old_zoom = self.zoom
                    self.zoom *= (1 - config.ZOOM_STEP)
                    
                    # Adjust pan to keep mouse position fixed
                    zoom_factor = self.zoom / old_zoom - 1
                    self.pan_x -= rel_x * zoom_factor
                    self.pan_y -= rel_y * zoom_factor
                elif event.key == config.LEFT_KEY and self.audio_started:
                    # Calculate current position and new target position
                    current_pos = (pygame.time.get_ticks() - self.start_time) / 1000.0
                    new_pos = max(0, current_pos - config.SKIP_SECONDS)
                    # Restart playback from new position
                    pygame.mixer.music.play(start=new_pos)
                    self.start_time = pygame.time.get_ticks() - int(new_pos * 1000)
                elif event.key == config.RIGHT_KEY and self.audio_started:
                    # Calculate current position and new target position
                    current_pos = (pygame.time.get_ticks() - self.start_time) / 1000.0
                    new_pos = min(self.duration, current_pos + config.SKIP_SECONDS)
                    # Restart playback from new position
                    pygame.mixer.music.play(start=new_pos)
                    self.start_time = pygame.time.get_ticks() - int(new_pos * 1000)
                elif event.key == pygame.K_f:  # Add F key for fullscreen toggle
                    self.is_fullscreen = not self.is_fullscreen
                    self.update_display_mode()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # Left click
                if self.progress_hover and self.audio_started:
                    # Calculate new position and seek to it
                    click_position = event.pos[0] / self.width
                    new_time = click_position * self.duration
                    pygame.mixer.music.play(start=new_time)
                    self.start_time = pygame.time.get_ticks() - int(new_time * 1000)
        return True

    def update_fractal_params(self, features: Dict[str, Any]) -> None:
        """Update fractal parameters based on audio features."""
        if features is None:
            return

        # Store last interesting pattern
        if not hasattr(self, 'last_interesting_pattern'):
            self.last_interesting_pattern = {
                'c_real': self.smoothed_c_real,
                'c_imag': self.smoothed_c_imag,
                'interest_score': 0.0
            }

        # Calculate interest score based on multiple features
        spectral_interest = features['mean_centroid'] / config.MIN_CENTROID  # Normalized spectral centroid
        pitch_interest = (features['mean_f0'] - 65) / (1000 - 65)  # Normalized pitch
        energy_interest = features['rms'] * 2.0  # Amplified RMS
        chroma_interest = np.max(features['mean_chroma'])  # Peak chroma strength
        
        # Add spectral complexity analysis
        spectral_spread = np.std(features['mean_chroma']) * 3.0  # Measure of how "spread out" the frequencies are
        energy_variation = features['onset_strength'] * 2.0  # Capture sudden energy changes
        
        # Calculate textural density - high when multiple frequency bands are active
        chroma_active = np.sum(features['mean_chroma'] > 0.2) / 12.0  # Fraction of active frequency bands
        
        # Combine features into overall interest score with emphasis on texture and energy
        current_interest = (
            spectral_interest * 0.2 +      # Weight spectral content
            pitch_interest * 0.15 +        # Reduced pitch weight
            energy_interest * 0.25 +       # Increased energy weight
            chroma_interest * 0.15 +       # Harmonic content
            spectral_spread * 0.15 +       # New: spectral complexity
            energy_variation * 0.05 +      # New: sudden changes
            chroma_active * 0.05           # New: textural density
        )

        # Calculate fractal parameters
        # Use a combination of pitch and spectral spread for angle
        angle = (
            ((features['mean_f0'] - 65) / (1000 - 65)) * np.pi +  # Base angle from pitch
            (spectral_spread * np.pi)                              # Additional rotation from complexity
        ) % (2 * np.pi)
        
        # Modify radius based on both spectral centroid and energy
        base_radius = 0.7885
        energy_factor = np.clip(features['rms'] * 2.0, 0, 1)  # Normalized energy
        spectral_factor = features['mean_centroid'] / config.MIN_CENTROID
        
        # Combine factors with emphasis on energy for intense moments
        radius = base_radius * (
            0.9 +  # Base scale
            0.15 * energy_factor +  # Energy contribution
            0.05 * spectral_factor  # Spectral contribution
        )
        radius = clamp(radius, 0.7, 0.85)  # Keep within interesting range
        
        # Calculate new target position
        target_c_real = radius * np.cos(angle)
        target_c_imag = radius * np.sin(angle)

        # If current pattern is more interesting than the last one, update it
        if current_interest > self.last_interesting_pattern['interest_score']:
            self.last_interesting_pattern = {
                'c_real': target_c_real,
                'c_imag': target_c_imag,
                'interest_score': current_interest
            }
        
        # Decay the interest score of the stored pattern
        self.last_interesting_pattern['interest_score'] *= 0.995  # Slow decay
        
        # Choose between current target and last interesting pattern
        if current_interest > 0.6:  # High interest threshold
            # Use current target
            final_target_real = target_c_real
            final_target_imag = target_c_imag
        else:
            # Blend with last interesting pattern based on current interest
            blend = current_interest / 0.6  # Normalize to [0, 1]
            final_target_real = blend * target_c_real + (1 - blend) * self.last_interesting_pattern['c_real']
            final_target_imag = blend * target_c_imag + (1 - blend) * self.last_interesting_pattern['c_imag']

        # Clamp values
        final_target_real = clamp(final_target_real, config.C_REAL_MIN, config.C_REAL_MAX)
        final_target_imag = clamp(final_target_imag, config.C_IMAG_MIN, config.C_IMAG_MAX)

        # Use adaptive smoothing - slower for interesting patterns
        base_alpha = config.ALPHA * 0.5
        adaptive_alpha = base_alpha * (1.0 - current_interest * 0.5)  # Reduce alpha for interesting patterns
        
        self.smoothed_c_real = adaptive_alpha * final_target_real + (1 - adaptive_alpha) * self.smoothed_c_real
        self.smoothed_c_imag = adaptive_alpha * final_target_imag + (1 - adaptive_alpha) * self.smoothed_c_imag

        # Update rotation based on energy and spectral content
        energy = features['rms']
        spectral_factor = features['mean_centroid'] / config.MIN_CENTROID
        if energy > 0.1:
            # Rotate faster for high energy + high frequency content
            rotation_speed = config.ROTATION_SPEED * (1.0 + spectral_factor)
            self.rotation_angle += rotation_speed
        self.rotation_angle %= 2 * np.pi

        # Update color based on chroma and energy
        self.color_phase = (self.color_phase + config.COLOR_SPEED) % 1.0
        dominant_chroma = np.argmax(features['mean_chroma'])
        
        # Store last dominant chroma if not exists
        if not hasattr(self, 'last_dominant_chroma'):
            self.last_dominant_chroma = dominant_chroma
            self.last_chroma_strength = 0.0
            self.color_transition_progress = 1.0
            self.target_color = self.color.copy()
        
        # Only change color on strong musical events and when significantly different from last color
        chroma_strength = features['mean_chroma'][dominant_chroma]
        chroma_changed = dominant_chroma != self.last_dominant_chroma
        strength_increased = chroma_strength > self.last_chroma_strength * 1.2  # 20% increase threshold
        
        if ((chroma_strength > 0.3 and energy > 0.12) and  # Lower thresholds for smoother transitions
            (chroma_changed or strength_increased) and
            self.color_transition_progress >= 0.8):  # Wait for current transition to mostly complete
            
            # Base hue on chroma with smoother spectral influence
            spectral_offset = 0.05 * (features['mean_centroid'] / config.MIN_CENTROID)  # Reduced spectral influence
            target_hue = ((dominant_chroma / 12.0) + self.color_phase + spectral_offset) % 1.0
            
            # Smoother saturation adjustment
            base_saturation = config.SATURATION * 0.9  # Slightly reduced base saturation
            energy_boost = min(0.3, energy * 0.4)  # Limited energy influence
            dynamic_saturation = clamp(base_saturation + energy_boost, 0.5, 0.95)
            
            # Convert to RGB
            r, g, b = colorsys.hsv_to_rgb(target_hue, dynamic_saturation, config.VALUE)
            self.target_color = np.array([r * 255, g * 255, b * 255], dtype=np.float32)
            
            # Reset transition progress
            self.color_transition_progress = 0.0
            
            # Update last values
            self.last_dominant_chroma = dominant_chroma
            self.last_chroma_strength = chroma_strength
        
        # Always progress the transition
        self.color_transition_progress = min(1.0, self.color_transition_progress + self.frame_delta * 2.0)
        
        # Smooth color transition using improved easing
        if self.color_transition_progress < 1.0:
            # Use smoother easing function
            ease = 0.5 - math.cos(math.pi * (0.2 + 0.8 * self.color_transition_progress)) * 0.5
            self.color = self.color * (1.0 - ease) + self.target_color * ease

    def draw_ui(self, current_time: float, total_time: float) -> None:
        """Draw UI elements including interactive progress bar."""
        # Draw progress bar background
        self.progress_surface.fill((30, 30, 30))
        
        # Draw progress
        progress = current_time / total_time
        pygame.draw.rect(
            self.progress_surface,
            (100, 200, 100),
            (0, 0, int(self.width * progress), self.progress_height)
        )
        
        # Draw hover preview
        if self.progress_hover:
            # Draw hover line
            hover_x = int(self.progress_hover_time / total_time * self.width)
            pygame.draw.line(
                self.progress_surface,
                (255, 255, 255),
                (hover_x, 0),
                (hover_x, self.progress_height),
                2
            )
            
            # Draw time preview tooltip
            preview_text = format_time(self.progress_hover_time)
            text_surface = self.preview_font.render(preview_text, True, (255, 255, 255))
            text_rect = text_surface.get_rect()
            
            # Position tooltip above progress bar
            tooltip_x = max(10, min(hover_x - text_rect.width // 2, self.width - text_rect.width - 10))
            tooltip_y = self.height - self.progress_height - text_rect.height - 5
            
            # Draw tooltip background
            padding = 4
            background_rect = pygame.Rect(
                tooltip_x - padding,
                tooltip_y - padding,
                text_rect.width + padding * 2,
                text_rect.height + padding * 2
            )
            pygame.draw.rect(self.screen, (0, 0, 0, 180), background_rect)
            pygame.draw.rect(self.screen, (255, 255, 255), background_rect, 1)
            
            # Draw tooltip text
            self.screen.blit(text_surface, (tooltip_x, tooltip_y))
        
        # Update progress bar position when window is resized
        self.progress_rect.y = self.height - self.progress_height
        self.progress_rect.width = self.width
        
        # Draw progress bar
        self.screen.blit(self.progress_surface, (0, self.height - self.progress_height))
        
        # Draw current time
        self.ui_surface.fill((0, 0, 0, 0))
        time_text = f"{format_time(current_time)} / {format_time(total_time)}"
        text_surface = self.font.render(time_text, True, (255, 255, 255))
        self.ui_surface.blit(text_surface, (10, 10))
        self.screen.blit(self.ui_surface, (0, 0))

    def blend_frames(self, prev_frame: np.ndarray, next_frame: np.ndarray, alpha: float) -> np.ndarray:
        """Blend between two frames using linear interpolation."""
        if prev_frame is None or next_frame is None:
            return next_frame if next_frame is not None else prev_frame
        
        # Ensure alpha is between 0 and 1
        alpha = max(0.0, min(1.0, alpha))
        
        # Linear interpolation between frames
        return prev_frame * (1 - alpha) + next_frame * alpha

    def run(self, audio_processor: AudioProcessor) -> None:
        """Main visualization loop."""
        self.running = True
        self.duration = audio_processor.duration
        
        # Start audio playback
        if not self.start_audio():
            return
        
        self.start_time = pygame.time.get_ticks()
        self.last_frame_time = self.start_time
        
        # Pre-fetch first chunk of features
        last_chunk_idx = -1
        current_features = None
        
        with Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Visualizing...", total=audio_processor.total_chunks)
            
            while self.running:
                current_time = pygame.time.get_ticks()
                self.frame_delta = (current_time - self.last_frame_time) / 1000.0
                self.last_frame_time = current_time
                
                self.running = self.handle_events()
                
                if not self.paused:
                    self.current_time = (current_time - self.start_time) / 1000.0
                    chunk_idx = int(self.current_time / config.CHUNK_DURATION)
                    
                    if chunk_idx >= audio_processor.total_chunks or not pygame.mixer.music.get_busy():
                        break
                    
                    # Only get new features if we've moved to a new chunk
                    if chunk_idx != last_chunk_idx:
                        current_features = audio_processor.get_chunk_features(chunk_idx)
                        last_chunk_idx = chunk_idx
                    
                    # Update parameters and generate new frame
                    if current_features is not None:
                        self.update_fractal_params(current_features)
                        
                        # Generate new fractal frame
                        new_image = self.fractal_gen.generate(
                            config.MAX_ITER_DEFAULT,
                            self.zoom,
                            self.smoothed_c_real,
                            self.smoothed_c_imag,
                            self.color,
                            self.rotation_angle
                        )
                        
                        if new_image is not None:
                            # Convert the NumPy array to a Pygame surface first
                            surface = pygame.surfarray.make_surface(np.transpose(new_image, (1, 0, 2)))
                            
                            # Apply panning to the generated image
                            pan_offset_x = int(self.pan_x * self.zoom * self.width / 3.0)
                            pan_offset_y = int(self.pan_y * self.zoom * self.height / 2.0)
                            
                            # Create a larger surface for panning
                            pan_surface = pygame.Surface((self.width + abs(pan_offset_x) * 2, 
                                                        self.height + abs(pan_offset_y) * 2))
                            pan_surface.fill((0, 0, 0))
                            
                            # Center the image on the larger surface
                            center_x = (pan_surface.get_width() - surface.get_width()) // 2
                            center_y = (pan_surface.get_height() - surface.get_height()) // 2
                            pan_surface.blit(surface, (center_x - pan_offset_x, center_y - pan_offset_y))
                            
                            # Crop back to original size
                            crop_rect = pygame.Rect(
                                abs(pan_offset_x), 
                                abs(pan_offset_y), 
                                self.width, 
                                self.height
                            )
                            surface = pan_surface.subsurface(crop_rect)
                            
                            # Scale to viewport
                            scaled_surface = pygame.transform.scale(surface, (self.viewport_width, self.viewport_height))
                            
                            # Update frame buffers with surfaces instead of arrays
                            if self.current_image is not None:
                                self.prev_image = self.current_image.copy()
                            self.current_image = scaled_surface
                            
                            # Skip frame interpolation if we're falling behind
                            if self.frame_delta > 1.0 / config.FPS:
                                self.transition_progress = 1.0
                            else:
                                self.transition_progress = min(1.0, self.transition_progress + self.frame_delta / self.transition_speed)
                            
                            # Clear screen and draw
                            self.screen.fill((0, 0, 0))
                            
                            # Draw either the current frame or blend between frames
                            if self.prev_image is None or self.transition_progress >= 1.0:
                                self.screen.blit(self.current_image, (self.viewport_x, self.viewport_y))
                            else:
                                # Blend frames using alpha
                                self.prev_image.set_alpha(int((1.0 - self.transition_progress) * 255))
                                self.current_image.set_alpha(int(self.transition_progress * 255))
                                self.screen.blit(self.prev_image, (self.viewport_x, self.viewport_y))
                                self.screen.blit(self.current_image, (self.viewport_x, self.viewport_y))
                            
                            # Draw UI
                            self.draw_ui(self.current_time, audio_processor.total_time)
                            self.draw_menu()
                            self.draw_menu_hint()
                            
                            # Record frame if needed
                            if self.recording and self.video_writer is not None:
                                # Get the screen content for recording
                                screen_array = pygame.surfarray.array3d(self.screen)
                                self.video_writer.write(cv2.cvtColor(screen_array, cv2.COLOR_RGB2BGR))
                            
                            pygame.display.flip()
                            progress.update(task, completed=chunk_idx)
                            
                            # Reset transition progress for next frame
                            if self.transition_progress >= 1.0:
                                self.transition_progress = 0.0
                
                # Only sleep if we're ahead of schedule
                frame_time = (pygame.time.get_ticks() - current_time) / 1000.0
                if frame_time < 1.0 / config.FPS:
                    self.clock.tick(config.FPS)
                else:
                    self.clock.tick()  # Just update the clock without sleeping

        self.cleanup()

    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            # Stop and unload audio first
            if self.audio_started:
                pygame.mixer.music.stop()
                pygame.mixer.music.unload()
                pygame.mixer.quit()  # Quit mixer subsystem first
                pygame.time.wait(100)  # Give time for audio to be released
            
            # Clean up other resources
            if self.recording:
                self.stop_recording()
            if self.fractal_gen:
                self.fractal_gen.cleanup()
            
            # Clear image buffers
            self.prev_image = None
            self.current_image = None
            
            # Finally quit pygame
            pygame.quit()
            pygame.time.wait(100)  # Give time for pygame to clean up
            
        except Exception as e:
            console.print(f"[yellow]Warning during cleanup: {e}[/yellow]")
        finally:
            # Try multiple times to clean up temp files
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    cleanup_temp_files()
                    break
                except Exception as e:
                    if attempt < max_attempts - 1:
                        pygame.time.wait(500)  # Wait longer between attempts
                    else:
                        console.print(f"[yellow]Failed to clean up temp files after {max_attempts} attempts: {e}[/yellow]") 