import numpy as np
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QColor, QImage, QBrush, QPen
from PyQt6.QtCore import Qt, QRectF

class SimulationCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Physics matrices (received from worker thread via GUI)
        self.Phi_static = None
        self.base_rgb = np.array([0, 255, 0], dtype=np.float64)  # Default Green
        self.params = None
        self.model_type = 'analytical'
        
        # Display settings
        self.display_mode = 'wave'  # 'wave' or 'photon'
        self.resolution_mode = 'Medium'
        self.show_3d = False
        
        # Phase variable for wave animation
        self.t_phase = 0.0
        
        # Circular buffer for landed photons
        self.max_photons = 100000
        self.photon_coords = np.zeros(self.max_photons, dtype=np.float64)
        self.photon_pointer = 0
        self.photon_count = 0
        
        # Persistent transparent QImage to act as a back-buffer for accumulated photons
        # This converts rendering cost from O(N) to O(1) in the 60 FPS loop
        self.photon_image = None
        
        # Active flying wave packets (for Copenhagen visualization)
        self.active_packets = []
        
        # Keep references of numpy array buffers to prevent Garbage Collection crashes
        self.wave_image_data = None
        self.wave_image = None
        
        # Layout metrics
        self.margin_top = 20
        self.margin_bottom = 40
        self.margin_left = 40
        self.margin_right = 40

    def clear_photons(self):
        """Resets the circular buffer of accumulated photons."""
        self.photon_pointer = 0
        self.photon_count = 0
        self.active_packets.clear()
        if self.photon_image is not None:
            self.photon_image.fill(Qt.GlobalColor.transparent)
        self.update()

    def add_photon_landings_batch(self, x_coords):
        """Adds a list of photon landing events using O(1) buffer and a single QPainter lock."""
        if len(x_coords) == 0:
            return
            
        for x_coord in x_coords:
            self.photon_coords[self.photon_pointer] = x_coord
            self.photon_pointer = (self.photon_pointer + 1) % self.max_photons
            self.photon_count = min(self.photon_count + 1, self.max_photons)
            
        # Draw all points onto the back-buffer QImage using a single painter session (Batch Rendering!)
        if self.photon_image is not None and self.params:
            sim_width = self.width() - (self.margin_left + self.margin_right)
            screen_w_physical = 0.05 * float(self.params['L'])
            
            image_painter = QPainter(self.photon_image)
            image_painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            
            color_theme = QColor(int(self.base_rgb[0]), int(self.base_rgb[1]), int(self.base_rgb[2]), 180)
            image_painter.setPen(Qt.PenStyle.NoPen)
            image_painter.setBrush(QBrush(color_theme))
            
            dot_radius = 2.5
            
            for x_coord in x_coords:
                norm_x = (x_coord + screen_w_physical/2.0) / screen_w_physical
                cx = self.margin_left + norm_x * sim_width
                cy = self.height() - self.margin_bottom
                image_painter.drawEllipse(QRectF(cx - dot_radius, cy - dot_radius, 2.0 * dot_radius, 2.0 * dot_radius))
                
            image_painter.end()

    def recreate_photon_image(self):
        """Re-allocates the persistent QImage buffer and redraws all accumulated photons."""
        if self.width() <= 0 or self.height() <= 0:
            return
            
        self.photon_image = QImage(self.size(), QImage.Format.Format_ARGB32)
        self.photon_image.fill(Qt.GlobalColor.transparent)
        
        # Draw all existing photons currently in the circular buffer using a single painter session
        if self.photon_count > 0 and self.params:
            sim_width = self.width() - (self.margin_left + self.margin_right)
            screen_w_physical = 0.05 * float(self.params['L'])
            
            image_painter = QPainter(self.photon_image)
            image_painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            
            color_theme = QColor(int(self.base_rgb[0]), int(self.base_rgb[1]), int(self.base_rgb[2]), 180)
            image_painter.setPen(Qt.PenStyle.NoPen)
            image_painter.setBrush(QBrush(color_theme))
            
            dot_radius = 2.5
            
            # Draw valid coords
            valid_coords = self.photon_coords[:self.photon_count]
            for xc in valid_coords:
                norm_x = (xc + screen_w_physical/2.0) / screen_w_physical
                cx = self.margin_left + norm_x * sim_width
                cy = self.height() - self.margin_bottom
                image_painter.drawEllipse(QRectF(cx - dot_radius, cy - dot_radius, 2.0 * dot_radius, 2.0 * dot_radius))
                
            image_painter.end()

    def resizeEvent(self, event):
        """Triggers re-allocation and re-drawing of the photon back-buffer on window resize."""
        super().resizeEvent(event)
        self.recreate_photon_image()

    def spawn_photon(self, start_x_coord, final_x_coord):
        """Spawns a new wave packet that will propagate from start_x to final_x."""
        # A packet starts at the slit plane (y=0) at start_x and moves to the screen (y=1.0) at final_x
        packet = {
            'y': 0.0,
            'start_x': start_x_coord,
            'final_x': final_x_coord,
            'speed': 0.05  # Normalized speed per frame
        }
        self.active_packets.append(packet)

    def update_animation(self, speed_factor):
        """Updates the wave phase and propagates active wave packets."""
        # 1. Update wave phase (Modulo 2*pi to prevent floating-point precision leak)
        omega = 2.0 * np.pi * 0.05 * speed_factor
        self.t_phase = (self.t_phase + omega) % (2.0 * np.pi)
        
        # 2. Propagate and collapse wave packets
        still_active = []
        landed_this_frame = []
        for p in self.active_packets:
            p['y'] += p['speed'] * speed_factor
            if p['y'] >= 1.0:
                landed_this_frame.append(p['final_x'])
            else:
                still_active.append(p)
        self.active_packets = still_active
        
        # Batch render collapsed photons
        if landed_this_frame:
            self.add_photon_landings_batch(landed_this_frame)
            
        self.update()

    def set_physics_data(self, Phi_static, base_rgb, params, model_type):
        """
        Receives fresh spatial matrices from the persistent QThread.
        Guarantees zero-copy safety by updating local references.
        """
        self.Phi_static = Phi_static
        self.base_rgb = base_rgb
        self.params = params
        self.model_type = model_type
        self.update_wave_image()
        self.update()

    def update_wave_image(self):
        """
        Computes 2D wave intensity field, applies direct RGB mapping and clipping,
        and constructs a QImage with explicit NumPy reference retention.
        """
        if self.Phi_static is None:
            self.wave_image = None
            return
            
        # Rotate wave phase in 60 FPS loop (Space-Time Separation)
        field_val = (self.Phi_static.real * np.cos(self.t_phase) + 
                     self.Phi_static.imag * np.sin(self.t_phase))
        
        # Normalize and apply non-linear dynamic range compression (Gamma = 0.5 / Square Root)
        # This resolves the Relative Normalization Trap for unbalanced multi-slits
        max_val = np.max(np.abs(field_val)) + 1e-12
        ratio = np.abs(field_val) / max_val
        compressed_ratio = ratio ** 0.5
        normalized = (np.sign(field_val) * compressed_ratio + 1.0) / 2.0  # Map [-1, 1] to [0, 1]
        
        # Vectorized RGB mapping (SIMD C-speed NumPy array construction)
        rgb_data = normalized[:, :, np.newaxis] * self.base_rgb[np.newaxis, np.newaxis, :]
        
        # Strict clipping to [0, 255] to prevent overflow/wraparound artifacts
        self.wave_image_data = np.clip(rgb_data, 0.0, 255.0).astype(np.uint8)
        
        # Dimensions
        height, width, _ = self.wave_image_data.shape
        bytesPerLine = 3 * width
        
        # Create QImage pointing to NumPy array memory buffer (explicitly kept alive in self.wave_image_data)
        self.wave_image = QImage(
            self.wave_image_data.data,
            width,
            height,
            bytesPerLine,
            QImage.Format.Format_RGB888
        )

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Set dark theme background
        painter.fillRect(self.rect(), QColor(15, 15, 20))
        
        # Define simulation boundaries
        sim_rect = QRectF(
            self.margin_left,
            self.margin_top,
            self.width() - (self.margin_left + self.margin_right),
            self.height() - (self.margin_top + self.margin_bottom)
        )
        
        # Initialize back-buffer if needed
        if self.photon_image is None or self.photon_image.size() != self.size():
            self.recreate_photon_image()
            
        # 1. Draw Wave field in the background
        # In Photon mode, we draw the probability field with a low opacity (15%)
        # In Wave mode, we draw it fully (100%)
        if self.Phi_static is not None:
            self.update_wave_image()
            if self.wave_image is not None:
                # Draw the wave field scaled to fit the simulation rectangle
                painter.setOpacity(1.0 if self.display_mode == 'wave' else 0.15)
                painter.drawImage(sim_rect, self.wave_image)
                painter.setOpacity(1.0)
                
        # 2. Draw physical components (Slits barrier, Screen, Source)
        self.draw_physical_layout(painter, sim_rect)
        
        # 3. Draw active propagating wave packets (Copenhagen visuals)
        if self.display_mode == 'photon':
            self.draw_wave_packets(painter, sim_rect)
            
        # 4. Draw accumulated photon dots on the detector screen
        # instantaneous O(1) blit of the persistent back-buffer QImage
        if self.display_mode == 'photon' and self.photon_image is not None:
            painter.drawImage(self.rect(), self.photon_image)

    def draw_physical_layout(self, painter, rect):
        """Draws the laser source, slits barrier, and detector screen boundaries."""
        # Colors
        color_theme = QColor(int(self.base_rgb[0]), int(self.base_rgb[1]), int(self.base_rgb[2]))
        pen_barrier = QPen(QColor(100, 100, 100, 180), 3)
        pen_active = QPen(color_theme, 2)
        
        # Draw Slit Barrier Plane (At Y = 15% of simulation height)
        barrier_y = rect.top() + rect.height() * 0.15
        
        # If parameters exist, draw slits dynamically
        if self.params and 'slits' in self.params:
            slits = self.params['slits']
            screen_w_physical = 0.05 * float(self.params['L'])  # Match physics width
            
            # Map physical meters coordinates to screen pixel coordinates
            def to_pixel_x(phys_x):
                norm_x = (phys_x + screen_w_physical/2.0) / screen_w_physical
                return rect.left() + norm_x * rect.width()
                
            # Draw barrier chunks
            last_x = rect.left()
            painter.setPen(pen_barrier)
            
            for s in slits:
                sx = float(s['x'])
                sw = float(s['w'])
                active = s['active']
                
                slit_left = to_pixel_x(sx - sw/2.0)
                slit_right = to_pixel_x(sx + sw/2.0)
                
                # Draw solid block from last_x to slit_left
                painter.drawLine(int(last_x), int(barrier_y), int(slit_left), int(barrier_y))
                
                # Draw active slit indicator if active
                if active:
                    painter.setPen(pen_active)
                    painter.drawLine(int(slit_left), int(barrier_y), int(slit_right), int(barrier_y))
                    painter.setPen(pen_barrier)
                    
                last_x = slit_right
                
            # Draw final block to the right edge
            painter.drawLine(int(last_x), int(barrier_y), int(rect.right()), int(barrier_y))
            
            # Draw Secondary Aperture if active
            if self.params['sec_active']:
                sec_x = float(self.params['sec_x'])
                sec_w = float(self.params['sec_w'])
                
                sec_left = to_pixel_x(sec_x - sec_w/2.0)
                sec_right = to_pixel_x(sec_x + sec_w/2.0)
                
                painter.setPen(QPen(QColor(230, 150, 30), 2)) # Golden color for secondary source
                painter.drawLine(int(sec_left), int(barrier_y), int(sec_right), int(barrier_y))
                
        else:
            painter.setPen(pen_barrier)
            painter.drawLine(int(rect.left()), int(barrier_y), int(rect.right()), int(barrier_y))
            
        # Draw Detector Screen Plane (At Y = 100% of simulation height)
        screen_y = rect.bottom()
        painter.setPen(QPen(QColor(80, 80, 90), 4))
        painter.drawLine(int(rect.left()), int(screen_y), int(rect.right()), int(screen_y))

    def draw_wave_packets(self, painter, rect):
        """Draws the propagating and expanding probability wave packets (circles)."""
        color_theme = QColor(int(self.base_rgb[0]), int(self.base_rgb[1]), int(self.base_rgb[2]))
        
        barrier_y = rect.top() + rect.height() * 0.15
        screen_y = rect.bottom()
        prop_height = screen_y - barrier_y
        
        if not self.params:
            return
            
        screen_w_physical = 0.05 * float(self.params['L'])
        def to_pixel_x(phys_x):
            norm_x = (phys_x + screen_w_physical/2.0) / screen_w_physical
            return rect.left() + norm_x * rect.width()
            
        for p in self.active_packets:
            cy = barrier_y + p['y'] * prop_height
            # Angled trajectory position calculation: x(y) = start_x + y * (final_x - start_x)
            current_x = p['start_x'] + p['y'] * (p['final_x'] - p['start_x'])
            cx = to_pixel_x(current_x)
            
            max_radius = 25.0
            radius = 2.0 + p['y'] * max_radius
            
            opacity = int((1.0 - p['y']) * 180)  # Fades out near screen
            painter.setPen(QPen(QColor(color_theme.red(), color_theme.green(), color_theme.blue(), opacity), 2))
            painter.setBrush(QBrush(QColor(color_theme.red(), color_theme.green(), color_theme.blue(), opacity // 4)))
            
            painter.drawEllipse(QRectF(cx - radius, cy - radius, 2.0 * radius, 2.0 * radius))
