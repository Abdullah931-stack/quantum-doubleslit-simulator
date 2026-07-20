import time
import numpy as np
from PyQt6.QtCore import QTimer, Qt, pyqtSlot
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QColor, QFont
try:
    from pyqtgraph.opengl import GLViewWidget
    from PyQt6.QtGui import QVector3D as Vector3D
    OPENGL_AVAILABLE = True
except ImportError:
    GLViewWidget = QWidget
    Vector3D = None
    OPENGL_AVAILABLE = False

class CustomGLViewWidget(GLViewWidget):
    """
    Subclass of GLViewWidget providing advanced 3D camera controls:
    - Dynamic Focal Center Panning parallel to the view plane (Middle/Right button drag).
    - Stable Top-Down View (up to 90 degrees elevation) without Gimbal Lock.
    - Smooth Inertial Damping on camera rotation and panning.
    - 2D Overlay texts for spatial alignment.
    """
    def __init__(self, parent=None):
        if not OPENGL_AVAILABLE:
            super().__init__(parent)
            return
            
        super().__init__(parent)
        
        # Mouse interaction trackers
        self.last_mouse_pos = None
        self.mouse_pressed_btn = None
        self.last_move_time = 0.0
        
        # Low-pass filter histories for smooth inertia
        self.rot_history = []
        self.pan_history = []
        
        # Inertia state variables
        self.rot_velocity_azimuth = 0.0
        self.rot_velocity_elevation = 0.0
        self.pan_velocity = Vector3D(0.0, 0.0, 0.0)
        self.damping_factor = 0.90  # Exponential decay rate per frame
        
        # Start high-resolution timer for smooth physics damping animation
        self.damping_timer = QTimer(self)
        self.damping_timer.timeout.connect(self.apply_camera_damping)
        self.damping_timer.start(16)  # ~60 FPS

    def mousePressEvent(self, ev):
        if not OPENGL_AVAILABLE:
            super().mousePressEvent(ev)
            return
            
        self.last_mouse_pos = ev.position()
        self.mouse_pressed_btn = ev.button()
        self.last_move_time = time.perf_counter()
        
        # Stop current inertia on user interaction
        self.rot_velocity_azimuth = 0.0
        self.rot_velocity_elevation = 0.0
        self.pan_velocity = Vector3D(0.0, 0.0, 0.0)
        self.rot_history.clear()
        self.pan_history.clear()
        
        ev.accept()

    def mouseMoveEvent(self, ev):
        if not OPENGL_AVAILABLE:
            super().mouseMoveEvent(ev)
            return
            
        if self.last_mouse_pos is None:
            return
            
        pos = ev.position()
        delta_x = pos.x() - self.last_mouse_pos.x()
        delta_y = pos.y() - self.last_mouse_pos.y()
        self.last_mouse_pos = pos
        self.last_move_time = time.perf_counter()
        
        # Mode A: Left Button Drag -> Orbit Rotation
        if self.mouse_pressed_btn == Qt.MouseButton.LeftButton:
            da = -delta_x * 0.3
            de = delta_y * 0.3
            
            self.opts['azimuth'] += da
            self.opts['elevation'] = np.clip(self.opts['elevation'] + de, -90.0, 90.0)
            
            # Record in history for low-pass filter
            self.rot_history.append((da, de))
            if len(self.rot_history) > 4:
                self.rot_history.pop(0)
                
            self.update()
            
        # Mode B: Middle or Right Button Drag -> Parallel Panning
        elif self.mouse_pressed_btn in (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton):
            az = np.radians(self.opts['azimuth'])
            el = np.radians(self.opts['elevation'])
            
            # Local Right vector: orthogonal to azimuth in horizontal plane
            rx = -np.sin(az)
            ry = np.cos(az)
            rz = 0.0
            
            # Local Up vector: perpendicular to Right and view vectors
            ux = -np.sin(el) * np.cos(az)
            uy = -np.sin(el) * np.sin(az)
            uz = np.cos(el)
            
            pan_scale = 0.0015 * self.opts['distance']
            
            # Translate panning inputs to world displacements
            dx = -delta_x * rx * pan_scale + delta_y * ux * pan_scale
            dy = -delta_x * ry * pan_scale + delta_y * uy * pan_scale
            dz = -delta_x * rz * pan_scale + delta_y * uz * pan_scale
            
            center = self.opts['center']
            center.setX(center.x() + dx)
            center.setY(center.y() + dy)
            center.setZ(center.z() + dz)
            
            # Record in history for low-pass filter
            self.pan_history.append(Vector3D(dx, dy, dz))
            if len(self.pan_history) > 4:
                self.pan_history.pop(0)
                
            self.update()
            
        ev.accept()

    def mouseReleaseEvent(self, ev):
        if not OPENGL_AVAILABLE:
            super().mouseReleaseEvent(ev)
            return
            
        # Zero-Drift Mechanics: Detect if user stopped dragging before releasing (T > 40ms)
        time_since_move = (time.perf_counter() - self.last_move_time) * 1000.0
        
        if time_since_move > 40.0:
            # Stopped before release -> Zero inertia velocity
            self.rot_velocity_azimuth = 0.0
            self.rot_velocity_elevation = 0.0
            self.pan_velocity = Vector3D(0.0, 0.0, 0.0)
        else:
            # Fast flick/release -> Apply low-pass filtered velocities for smooth inertia
            if self.mouse_pressed_btn == Qt.MouseButton.LeftButton and self.rot_history:
                self.rot_velocity_azimuth = sum(h[0] for h in self.rot_history) / len(self.rot_history)
                self.rot_velocity_elevation = sum(h[1] for h in self.rot_history) / len(self.rot_history)
            elif self.mouse_pressed_btn in (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton) and self.pan_history:
                vx = sum(v.x() for v in self.pan_history) / len(self.pan_history)
                vy = sum(v.y() for v in self.pan_history) / len(self.pan_history)
                vz = sum(v.z() for v in self.pan_history) / len(self.pan_history)
                self.pan_velocity = Vector3D(vx, vy, vz)
                
        self.rot_history.clear()
        self.pan_history.clear()
        self.last_mouse_pos = None
        self.mouse_pressed_btn = None
        ev.accept()

    def wheelEvent(self, ev):
        """Logarithmic zoom scaling with 5x expanded limit range."""
        if not OPENGL_AVAILABLE:
            super().wheelEvent(ev)
            return
            
        delta = ev.angleDelta().y()
        zoom_factor = 0.85 if delta > 0 else 1.15
        self.opts['distance'] = np.clip(self.opts['distance'] * zoom_factor, 1.0, 500.0)
        self.update()
        ev.accept()

    @pyqtSlot()
    def apply_camera_damping(self):
        """Applies exponential decay and deadzones to camera velocity when idle."""
        if not OPENGL_AVAILABLE or self.mouse_pressed_btn is not None:
            return
            
        needs_update = False
        
        # 1. Rotational inertia decay
        if abs(self.rot_velocity_azimuth) > 0.005 or abs(self.rot_velocity_elevation) > 0.005:
            self.opts['azimuth'] += self.rot_velocity_azimuth
            self.opts['elevation'] = np.clip(self.opts['elevation'] + self.rot_velocity_elevation, -90.0, 90.0)
            
            self.rot_velocity_azimuth *= self.damping_factor
            self.rot_velocity_elevation *= self.damping_factor
            
            if abs(self.rot_velocity_azimuth) < 0.01 and abs(self.rot_velocity_elevation) < 0.01:
                self.rot_velocity_azimuth = 0.0
                self.rot_velocity_elevation = 0.0
            needs_update = True
            
        # 2. Panning inertia decay
        if self.pan_velocity.length() > 0.0005:
            center = self.opts['center']
            center.setX(center.x() + self.pan_velocity.x())
            center.setY(center.y() + self.pan_velocity.y())
            center.setZ(center.z() + self.pan_velocity.z())
            
            self.pan_velocity *= self.damping_factor
            
            if self.pan_velocity.length() < 0.002:
                self.pan_velocity = Vector3D(0.0, 0.0, 0.0)
            needs_update = True
            
        if needs_update:
            self.update()

    def reset_view_perspective(self, grid_size=120):
        """Resolution-Aware Perspective Reset. Centers focus and scales distance."""
        if not OPENGL_AVAILABLE:
            return
        self.opts['center'] = Vector3D(0.0, 0.0, 0.0)
        self.opts['distance'] = float(grid_size) * 1.25
        self.opts['elevation'] = 30.0
        self.opts['azimuth'] = -45.0
        self.update()

    def reset_view_top_down(self, grid_size=120):
        """Resolution-Aware Top-Down Reset. Centers focus and scales distance."""
        if not OPENGL_AVAILABLE:
            return
        self.opts['center'] = Vector3D(0.0, 0.0, 0.0)
        self.opts['distance'] = float(grid_size) * 1.25
        self.opts['elevation'] = 90.0
        self.opts['azimuth'] = 0.0
        self.update()

    def paintGL(self):
        """Overrides paintGL to render a beautiful 2D HUD text overlay on top of the 3D scene."""
        # 1. Call standard OpenGL draw pipeline
        super().paintGL()
        
        # 2. Draw 2D overlays on top of the viewport
        if not OPENGL_AVAILABLE:
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw Top-Left title banner
        painter.setPen(QColor(0, 255, 102, 220)) # Glowing neon green
        painter.setFont(QFont("Outfit", 9, QFont.Weight.Bold))
        painter.drawText(15, 25, "3D WAVE FIELD VISUALIZER")
        
        # Draw Bottom-Left Aperture Plane label
        painter.setPen(QColor(180, 180, 200, 220))
        painter.setFont(QFont("Outfit", 8))
        painter.drawText(15, self.height() - 15, "Aperture Plane (y = 0)")
        
        # Draw Bottom-Right Observation Screen label
        painter.drawText(self.width() - 180, self.height() - 15, "Observation Screen (y = L)")
        
        painter.end()
