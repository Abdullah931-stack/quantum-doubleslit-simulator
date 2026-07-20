import numpy as np
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QGridLayout, QSlider, QCheckBox, QLabel, QPushButton, 
                             QComboBox, QStatusBar, QScrollArea, QGroupBox, QFrame)
from PyQt6.QtCore import QThread, QObject, pyqtSignal, pyqtSlot, QTimer, QMutex, QMetaObject, Qt
from PyQt6.QtGui import QCursor, QFont, QColor
import pyqtgraph as pg
try:
    from pyqtgraph.opengl import GLSurfacePlotItem, GLBoxItem, GLAxisItem
    from custom_gl_view import CustomGLViewWidget, OPENGL_AVAILABLE, Vector3D
except ImportError:
    CustomGLViewWidget = None
    OPENGL_AVAILABLE = False
    Vector3D = None

from simulation_canvas import SimulationCanvas
import physics

# Global Stylesheet for Sleek Dark Glassmorphism Theme
GUI_STYLESHEET = """
QMainWindow {
    background-color: #0d0d12;
}
QFrame#sidebar_frame {
    background-color: #14141e;
    border-radius: 12px;
    border: 1px solid #232332;
}
QGroupBox {
    color: #e0e0ed;
    border: 1px solid #232332;
    border-radius: 8px;
    margin-top: 15px;
    padding-top: 15px;
    font-weight: bold;
    font-size: 11px;
}
QLabel {
    color: #a0a0b8;
    font-size: 11px;
}
QSlider::groove:horizontal {
    height: 4px;
    background: #232332;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #00ff66;
    width: 14px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 7px;
}
QSlider::handle:horizontal:disabled {
    background: #555566;
}
QCheckBox {
    color: #a0a0b8;
    font-size: 11px;
}
QCheckBox::indicator {
    width: 13px;
    height: 13px;
    border: 1px solid #232332;
    border-radius: 3px;
    background-color: #1a1a26;
}
QCheckBox::indicator:checked {
    background-color: #00ff66;
    border: 1px solid #00ff66;
}
QComboBox {
    background-color: #1a1a26;
    color: #e0e0ed;
    border: 1px solid #232332;
    border-radius: 4px;
    padding: 3px 10px;
    font-size: 11px;
}
QPushButton {
    background-color: #1e1e2d;
    color: #e0e0ed;
    border-radius: 6px;
    padding: 8px 15px;
    font-weight: bold;
    font-size: 11px;
    border: 1px solid #2d2d3d;
}
QPushButton:hover {
    background-color: #2b2b3d;
    border-color: #00ff66;
}
QPushButton#btn_play {
    background-color: #0d4f29;
    border: 1px solid #148043;
}
QPushButton#btn_play:hover {
    background-color: #148043;
    border-color: #00ff66;
}
QPushButton#btn_clear {
    background-color: #4f0d1a;
    border: 1px solid #80142b;
}
QPushButton#btn_clear:hover {
    background-color: #80142b;
    border-color: #ff3366;
}
QStatusBar {
    color: #80809a;
    background-color: #0d0d12;
    font-size: 10px;
}
"""

class PhysicsWorker(QObject):
    """
    Persistent Physics Engine Worker.
    Calculates spatial grids using NumPy. Checks abort flag to cancel early.
    Communicates via Zero-Copy reference passing.
    """
    sig_result_ready = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict, str)
    sig_started = pyqtSignal()
    sig_error = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.mutex = QMutex()
        self.params = None
        self.model_type = 'analytical'
        self.resolution_mode = 'Medium'
        self.abort = False
        
    @pyqtSlot(dict, str, str)
    def request_calculation(self, params, model_type, resolution_mode):
        self.mutex.lock()
        self.params = params
        self.model_type = model_type
        self.resolution_mode = resolution_mode
        self.abort = True  # Signal to abort any ongoing work
        self.mutex.unlock()
        
        QMetaObject.invokeMethod(self, "do_work", Qt.ConnectionType.QueuedConnection)
        
    @pyqtSlot()
    def do_work(self):
        self.mutex.lock()
        params = self.params
        model_type = self.model_type
        res_mode = self.resolution_mode
        self.abort = False
        self.mutex.unlock()
        
        if params is None:
            return
            
        self.sig_started.emit()
        
        def check_abort():
            return self.abort
            
        try:
            Phi_static, I_x, psi_screen, x_grid, base_rgb = physics.compute_simulation_data(
                model_type, params, res_mode, check_abort
            )
            
            if self.abort or Phi_static is None:
                return  # Abort sending result
                
            self.sig_result_ready.emit(Phi_static, I_x, psi_screen, x_grid, base_rgb, params, model_type)
        except Exception as e:
            error_msg = str(e)
            print(f"Error in physics calculation: {error_msg}")
            self.sig_error.emit(error_msg)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Quantum Double-Slit Simulator & Wave Interference")
        self.resize(1280, 800)
        self.setStyleSheet(GUI_STYLESHEET)
        
        # Thread-safe random number generator
        self.rng = np.random.default_rng()
        
        # State variables
        self.model_type = 'analytical'  # Default analytical
        self.resolution_mode = 'Medium'
        self.speed_factor = 1.0
        self.is_running = True
        self.show_3d = OPENGL_AVAILABLE
        self.plot_update_counter = 0
        self.gl_slit_boxes = []
        self.gl_screen_box = None
        self.gl_axis = None
        
        # Setup background thread worker
        self.physics_thread = QThread()
        self.worker = PhysicsWorker()
        self.worker.moveToThread(self.physics_thread)
        self.physics_thread.start()
        
        # Debounce and timing controls
        self.calc_debounce_timer = QTimer(self)
        self.calc_debounce_timer.setSingleShot(True)
        self.calc_debounce_timer.timeout.connect(self.dispatch_physics_calculation)
        
        self.photon_reset_timer = QTimer(self)
        self.photon_reset_timer.setSingleShot(True)
        self.photon_reset_timer.timeout.connect(self.canvas_reset_photons)
        
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.on_animation_frame)
        self.animation_timer.start(16)  # ~60 FPS
        
        # Initialize slit parameters
        self.slits_data = [
            {'x': -0.002, 'w': 0.0004, 'active': True},  # Slit 1
            {'x': 0.002, 'w': 0.0004, 'active': True},   # Slit 2
            {'x': 0.0, 'w': 0.0004, 'active': False},    # Slit 3
            {'x': 0.0, 'w': 0.0004, 'active': False},    # Slit 4
            {'x': 0.0, 'w': 0.0004, 'active': False}     # Slit 5
        ]
        
        # Setup UI
        self.setup_ui_layout()
        
        # Connect worker signals
        self.worker.sig_started.connect(self.on_worker_started)
        self.worker.sig_result_ready.connect(self.on_worker_results_ready)
        self.worker.sig_error.connect(self.on_worker_error)
        
        # Initial draw
        self.trigger_recalculation()

    def setup_ui_layout(self):
        # Central widget
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # 1. Sidebar Controls (Left Column)
        sidebar_frame = QFrame(self)
        sidebar_frame.setObjectName("sidebar_frame")
        sidebar_frame.setFixedWidth(320)
        main_layout.addWidget(sidebar_frame)
        
        sidebar_layout = QVBoxLayout(sidebar_frame)
        sidebar_layout.setContentsMargins(10, 10, 10, 10)
        
        # Scroll Area for sidebar to avoid overflow
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { background-color: transparent; }")
        sidebar_layout.addWidget(scroll)
        
        scroll_content = QWidget()
        scroll_content.setStyleSheet("background-color: transparent;")
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll.setWidget(scroll_content)
        
        # Group A: Model & Resolution Settings
        grp_settings = QGroupBox("Model & Resolution Settings", self)
        settings_grid = QGridLayout(grp_settings)
        
        settings_grid.addWidget(QLabel("Diffraction Model:", grp_settings), 0, 0)
        self.cmb_model = QComboBox(grp_settings)
        self.cmb_model.addItems(["Analytical (Fraunhofer)", "Numerical (Huygens-Fresnel)"])
        self.cmb_model.currentIndexChanged.connect(self.on_model_changed)
        settings_grid.addWidget(self.cmb_model, 0, 1)
        
        settings_grid.addWidget(QLabel("Grid Resolution:", grp_settings), 1, 0)
        self.cmb_res = QComboBox(grp_settings)
        self.cmb_res.addItems(["Low (100x100)", "Medium (200x200)", "High (300x300)"])
        self.cmb_res.setCurrentIndex(1) # Medium default
        self.cmb_res.currentIndexChanged.connect(self.on_resolution_changed)
        settings_grid.addWidget(self.cmb_res, 1, 1)
        
        scroll_layout.addWidget(grp_settings)
        
        # Group B: General Physics Parameters
        grp_physics = QGroupBox("General Physics Parameters", self)
        physics_grid = QGridLayout(grp_physics)
        
        # Wavelength
        physics_grid.addWidget(QLabel("Wavelength (λ):", grp_physics), 0, 0)
        self.lbl_lambda = QLabel("550 nm", grp_physics)
        physics_grid.addWidget(self.lbl_lambda, 0, 1, Qt.AlignmentFlag.AlignRight)
        self.sld_lambda = QSlider(Qt.Orientation.Horizontal, grp_physics)
        self.sld_lambda.setRange(380, 780)
        self.sld_lambda.setValue(550)
        self.sld_lambda.valueChanged.connect(self.on_physics_slider_changed)
        physics_grid.addWidget(self.sld_lambda, 1, 0, 1, 2)
        
        # Screen Distance
        physics_grid.addWidget(QLabel("Screen Distance (L):", grp_physics), 2, 0)
        self.lbl_L = QLabel("2.0 m", grp_physics)
        physics_grid.addWidget(self.lbl_L, 2, 1, Qt.AlignmentFlag.AlignRight)
        self.sld_L = QSlider(Qt.Orientation.Horizontal, grp_physics)
        self.sld_L.setRange(5, 50)  # 0.5m to 5.0m
        self.sld_L.setValue(20)
        self.sld_L.valueChanged.connect(self.on_physics_slider_changed)
        physics_grid.addWidget(self.sld_L, 3, 0, 1, 2)
        
        scroll_layout.addWidget(grp_physics)
        
        # Group C: Slits Controller
        grp_slits = QGroupBox("Main Slits Configuration", self)
        self.slits_layout = QVBoxLayout(grp_slits)
        
        # Slit count ComboBox
        count_layout = QHBoxLayout()
        count_layout.addWidget(QLabel("Number of Slits:", grp_slits))
        self.cmb_slit_count = QComboBox(grp_slits)
        self.cmb_slit_count.addItems(["1", "2", "3", "4", "5"])
        self.cmb_slit_count.setCurrentIndex(1) # Default 2 slits
        self.cmb_slit_count.currentIndexChanged.connect(self.on_slit_count_changed)
        count_layout.addWidget(self.cmb_slit_count)
        self.slits_layout.addLayout(count_layout)
        
        # Widget container for dynamic slit sliders
        self.slit_controls_container = QWidget(grp_slits)
        self.slit_controls_layout = QVBoxLayout(self.slit_controls_container)
        self.slit_controls_layout.setContentsMargins(0, 0, 0, 0)
        self.slits_layout.addWidget(self.slit_controls_container)
        
        scroll_layout.addWidget(grp_slits)
        
        # Group D: Secondary Source Configuration
        grp_secondary = QGroupBox("Secondary Aperture", self)
        sec_grid = QGridLayout(grp_secondary)
        
        self.chk_sec_active = QCheckBox("Enable Secondary Source", grp_secondary)
        self.chk_sec_active.setChecked(False)
        self.chk_sec_active.toggled.connect(self.on_physics_slider_changed)
        sec_grid.addWidget(self.chk_sec_active, 0, 0, 1, 2)
        
        sec_grid.addWidget(QLabel("Position (x):", grp_secondary), 1, 0)
        self.lbl_sec_x = QLabel("0.0 mm", grp_secondary)
        sec_grid.addWidget(self.lbl_sec_x, 1, 1, Qt.AlignmentFlag.AlignRight)
        self.sld_sec_x = QSlider(Qt.Orientation.Horizontal, grp_secondary)
        self.sld_sec_x.setRange(-10, 10)  # -1.0mm to +1.0mm
        self.sld_sec_x.setValue(0)
        self.sld_sec_x.valueChanged.connect(self.on_physics_slider_changed)
        sec_grid.addWidget(self.sld_sec_x, 2, 0, 1, 2)
        
        sec_grid.addWidget(QLabel("Width (w):", grp_secondary), 3, 0)
        self.lbl_sec_w = QLabel("0.4 mm", grp_secondary)
        sec_grid.addWidget(self.lbl_sec_w, 3, 1, Qt.AlignmentFlag.AlignRight)
        self.sld_sec_w = QSlider(Qt.Orientation.Horizontal, grp_secondary)
        self.sld_sec_w.setRange(1, 20)  # 0.1mm to 2.0mm
        self.sld_sec_w.setValue(4)
        self.sld_sec_w.valueChanged.connect(self.on_physics_slider_changed)
        sec_grid.addWidget(self.sld_sec_w, 4, 0, 1, 2)
        
        sec_grid.addWidget(QLabel("Phase Shift (Δφ):", grp_secondary), 5, 0)
        self.lbl_sec_phase = QLabel("0°", grp_secondary)
        sec_grid.addWidget(self.lbl_sec_phase, 5, 1, Qt.AlignmentFlag.AlignRight)
        self.sld_sec_phase = QSlider(Qt.Orientation.Horizontal, grp_secondary)
        self.sld_sec_phase.setRange(0, 360)
        self.sld_sec_phase.setValue(0)
        self.sld_sec_phase.valueChanged.connect(self.on_physics_slider_changed)
        sec_grid.addWidget(self.sld_sec_phase, 6, 0, 1, 2)
        
        sec_grid.addWidget(QLabel("Rel Amplitude (A):", grp_secondary), 7, 0)
        self.lbl_sec_amp = QLabel("1.0", grp_secondary)
        sec_grid.addWidget(self.lbl_sec_amp, 7, 1, Qt.AlignmentFlag.AlignRight)
        self.sld_sec_amp = QSlider(Qt.Orientation.Horizontal, grp_secondary)
        self.sld_sec_amp.setRange(0, 20)  # 0.0 to 2.0
        self.sld_sec_amp.setValue(10)
        self.sld_sec_amp.valueChanged.connect(self.on_physics_slider_changed)
        sec_grid.addWidget(self.sld_sec_amp, 8, 0, 1, 2)
        
        scroll_layout.addWidget(grp_secondary)
        
        # Group E: Simulation Controls
        grp_sim = QGroupBox("Simulation Settings", self)
        sim_layout = QVBoxLayout(grp_sim)
        
        # Display modes
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Display Mode:", grp_sim))
        self.cmb_display_mode = QComboBox(grp_sim)
        self.cmb_display_mode.addItems(["Wave Superposition", "Quantum Particles"])
        self.cmb_display_mode.currentIndexChanged.connect(self.on_display_mode_changed)
        mode_layout.addWidget(self.cmb_display_mode)
        sim_layout.addLayout(mode_layout)
        
        # 3D View Checkbox
        self.chk_show_3d = QCheckBox("Enable 3D View Widget", grp_sim)
        self.chk_show_3d.setChecked(True)
        if not OPENGL_AVAILABLE:
            self.chk_show_3d.setEnabled(False)
            self.chk_show_3d.setText("3D View (OpenGL Missing)")
        self.chk_show_3d.toggled.connect(self.on_3d_toggle_changed)
        sim_layout.addWidget(self.chk_show_3d)
        
        # Sharp Isolines Checkbox & Slider Controls
        self.chk_sharp_isolines = QCheckBox("Enable Sharp Isolines (Contours)", grp_sim)
        self.chk_sharp_isolines.setChecked(False)
        self.chk_sharp_isolines.setEnabled(OPENGL_AVAILABLE)
        self.chk_sharp_isolines.toggled.connect(self.on_sharp_isolines_toggled)
        sim_layout.addWidget(self.chk_sharp_isolines)
        
        self.isolines_container = QWidget(grp_sim)
        isolines_layout = QHBoxLayout(self.isolines_container)
        isolines_layout.setContentsMargins(0, 0, 0, 0)
        self.lbl_isolines_levels = QLabel("Levels: 5", self.isolines_container)
        self.sld_isolines_levels = QSlider(Qt.Orientation.Horizontal, self.isolines_container)
        self.sld_isolines_levels.setRange(3, 20)
        self.sld_isolines_levels.setValue(5)
        self.sld_isolines_levels.valueChanged.connect(self.on_isoline_levels_changed)
        
        isolines_layout.addWidget(QLabel("Isoline Levels:", self.isolines_container))
        isolines_layout.addWidget(self.sld_isolines_levels)
        isolines_layout.addWidget(self.lbl_isolines_levels)
        sim_layout.addWidget(self.isolines_container)
        self.isolines_container.setVisible(False)
        
        # Speed slider
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Speed:", grp_sim))
        self.lbl_speed = QLabel("1.0x", grp_sim)
        speed_layout.addWidget(self.lbl_speed, Qt.AlignmentFlag.AlignRight)
        self.sld_speed = QSlider(Qt.Orientation.Horizontal, grp_sim)
        self.sld_speed.setRange(0, 40)
        self.sld_speed.setValue(10)
        self.sld_speed.valueChanged.connect(self.on_speed_changed)
        sim_layout.addWidget(self.sld_speed)
        
        # 3D View Camera Reset Buttons
        btn_view_layout = QHBoxLayout()
        self.btn_reset_perspective = QPushButton("3D Perspective", grp_sim)
        self.btn_reset_perspective.clicked.connect(self.on_reset_perspective_clicked)
        self.btn_reset_perspective.setEnabled(OPENGL_AVAILABLE)
        
        self.btn_reset_topdown = QPushButton("Top-Down View (90°)", grp_sim)
        self.btn_reset_topdown.clicked.connect(self.on_reset_topdown_clicked)
        self.btn_reset_topdown.setEnabled(OPENGL_AVAILABLE)
        
        btn_view_layout.addWidget(self.btn_reset_perspective)
        btn_view_layout.addWidget(self.btn_reset_topdown)
        sim_layout.addLayout(btn_view_layout)
        
        # Controls Button Group
        btn_layout = QHBoxLayout()
        self.btn_play = QPushButton("Pause", grp_sim)
        self.btn_play.setObjectName("btn_play")
        self.btn_play.clicked.connect(self.on_play_clicked)
        btn_layout.addWidget(self.btn_play)
        
        self.btn_clear = QPushButton("Reset Screen", grp_sim)
        self.btn_clear.setObjectName("btn_clear")
        self.btn_clear.clicked.connect(self.on_clear_clicked)
        btn_layout.addWidget(self.btn_clear)
        sim_layout.addLayout(btn_layout)
        
        scroll_layout.addWidget(grp_sim)
        
        # 2. Main Visualization Panel (Center/Right Column)
        center_widget = QWidget(self)
        main_layout.addWidget(center_widget, stretch=1)
        
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(10)
        
        # Canvas Splitter Layout: Simulation canvas on top, Charts on bottom
        # Let's use QGridLayout to position Canvas, pg.PlotWidgets, and GLViewWidget
        vis_layout = QGridLayout()
        vis_layout.setSpacing(10)
        center_layout.addLayout(vis_layout)
        
        # Top Left: 2D Canvas Renderer
        self.canvas = SimulationCanvas(self)
        self.canvas.setMinimumHeight(380)
        vis_layout.addWidget(self.canvas, 0, 0, 1, 2)
        
        # Bottom Left: PyQtGraph 1D Intensity Plot
        self.plot_intensity = pg.PlotWidget(title="Theoretical vs. Experimental Intensity I(x)")
        self.plot_intensity.setBackground('#111116')
        self.plot_intensity.showGrid(x=True, y=True, alpha=0.15)
        self.curve_theory = self.plot_intensity.plot(pen=pg.mkPen('#00ff66', width=2), name="Theoretical")
        # Histogram step curve for landed photons
        self.curve_hist = self.plot_intensity.plot(pen=pg.mkPen('#ffaa00', width=1.5), name="Histogram")
        vis_layout.addWidget(self.plot_intensity, 1, 0)
        
        # Bottom Center: PyQtGraph Wavefunction ψ(x) Plot
        self.plot_psi = pg.PlotWidget(title="Wave Function Amplitude ψ(x) (Re: Blue, Im: Orange)")
        self.plot_psi.setBackground('#111116')
        self.plot_psi.showGrid(x=True, y=True, alpha=0.15)
        self.curve_psi_real = self.plot_psi.plot(pen=pg.mkPen('#0088ff', width=1.5))
        self.curve_psi_imag = self.plot_psi.plot(pen=pg.mkPen('#ff8800', width=1.5))
        vis_layout.addWidget(self.plot_psi, 1, 1)
        
        # Top Right: PyQtGraph OpenGL 3D Surface Widget (Visible/Hidden)
        if OPENGL_AVAILABLE:
            self.gl_view = CustomGLViewWidget()
            self.gl_view.setMinimumHeight(380)
            self.gl_view.setBackgroundColor(QColor(15, 15, 20))
            
            # Add a light semi-transparent reference grid on the floor (XY plane)
            from pyqtgraph.opengl import GLGridItem
            self.gl_grid = GLGridItem()
            self.gl_grid.setSize(x=300, y=300, z=0)
            self.gl_grid.setSpacing(x=10, y=10, z=0)
            self.gl_grid.setColor(QColor(60, 60, 80, 100))
            self.gl_view.addItem(self.gl_grid)
            
            self.gl_view.reset_view_perspective()
            self.gl_surface = GLSurfacePlotItem(computeNormals=False, smooth=False)
            self.gl_view.addItem(self.gl_surface)
            vis_layout.addWidget(self.gl_view, 0, 2, 2, 1)
            # Give specific stretches: top row gets more space, columns distributed nicely
            vis_layout.setRowStretch(0, 3)
            vis_layout.setRowStretch(1, 2)
            vis_layout.setColumnStretch(0, 3)
            vis_layout.setColumnStretch(1, 3)
            vis_layout.setColumnStretch(2, 4)
        else:
            vis_layout.setRowStretch(0, 3)
            vis_layout.setRowStretch(1, 2)
            vis_layout.setColumnStretch(0, 1)
            vis_layout.setColumnStretch(1, 1)
            
        # Status Bar
        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Initialize slit sliders
        self.update_dynamic_slit_controls()

    def update_dynamic_slit_controls(self):
        """Generates dynamic sliders and check boxes for each slit based on Selected N."""
        # Clear previous dynamic sliders
        while self.slit_controls_layout.count():
            item = self.slit_controls_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
                
        slit_count = int(self.cmb_slit_count.currentText())
        
        # Position bounds: depending on count, offset slits
        # Slit positions are auto-distributed, but user can tweak them
        d_val = 0.002 # 2.0 mm default spacing
        
        # Update self.slits_data size
        for idx in range(5):
            self.slits_data[idx]['active'] = (idx < slit_count)
            # Auto-align positions symmetrically:
            if slit_count > 1:
                # Symmetrical offsets
                self.slits_data[idx]['x'] = - (slit_count - 1) * d_val / 2.0 + idx * d_val
            else:
                self.slits_data[idx]['x'] = 0.0
                
        # Generate sliders for active slits
        for idx in range(slit_count):
            s_data = self.slits_data[idx]
            
            box = QGroupBox(f"Slit {idx + 1}", self.slit_controls_container)
            box_layout = QGridLayout(box)
            box_layout.setContentsMargins(5, 5, 5, 5)
            
            # Active Checkbox
            chk = QCheckBox("Active", box)
            chk.setChecked(s_data['active'])
            # Bind using lambda, capturing idx by default argument
            chk.toggled.connect(lambda state, i=idx: self.on_slit_toggled(i, state))
            box_layout.addWidget(chk, 0, 0, 1, 2)
            
            # Position Slider
            box_layout.addWidget(QLabel("Offset:", box), 1, 0)
            lbl_pos = QLabel(f"{s_data['x']*1e3:.2f} mm", box)
            box_layout.addWidget(lbl_pos, 1, 1, Qt.AlignmentFlag.AlignRight)
            
            sld_pos = QSlider(Qt.Orientation.Horizontal, box)
            # Range -10mm to +10mm in steps of 0.1mm (integer range -100 to 100)
            sld_pos.setRange(-100, 100)
            sld_pos.setValue(int(s_data['x'] * 1e4))
            sld_pos.valueChanged.connect(lambda val, i=idx, l=lbl_pos: self.on_slit_pos_changed(i, val, l))
            box_layout.addWidget(sld_pos, 2, 0, 1, 2)
            
            # Width Slider
            box_layout.addWidget(QLabel("Width:", box), 3, 0)
            lbl_w = QLabel(f"{s_data['w']*1e3:.2f} mm", box)
            box_layout.addWidget(lbl_w, 3, 1, Qt.AlignmentFlag.AlignRight)
            
            sld_w = QSlider(Qt.Orientation.Horizontal, box)
            # Range 0.05mm to 2.00mm in steps of 0.05mm (integer range 1 to 40)
            sld_w.setRange(1, 40)
            sld_w.setValue(int(s_data['w'] * 2e4))
            sld_w.valueChanged.connect(lambda val, i=idx, l=lbl_w: self.on_slit_width_changed(i, val, l))
            box_layout.addWidget(sld_w, 4, 0, 1, 2)
            
            self.slit_controls_layout.addWidget(box)
            
        # Trigger calculation
        self.trigger_recalculation()

    # Dynamic Slit Callbacks
    def on_slit_toggled(self, idx, checked):
        self.slits_data[idx]['active'] = checked
        self.trigger_recalculation()

    def on_slit_pos_changed(self, idx, int_val, label):
        val = int_val / 1e4  # Convert to meters
        self.slits_data[idx]['x'] = val
        label.setText(f"{val*1e3:.2f} mm")
        self.trigger_recalculation()

    def on_slit_width_changed(self, idx, int_val, label):
        val = int_val / 2e4  # Convert to meters
        self.slits_data[idx]['w'] = val
        label.setText(f"{val*1e3:.2f} mm")
        self.trigger_recalculation()

    # General Physics Callbacks
    def on_physics_slider_changed(self):
        # Update Wavelength text
        wl = self.sld_lambda.value()
        self.lbl_lambda.setText(f"{wl} nm")
        
        # Update Screen Distance text
        L = self.sld_L.value() / 10.0
        self.lbl_L.setText(f"{L:.1f} m")
        
        # Update Secondary Source texts
        sec_x = self.sld_sec_x.value() / 10.0
        self.lbl_sec_x.setText(f"{sec_x:.2f} mm")
        
        sec_w = self.sld_sec_w.value() / 10.0
        self.lbl_sec_w.setText(f"{sec_w:.1f} mm")
        
        phase = self.sld_sec_phase.value()
        self.lbl_sec_phase.setText(f"{phase}°")
        
        amp = self.sld_sec_amp.value() / 10.0
        self.lbl_sec_amp.setText(f"{amp:.1f}")
        
        self.trigger_recalculation()

    def on_slit_count_changed(self):
        self.update_dynamic_slit_controls()

    def on_model_changed(self, index):
        self.model_type = 'analytical' if index == 0 else 'numerical'
        self.trigger_recalculation()

    def on_resolution_changed(self, index):
        res_modes = ['Low', 'Medium', 'High']
        self.resolution_mode = res_modes[index]
        self.canvas.resolution_mode = self.resolution_mode
        
        # Guard against dimensions mismatch by stopping the timer
        self.animation_timer.stop()
        self.trigger_recalculation()

    def on_display_mode_changed(self, index):
        self.canvas.display_mode = 'wave' if index == 0 else 'photon'
        self.canvas.clear_photons()
        self.update_plots()

    def on_3d_toggle_changed(self, checked):
        self.show_3d = checked
        self.canvas.show_3d = checked
        self.chk_sharp_isolines.setEnabled(checked)
        self.btn_reset_perspective.setEnabled(checked and OPENGL_AVAILABLE)
        self.btn_reset_topdown.setEnabled(checked and OPENGL_AVAILABLE)
        if not checked:
            self.isolines_container.setVisible(False)
        else:
            self.isolines_container.setVisible(self.chk_sharp_isolines.isChecked())
            
        if OPENGL_AVAILABLE:
            self.gl_view.setVisible(checked)
            if checked:
                self.update_3d_plot()

    def on_reset_perspective_clicked(self):
        if OPENGL_AVAILABLE and hasattr(self, 'gl_view'):
            res_map = {'Low': 100, 'Medium': 200, 'High': 300}
            grid_size = res_map.get(self.resolution_mode, 200)
            self.gl_view.reset_view_perspective(grid_size)

    def on_reset_topdown_clicked(self):
        if OPENGL_AVAILABLE and hasattr(self, 'gl_view'):
            res_map = {'Low': 100, 'Medium': 200, 'High': 300}
            grid_size = res_map.get(self.resolution_mode, 200)
            self.gl_view.reset_view_top_down(grid_size)

    def on_sharp_isolines_toggled(self, checked):
        self.isolines_container.setVisible(checked)
        self.update_3d_plot()

    def on_isoline_levels_changed(self, val):
        self.lbl_isolines_levels.setText(f"Levels: {val}")
        self.update_3d_plot()

    def on_speed_changed(self, val):
        self.speed_factor = val / 10.0
        self.lbl_speed.setText(f"{self.speed_factor:.1f}x")

    def on_play_clicked(self):
        self.is_running = not self.is_running
        self.btn_play.setText("Play" if not self.is_running else "Pause")

    def on_clear_clicked(self):
        self.canvas.clear_photons()
        self.update_plots()

    # Recalculation Dispatch and Debouncing
    def trigger_recalculation(self):
        """Starts the 30ms debounce timer for physics recalculation."""
        self.calc_debounce_timer.start(30)
        
        # Reset/Debounce accumulated photon flushes
        # We start a 150ms timer to flush the photons only after sliding stops.
        self.photon_reset_timer.start(150)

    def canvas_reset_photons(self):
        """Debounced reset of accumulated photon dots."""
        self.canvas.clear_photons()
        self.update_plots()

    def dispatch_physics_calculation(self):
        """Dispatches the calculation job to the persistent PhysicsWorker QThread."""
        # 1. Pack physics parameters
        params = {
            'wavelength_nm': float(self.sld_lambda.value()),
            'L': float(self.sld_L.value() / 10.0),
            'slits': self.slits_data,
            'sec_active': self.chk_sec_active.isChecked(),
            'sec_x': float(self.sld_sec_x.value() / 10.0) * 1e-3,      # mm to meters
            'sec_w': float(self.sld_sec_w.value() / 10.0) * 1e-3,      # mm to meters
            'sec_phase': np.deg2rad(float(self.sld_sec_phase.value())),  # deg to rad
            'sec_amp': float(self.sld_sec_amp.value() / 10.0)
        }
        
        # 2. Emit call to worker thread
        self.worker.request_calculation(params, self.model_type, self.resolution_mode)

    # Thread slots
    @pyqtSlot()
    def on_worker_started(self):
        """Displays visual loading indicator when worker begins calculations."""
        self.status_bar.showMessage("Computing wavefield...")
        self.setCursor(QCursor(Qt.CursorShape.WaitCursor))

    @pyqtSlot(str)
    def on_worker_error(self, error_msg):
        """Restores cursor and displays computational error in status bar."""
        self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        self.status_bar.showMessage(f"Computation error: {error_msg}")
        if not self.animation_timer.isActive():
            self.animation_timer.start(16)

    @pyqtSlot(np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict, str)
    def on_worker_results_ready(self, Phi_static, I_x, psi_screen, x_grid, base_rgb, params, model_type):
        """
        Receives fresh spatial matrices from the persistent QThread.
        Guarantees zero-copy safety by updating local references.
        """
        # Restore normal cursor
        self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        
        # Check if sub-sources were capped at 150
        wavelength = float(params['wavelength_nm']) * 1e-9
        active_slits = [s for s in params['slits'] if s['active']]
        capped = False
        for s in active_slits:
            w_slit = float(s['w'])
            if int(4.0 * w_slit / wavelength) > 150:
                capped = True
                break
                
        if params['sec_active']:
            w_sec = float(params['sec_w'])
            if int(4.0 * w_sec / wavelength) > 150:
                capped = True
                
        if capped:
            self.status_bar.showMessage(f"Calculation complete ({model_type.capitalize()} Model) [Warning: Sub-source count capped at 150 to preserve performance]")
        else:
            self.status_bar.showMessage(f"Calculation complete ({model_type.capitalize()} Model)")
        
        # Set parameters in canvas
        self.canvas.set_physics_data(Phi_static, base_rgb, params, model_type)
        
        # Store intensity and wavefunction local references for plots
        self.I_x = I_x
        self.psi_screen = psi_screen
        self.x_grid = x_grid
        
        # Restart animation timer (safe hand-off completed)
        if not self.animation_timer.isActive():
            self.animation_timer.start(16)
            
        # Update 1D curves and plots
        self.update_plots()
        
        # Update OpenGL 3D surface plot if visible
        self.update_3d_plot()

    # Animation Frame Loop
    def on_animation_frame(self):
        if not self.is_running:
            return
            
        # 1. Update 2D Canvas animation (wave propagation and packet travel)
        self.canvas.update_animation(self.speed_factor)
        
        # 2. In Photon Mode: Generate incoming photons based on emission rate
        if self.canvas.display_mode == 'photon' and hasattr(self, 'I_x') and self.I_x is not None:
            # Number of photons to emit in this frame: depends on speed factor
            avg_rate = 5.0 * self.speed_factor
            num_photons = self.rng.poisson(avg_rate)
            
            if num_photons > 0:
                # Sample coordinate using O(N log M) vectorized inverse transform sampler
                land_coords = physics.sample_photons(self.x_grid, self.I_x, num_photons, self.rng)
                
                # Retrieve active slits list to determine photon origins
                active_slits = []
                if self.canvas.params and 'slits' in self.canvas.params:
                    active_slits = [s for s in self.canvas.params['slits'] if s['active']]
                
                # Secondary source is also an active aperture
                if self.canvas.params and self.canvas.params.get('sec_active', False):
                    # Represent secondary source as a virtual slit for origin selection
                    active_slits.append({
                        'x': float(self.canvas.params['sec_x']),
                        'w': float(self.canvas.params['sec_w'])
                    })
                    
                if active_slits:
                    psi_slits = self.canvas.params.get('psi_slits', None)
                    
                    if psi_slits is not None and len(psi_slits) == len(active_slits):
                        dx_grid = self.x_grid[1] - self.x_grid[0]
                        
                        for xc in land_coords:
                            # Find nearest index on the grid for xc
                            idx = int(np.clip(np.round((xc - self.x_grid[0]) / dx_grid), 0, len(self.x_grid) - 1))
                            
                            # Calculate local intensity for each slit at this landing position (Bayesian Origin Sampling)
                            weights = [np.abs(psi_slits[i][idx])**2 for i in range(len(active_slits))]
                            sum_w = sum(weights)
                            
                            if sum_w > 1e-12:
                                probs = [w / sum_w for w in weights]
                            else:
                                # Fallback to static width probabilities if local intensity is zero
                                total_width = sum(float(s['w']) for s in active_slits)
                                probs = [float(s['w']) / total_width for s in active_slits]
                                
                            slit_idx = self.rng.choice(len(active_slits), p=probs)
                            x_start = float(active_slits[slit_idx]['x'])
                            self.canvas.spawn_photon(x_start, xc)
                    else:
                        # Fallback to static width probabilities if psi_slits is missing
                        widths = [float(s['w']) for s in active_slits]
                        total_w = sum(widths)
                        probs = [w / total_w for w in widths]
                        for xc in land_coords:
                            slit_idx = self.rng.choice(len(active_slits), p=probs)
                            x_start = float(active_slits[slit_idx]['x'])
                            self.canvas.spawn_photon(x_start, xc)
                else:
                    for xc in land_coords:
                        self.canvas.spawn_photon(0.0, xc)
                    
            # Update plots histogram only once every 5 frames to avoid UI thread bottlenecks (Decoupled Plotting!)
            self.plot_update_counter = (self.plot_update_counter + 1) % 5
            if self.plot_update_counter == 0:
                self.update_plots()
            
        pass

    def update_plots(self):
        """Updates pyqtgraph curves representing intensity and wavefunction."""
        if not hasattr(self, 'I_x') or self.I_x is None:
            return
            
        # 1. Plot Intensity curve
        x_mm = self.x_grid * 1e3  # Convert meters to mm for graph
        self.curve_theory.setData(x_mm, self.I_x)
        
        # 2. Handle experimental histogram overlay
        if self.canvas.display_mode == 'photon' and self.canvas.photon_count > 0:
            valid_coords = self.canvas.photon_coords[:self.canvas.photon_count]
            # Create histogram
            counts, bins = np.histogram(valid_coords, bins=60, range=(self.x_grid[0], self.x_grid[-1]))
            
            # Normalize histogram to scale with theoretical intensity peak
            max_counts = np.max(counts)
            max_intensity = np.max(self.I_x)
            
            if max_counts > 0:
                normalized_counts = (counts / max_counts) * max_intensity
            else:
                normalized_counts = np.zeros_like(counts)
                
            # Repeat values for stepMode="center"
            bin_centers = 0.5 * (bins[:-1] + bins[1:]) * 1e3
            self.curve_hist.setData(bin_centers, normalized_counts)
            self.curve_hist.setVisible(True)
        else:
            self.curve_hist.setVisible(False)
            
        # 3. Plot Wavefunction Real/Imag curves
        if hasattr(self, 'psi_screen') and self.psi_screen is not None:
            self.curve_psi_real.setData(x_mm, self.psi_screen.real)
            self.curve_psi_imag.setData(x_mm, self.psi_screen.imag)

    def update_3d_plot(self):
        """Updates the OpenGL 3D surface plot. Only updates when visible."""
        if not OPENGL_AVAILABLE or not getattr(self, 'show_3d', False) or not hasattr(self, 'gl_surface'):
            return

        if hasattr(self, 'chk_show_3d') and not self.chk_show_3d.isChecked():
            return
            
        if self.canvas.Phi_static is None:
            return
            
        try:
            # Calculate time-independent intensity (magnitude squared)
            z = np.abs(self.canvas.Phi_static) ** 2
            
            # Normalize heights for clear 3D presentation
            z_max = np.max(z) + 1e-12
            normalized_z = z / z_max
            
            # Apply Non-Linear Dynamic Range Compression (Gamma = 0.5 / Square Root of Intensity)
            # This prevents the Relative Normalization Trap and keeps weaker peaks visible
            normalized_z = normalized_z ** 0.5
            
            # Apply Sharp Isolines (Quantization) if enabled
            if hasattr(self, 'chk_sharp_isolines') and self.chk_sharp_isolines.isChecked():
                N = self.sld_isolines_levels.value()
                # Floor Quantization Step clipped to [0.0, 1.0]
                normalized_z = np.clip(np.floor(normalized_z * N) / (N - 1), 0.0, 1.0)
                
            # Solve the Transpose Trap by transposing NumPy matrix before OpenGL drawing
            z_projected = normalized_z.T * 3.0  # Scale height by 3.0 units
            
            # Continuous Dynamic Glow Shader color mapping
            ny, nx = z_projected.shape
            colors = np.zeros((ny, nx, 4))
            
            c_lambda = self.canvas.base_rgb / 255.0
            z_val = normalized_z.T[:, :, np.newaxis]
            
            # Vectorized two-stage color mapping
            color_low = 2.0 * z_val * c_lambda[np.newaxis, np.newaxis, :]
            color_high = c_lambda[np.newaxis, np.newaxis, :] + 2.0 * (z_val - 0.5) * (1.0 - c_lambda[np.newaxis, np.newaxis, :])
            
            color_rgb = np.where(z_val <= 0.5, color_low, color_high)
            colors[:, :, :3] = color_rgb
            
            # Dynamic Alpha Transparency: alpha(z) = z**0.7 (scaled by max opacity 0.8)
            colors[:, :, 3] = (normalized_z.T ** 0.7) * 0.8
            
            # Flatten to (ny * nx, 4) to prevent internal PyQtGraph index errors
            colors_flat = colors.reshape(-1, 4)
            
            # Solve the Mesh Center Shift: center the mesh itself around (0,0) in world coordinates
            # This aligns the mesh center with the camera target center!
            x_coords = np.linspace(-nx/2.0, nx/2.0, nx)
            y_coords = np.linspace(-ny/2.0, ny/2.0, ny)
            
            # Draw on GPU
            self.gl_surface.setData(x=x_coords, y=y_coords, z=z_projected, colors=colors_flat)
            
            # --- UPDATE PHYSICAL REFERENCE GEOMETRY (SLIT BARS & DETECTOR BOARD) ---
            # 1. Clear old slit boxes and screen box
            if hasattr(self, 'gl_slit_boxes') and self.gl_slit_boxes:
                for box in self.gl_slit_boxes:
                    try:
                        self.gl_view.removeItem(box)
                    except Exception:
                        pass
                self.gl_slit_boxes.clear()
                
            if hasattr(self, 'gl_screen_box') and self.gl_screen_box is not None:
                try:
                    self.gl_view.removeItem(self.gl_screen_box)
                except Exception:
                    pass
                self.gl_screen_box = None
                
            # 2. Re-create Slit Wall blocks dynamically based on current Sliders
            if self.canvas.params:
                params = self.canvas.params
                L = float(params['L'])
                screen_w_physical = 0.05 * L
                # Scale factor from meters to 3D grid coordinate units
                scale = nx / screen_w_physical
                
                active_slits = [s for s in params['slits'] if s['active']]
                active_slits_3d = []
                for s in active_slits:
                    sx_3d = float(s['x']) * scale
                    sw_3d = float(s['w']) * scale
                    active_slits_3d.append((sx_3d, sw_3d))
                    
                # Sort slits from left to right
                active_slits_3d.sort(key=lambda s: s[0])
                
                # Construct solid panels between/around the slits
                x_start = -nx / 2.0
                wall_y = -ny / 2.0
                wall_height = 8.0  # Visual height of the wall panels
                wall_thickness = 1.0
                
                # Color of slit barrier panels: solid dark steel gray
                panel_color = QColor(70, 70, 85, 220)
                
                for i, (sx, sw) in enumerate(active_slits_3d):
                    left_edge = sx - sw / 2.0
                    # Calculate width of the solid panel to the left of this slit
                    w_panel = left_edge - x_start
                    if w_panel > 0.1:
                        panel = GLBoxItem(size=Vector3D(w_panel, wall_thickness, wall_height))
                        panel.translate(x_start, wall_y - wall_thickness/2.0, 0)
                        panel.setColor(panel_color)
                        self.gl_view.addItem(panel)
                        self.gl_slit_boxes.append(panel)
                    # Advance starting position to the right edge of this slit
                    x_start = sx + sw / 2.0
                    
                # Final solid panel to the right edge
                w_panel_end = (nx / 2.0) - x_start
                if w_panel_end > 0.1:
                    panel = GLBoxItem(size=Vector3D(w_panel_end, wall_thickness, wall_height))
                    panel.translate(x_start, wall_y - wall_thickness/2.0, 0)
                    panel.setColor(panel_color)
                    self.gl_view.addItem(panel)
                    self.gl_slit_boxes.append(panel)
                    
                # 3. Create observation screen box at the back
                screen_color = QColor(35, 35, 40, 245) # Dark gray detector board
                self.gl_screen_box = GLBoxItem(size=Vector3D(nx, 1.0, 10.0))
                self.gl_screen_box.translate(-nx/2.0, ny/2.0 + 0.5, 0)
                self.gl_screen_box.setColor(screen_color)
                self.gl_view.addItem(self.gl_screen_box)
                
            # 4. Re-create and translate Axis Gizmo dynamically to front-left corner
            if hasattr(self, 'gl_axis') and self.gl_axis is not None:
                try:
                    self.gl_view.removeItem(self.gl_axis)
                except Exception:
                    pass
            self.gl_axis = GLAxisItem()
            self.gl_axis.setSize(x=15, y=15, z=15)
            self.gl_axis.translate(-nx/2.0 - 15, -ny/2.0, 0)
            self.gl_view.addItem(self.gl_axis)
        except Exception as e:
            print(f"Error in update_3d_plot: {e}")
            pass

    def closeEvent(self, event):
        # Gracefully stop thread worker under mutex lock
        self.worker.mutex.lock()
        self.worker.abort = True
        self.worker.mutex.unlock()
        
        self.physics_thread.quit()
        self.physics_thread.wait()
        event.accept()
