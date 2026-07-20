# Quantum Double-Slit Simulator - User Guide

Welcome to the **Quantum Double-Slit Simulator**, a high-fidelity interactive simulation application that visualizes the wave-particle duality of light, quantum superposition, and wave function collapse.

---

## 1. Installation and Execution

### Prerequisites
Make sure you have Python (version 3.11 or newer) installed. The simulator is designed to run efficiently on Windows, macOS, and Linux using Python, NumPy, PyQt6, PyQtGraph, and PyOpenGL.

### Setup Instructions
1. Open a terminal or Command Prompt in the project directory.
2. Install the package dependencies listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python main.py
   ```

---

## 2. Interface Layout

The graphical interface is divided into two primary columns:

### A. Sidebar Control Panel (Left Column)
- **Model & Resolution Settings:**
  - **Diffraction Model:** Toggle between **Analytical (Fraunhofer)** (fast paraxial approximation) and **Numerical (Angular Spectrum Method - ASM)** (high-fidelity non-paraxial Fourier propagator with evanescent wave filtering and dynamic band-limiting).
  - **Grid Resolution:** Tweak grid sizes (Low: 60x60, Medium: 120x120, High: 200x200) to adjust the spatial density of the calculations.
- **General Physics Parameters:**
  - **Wavelength (λ):** Change the wavelength of the laser from 380 nm (violet) to 780 nm (red). The color of the wave pattern updates automatically to represent the actual visible spectrum.
  - **Screen Distance (L):** Adjust the distance between the slit barrier and the detector screen (0.5m to 5.0m).
- **Main Slits Configuration:**
  - Select the number of slits (1 to 5).
  - Modify parameters for each slit independently: toggle active state, adjust offset center position, and change slit width.
- **Secondary Aperture:**
  - Enable an additional diffraction source with adjustable position, width, relative amplitude, and phase shift (0 to 360 degrees) to observe asymmetrical and interferometric patterns.
- **Simulation Settings:**
  - **Display Mode:** Choose **Wave Superposition** (visualizes propagating fields) or **Quantum Particles** (visualizes photon-by-photon emission, angled wave trajectories, and wave function collapse).
  - **3D View:** Toggle the 3D OpenGL surface widget. Turn it off to save CPU/GPU cycles on low-end systems.
  - **Speed & Control:** Pause/play the time progression, adjust simulation speed, or reset accumulated photons.

### B. Visualization Panels (Right Column)
- **2D Simulation Canvas (Top Left):** Displays the propagating wave fronts (in Wave Mode) or the expanding probability packets and collapsing photon impacts (in Particle Mode).
- **1D Intensity Plot (Bottom Left):** Displays the continuous theoretical probability curve (green) overlaid with the normalized experimental photon landing histogram (orange).
- **1D Wavefunction Plot (Bottom Center):** Displays the Real (blue) and Imaginary (orange) components of the screen wave function $\psi(x)$.
- **3D Wave Surface (Right Panel):** Displays the real-time propagating surface plot of wave intensity.

---

## 3. Interactive 3D Camera Controls

The 3D visualization panel features a physical layout including the **Aperture Slit Wall** (at $y=0$) and the **Detector Screen** (at $y=L$), allowing full 3D spatial navigation:
* **Rotate**: Left-click and drag the mouse to orbit around the focal center.
* **Pan / Swim**: Right-click (or middle-click) and drag to translate the camera focal center parallel to the view plane. This allows you to fly or "swim" through the 3D space.
* **Zoom**: Use the scroll wheel to zoom in and out.
* **Top-Down View**: Rotate the camera directly above the plane (elevation $90^\circ$, azimuth $270^\circ$) to observe a 2D topographical intensity contour map.
* **Zero-Drift Inertia**: Releasing the mouse while dragging triggers smooth kinetic momentum decay. If the mouse is held still for more than 50ms before releasing, it immediately anchors without drift.

---

## 4. Physical Verification (Calibration Check)

To verify the physical accuracy of the simulation, you can validate the fringe spacing against the analytical paraxial double-slit interference equation:

$$\Delta x = \frac{\lambda L}{d}$$

### Calibration Test Procedure:
1. Set the **Diffraction Model** to **Analytical**.
2. Select **2 Slits** in the configuration box.
3. Configure the parameters:
   - Wavelength $\lambda = 500\text{ nm} = 500 \times 10^{-9}\text{ m}$.
   - Screen distance $L = 2.0\text{ m}$.
   - Slit 1 Position: $-1.0\text{ mm}$, Slit 2 Position: $+1.0\text{ mm}$ (Yielding spacing $d = 2.0\text{ mm} = 2.0 \times 10^{-3}\text{ m}$).
4. Calculate the theoretical fringe spacing:
   $$\Delta x = \frac{500 \times 10^{-9} \times 2.0}{2.0 \times 10^{-3}} = 5 \times 10^{-4}\text{ m} = 0.50\text{ mm}$$
5. Observe the **1D Intensity Plot**:
   - Verify that the peaks (bright fringes) are spaced exactly $0.50\text{ mm}$ apart on the horizontal $x$-axis.
   - You can run the automated validation script `python test_verification.py` to check that the percentage error is exactly $0.000\%$.
