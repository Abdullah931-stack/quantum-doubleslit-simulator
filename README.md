# Quantum Double-Slit Simulator & Wave Interference

An interactive, high-fidelity desktop simulator that models the wave-particle duality of light, quantum superposition, non-paraxial diffraction propagation, and localized wave function collapse. Built using Python, PyQt6, PyQtGraph, NumPy, and PyOpenGL.

---

## Key Features

1. **Dual Physics Engines**:
   * **Analytical Model**: Fast paraxial Fraunhofer diffraction approximations using sinc envelopes.
   * **Numerical Model (Band-Limited ASM)**: High-fidelity non-paraxial propagation solving the Helmholtz equation in Fourier space. Implements evanescent wave filtering, dynamic band-limiting to prevent phase aliasing, and smooth raised-cosine anti-aliasing windows.
2. **Interactive 3D Visualizer**: Hardware-accelerated real-time 3D surface plot representing wave intensity, with full mouse-driven rotation, translation ("swimming"), zoom, and top-down contour mapping.
3. **Quantum Particle Mode**: Simulates photon-by-photon emission, traveling wave packets, and screen collapse (Born rule).
4. **Bayesian Slit Origin Sampling**: Statistically models photon paths based on local wave intensities, eliminating unrealistic cross-over patterns and matching Bohmian mechanics trajectories.
5. **Adaptive Grid Resampling & Sub-pixel Oversampling**: Resolves tiny slits ($0.05$ mm) far below default grid spacing ($0.33$ mm) using adaptive grids and fractional pixel boundary integration.
6. **Coherent Secondary Source**: Interactive secondary source with phase offset control for advanced interference and asymmetry studies.

---

## Repository Structure

* `main.py`: Entry point for the simulation application.
* `gui.py`: Main window, layout definition, and controller interface.
* `physics.py`: The core physics engine (BL-ASM and paraxial formulations).
* `simulation_canvas.py`: Custom Qt canvas drawing the 2D wavefield and quantum particles.
* `custom_gl_view.py`: OpenGL 3D surface widget subclass.
* `test_verification.py`: Automated physical verification and performance validation suite.
* `requirements.txt`: Package dependencies.
* **[Docs/](./Docs/)**:
  * [Scientific_Documentation.md](./Docs/Scientific_Documentation.md): Academic physics documentation mapping formulas to code.
  * [theory.md](./Docs/theory.md): Physics paper documenting wave equations, ASM propagation, and quantum measurement.
  * [user_guide.md](./Docs/user_guide.md): Operational guide with user interface controls and paraxial calibration checks.

---

## Installation & Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/Abdullah931-stack/quantum-doubleslit-simulator.git
   cd quantum-doubleslit-simulator
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the simulator:
   ```bash
   python main.py
   ```

---

## Physical Verification & Testing

To run the automated physical verification suite testing paraxial convergence, floating-point stability, and sampler performance:
```bash
python test_verification.py
```
Output:
```text
==================================================
RUNNING AUTOMATED PHYSICAL & COMPUTATIONAL TESTS
==================================================
[Test 1] Floating-Point Precision & Epsilon Check... -> PASSED
[Test 2] Fringe Spacing Paraxial Check (Fraunhofer limit)... -> PASSED (Error: 0.000%)
[Test 3] Numerical Model Convergence Check (MSE Match)... -> PASSED (Match: 99.9964%)
[Test 4] Inverse Transform Sampling Speed Check... -> PASSED (Duration: 0.82 ms)
==================================================
```
