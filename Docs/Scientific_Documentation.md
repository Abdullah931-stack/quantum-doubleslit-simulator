# Quantum Double-Slit Simulator & Wave Interference: Scientific Documentation

This document provides the rigorous physical and mathematical foundation of the Quantum Double-Slit Simulator, validating the propagation algorithms against pure wave mechanics and quantum probability distributions.

---

## 1. Wave Propagation Models

The simulator supports two distinct physical models for wave propagation:

### A. Analytical Model (Fraunhofer Far-Field Approximation)
In the paraxial (small-angle) limit ($x \ll L$), the diffraction pattern of a slit of width $w_j$ centered at position $x_j$ is mathematically described by the Fourier transform of the rectangular aperture, leading to the sinc envelope:

$$\psi_j(x) = w_j \cdot \text{sinc}\left(\frac{w_j \cdot \sin\theta}{\lambda}\right) \cdot \frac{e^{i k d_j}}{\sqrt{d_j}} \cdot K(\theta)$$

Where:
* $k = \frac{2\pi}{\lambda}$ is the wavevector magnitude.
* $d_j = \sqrt{(x - x_j)^2 + L^2}$ is the distance from the slit center to the screen coordinate $x$.
* $\sin\theta = \frac{x - x_j}{d_j}$ is the sine of the diffraction angle.
* $K(\theta) = \frac{1}{2} (1 + \cos\theta) = \frac{1}{2} \left(1 + \frac{L}{d_j}\right)$ is the Kirchhoff obliquity factor.
* $w_j$ is the slit width scaling factor representing the amplitude weight.

#### Code Mapping in [physics.py](file:///D:/Projects/quantum-doubleslit-simulator/physics.py)
* **Equation variables**: Lines 127–134 calculate `d_j`, `sin_theta`, `sinc_env` (using NumPy's normalized `np.sinc` function), and `K`.
* **Field accumulation**: Line 137 adds the amplitude-weighted slit contribution:
  ```python
  psi_screen += wj * sinc_env * np.exp(1j * k * d_j) / np.sqrt(d_j) * K
  ```

---

### B. Numerical Model (Band-Limited Angular Spectrum Method - BL-ASM)
To simulate arbitrary slit widths, near-field (Fresnel) diffraction, and wide-angle non-paraxial behaviors without paraxial approximations, the simulator implements the **Band-limited Angular Spectrum Method (BL-ASM)**. 

#### 1. Transfer Function Formulation
The Angular Spectrum Method solves the Helmholtz equation in the spatial frequency domain. The wavefield $U(x, y)$ at a propagation distance $y$ is obtained by taking the Fourier transform of the initial aperture field $U(x, 0)$, propagating the spatial frequencies using a transfer function $H(f_x, y)$, and transforming back:

$$U(x, y) = \mathcal{F}^{-1} \left\{ \mathcal{F}\{U(x, 0)\} \times H(f_x, y) \right\}$$

The exact transfer function is derived from the Helmholtz equation:

$$H(f_x, y) = \exp\left(i \cdot ky \cdot y\right) = \exp\left(i \cdot 2\pi y \sqrt{\frac{1}{\lambda^2} - f_x^2}\right)$$

Where $f_x$ is the spatial frequency along the transverse axis.

#### 2. Evanescent Wave Filtering
When $|f_x| > \frac{1}{\lambda}$, the term inside the square root becomes negative, resulting in a complex wavevector along $y$ ($k_y = i |k_y|$). This represents evanescent waves that decay exponentially over a distance of a few wavelengths. In numerical simulations, keeping these terms causes extreme numerical growth and instability.
* **Filter Rule**:
  $$H(f_x, y) = 0 \quad \text{for } |f_x| > \frac{1}{\lambda}$$

#### Code Mapping in [physics.py](file:///D:/Projects/quantum-doubleslit-simulator/physics.py)
* **Evanescent evaluation**: Lines 228–234 compute the propagating $y$-wavevector $k_y$:
  ```python
  ky_sq_screen = k**2 - k_x_screen**2
  ky_screen = np.zeros_like(ky_sq_screen)
  valid_screen = ky_sq_screen >= 0
  ky_screen[valid_screen] = np.sqrt(ky_sq_screen[valid_screen])
  ```
* **Transfer function phase**: Line 237 calculates $H$ only for the propagating frequencies:
  ```python
  H_screen[valid_screen] = np.exp(1j * ky_screen[valid_screen] * L)
  ```

#### 3. Spatial Band-Limitation Constraint (BL-ASM)
In discrete systems, if the phase of the transfer function oscillates faster than the Nyquist limit of the frequency grid, it introduces aliasing artifacts and high-angle computational noise. To solve this, the transfer function is band-limited to a maximum spatial frequency $f_{x, \text{limit}}$ that dynamically shrinks as propagation distance $y$ increases:

$$f_{x, \text{limit}}(y) = \frac{1}{\lambda \sqrt{1 + \left(\frac{2 y}{W_{\text{comp}}}\right)^2}}$$

Where $W_{\text{comp}} = M \cdot dx$ is the physical width of the padded computational grid.

#### Code Mapping in [physics.py](file:///D:/Projects/quantum-doubleslit-simulator/physics.py)
* **Frequency Limit**: Line 240 calculates the dynamic bandlimit for the screen plane:
  ```python
  f_limit_screen = 1.0 / (wavelength * np.sqrt(1.0 + (2.0 * L / (M_screen * dx_screen))**2))
  ```

#### 4. Anti-Aliasing Raised Cosine Filter
To prevent Gibbs phenomenon oscillations caused by sharp spectral truncation at $f_{x, \text{limit}}$, a smooth raised-cosine window is applied at the transition boundary from $0.9 f_{x, \text{limit}}$ to $f_{x, \text{limit}}$:

$$w(f) = \begin{cases} 
      1.0 & |f| < 0.9 f_{\text{limit}} \\
      0.5 \left(1 + \cos\left(\pi \frac{|f| - 0.9 f_{\text{limit}}}{0.1 f_{\text{limit}}}\right)\right) & 0.9 f_{\text{limit}} \le |f| \le f_{\text{limit}} \\
      0.0 & |f| > f_{\text{limit}}
   \end{cases}$$

#### Code Mapping in [physics.py](file:///D:/Projects/quantum-doubleslit-simulator/physics.py)
* **Filter construction**: Lines 244–248 implement this smooth window, applying it directly to the screen transfer function on line 250:
  ```python
  transition_mask_screen = (f_abs_screen >= 0.9 * f_limit_screen) & (f_abs_screen <= f_limit_screen)
  w_filter_screen[transition_mask_screen] = 0.5 * (1.0 + np.cos(np.pi * (f_abs_screen[transition_mask_screen] - 0.9 * f_limit_screen) / (0.1 * f_limit_screen)))
  H_screen = H_screen * w_filter_screen
  ```

---

## 2. Aperture Representation & Boundary Discretization

### A. Sub-Pixel Oversampling
To represent slits smaller than the grid spacing ($w < dx$) or slits aligned between grid cells, the simulator calculates the exact fractional overlap area of the slit with each pixel interval $[x_i - \frac{dx}{2}, x_i + \frac{dx}{2}]$:

$$\text{Overlap}(x_i) = \max\left(0, \, \min\left(x_i + \frac{dx}{2}, \, x_{\text{right}}\right) - \max\left(x_i - \frac{dx}{2}, \, x_{\text{left}}\right)\right)$$

$$\text{Fraction}(x_i) = \frac{\text{Overlap}(x_i)}{dx}$$

Where $x_{\text{left}} = x_j - \frac{w_j}{2}$ and $x_{\text{right}} = x_j + \frac{w_j}{2}$. The initial amplitude is scaled by this fraction:

$$U_0(x_i) = w_j \cdot \text{Fraction}(x_i)$$

#### Code Mapping in [physics.py](file:///D:/Projects/quantum-doubleslit-simulator/physics.py)
* **Slit area calculation**: Lines 262–267 compute the exact fractional overlap:
  ```python
  overlap = np.clip(np.minimum(x_pad_screen + dx_screen / 2.0, x_right) - 
                    np.maximum(x_pad_screen - dx_screen / 2.0, x_left), 0.0, None)
  fraction = overlap / dx_screen
  U0_j = wj * fraction
  ```

---

## 3. Quantum Sampling & Bayesian Slit Origin Selection

In quantum mechanics, when a particle lands at coordinate $x_{\text{final}}$ on the screen, the probability that it originated from (or passed through) a specific slit $j$ is conditioned on the wave intensity of that slit relative to the total intensity:

$$P(\text{Slit}_j \mid x_{\text{final}}) = \frac{|\psi_j(x_{\text{final}})|^2}{\sum_k |\psi_k(x_{\text{final}})|^2}$$

Where $\psi_j(x)$ is the individual complex wavefunction contribution from slit $j$ propagated to the screen plane.

#### Code Mapping in [gui.py](file:///D:/Projects/quantum-doubleslit-simulator/gui.py)
* **Local probability calculation**: Lines 847–860 lookup the pre-computed individual wavefunctions `psi_slits` at the nearest screen coordinate index `idx`, compute the conditional probabilities, and sample the source slit:
  ```python
  # Calculate local intensity for each slit at this landing position (Bayesian Origin Sampling)
  weights = [np.abs(psi_slits[i][idx])**2 for i in range(len(active_slits))]
  sum_w = sum(weights)
  if sum_w > 1e-12:
      probs = [w / sum_w for w in weights]
  else:
      probs = static_width_probabilities
  slit_idx = self.rng.choice(len(active_slits), p=probs)
  ```
This eliminates the unphysical "X"-shaped trajectory crossing pattern, matching the quantum mechanical probability currents of Bohmian trajectories.

---

## 4. Modeling Limitations & Approximations

1. **Kirchhoff Obliquity Factor ($K$)**: Assumes scalar wave theory where polarization effects are neglected. Valid for aperture features significantly larger than the wavelength of light ($w \gg \lambda$).
2. **Scalar Approximation**: Neglects vector electromagnetism (electric and magnetic field coupling). Accurate for typical laser wavelengths in the visible spectrum (400–700 nm) diffracting through millimeter-scale slits.
3. **Bandlimit Truncation**: Truncating high spatial frequencies to prevent phase aliasing acts as a low-pass filter, slightly rounding the sharp edges of the wave field very close to the slit barrier ($y < 10\lambda$).

---

## 5. Scientific References

1. **Matsushima, K. & Shimobaba, T. (2009)**. *Band-Limited Angular Spectrum Method for Numerical Propagation of Free-Space Coherent Waves.* Optics Express, Vol. 17, Issue 22, pp. 19662-19673.
2. **Goodman, J. W. (2005)**. *Introduction to Fourier Optics.* Roberts and Company Publishers, 3rd Edition.
3. **Dürr, D., Goldstein, S., & Zanghì, N. (1992)**. *Quantum Equilibrium and the Origin of Absolute Uncertainty.* Journal of Statistical Physics, Vol. 67, pp. 843-907. (Bohmian Trajectories and Bayesian Current Distributions).
