# Quantum Wave Interference and Angular Spectrum Diffraction Theory

This document outlines the mathematical and physical foundations of the quantum double-slit (and multi-slit) simulation engine, explaining the transition from classical wave equations to quantum wave function superposition, non-paraxial propagation, and localized particle detection.

---

## 1. The Quantum Wave Function and Schrödinger's Equation

In quantum mechanics, a free particle of mass $m$ propagating in space is described by a complex-valued wave function $\Psi(\mathbf{r}, t)$ which satisfies the time-dependent Schrödinger equation:

$$i\hbar \frac{\partial}{\partial t}\Psi(\mathbf{r}, t) = -\frac{\hbar^2}{2m}\nabla^2\Psi(\mathbf{r}, t)$$

For relativistic massless particles like photons, the propagation is governed by the electromagnetic wave equation, which can be decomposed into monochromatic spatial states of wavelength $\lambda$ and wavenumber $k = \frac{2\pi}{\lambda}$. The spatial component of the wave function $\psi(\mathbf{r})$ satisfies the Helmholtz equation:

$$\left(\nabla^2 + k^2\right)\psi(\mathbf{r}) = 0$$

Under the Born interpretation, the probability of detecting a photon at a specific coordinate $\mathbf{r}$ on a screen is proportional to the probability density function $P(\mathbf{r})$:

$$P(\mathbf{r}) = |\psi(\mathbf{r})|^2 = \psi(\mathbf{r}) \psi^*(\mathbf{r})$$

---

## 2. Feynman Path Integrals and the Superposition Principle

According to Feynman's Path Integral formulation, the total probability amplitude $\psi(A \to B)$ for a particle to travel from state $A$ to state $B$ is the sum of amplitudes over all possible space-time histories:

$$\psi(A \to B) = \int \mathcal{D}[\mathbf{r}(t)] e^{\frac{i}{\hbar} S[\mathbf{r}(t)]}$$

When a barrier with $N$ discrete slit apertures is placed between the source and the detector, all classical paths are blocked except those passing through the openings. The paths can be partitioned into $N$ classes, corresponding to the slit through which the particle passes. Thus, the total wave function at screen coordinate $x$ is the coherent superposition of the individual wave functions emerging from each active slit:

$$\psi_{\text{total}}(x) = \sum_{j \in \text{active}} \psi_j(x) + \psi_{\text{sec}}(x)$$

where $\psi_j(x)$ is the wave function component propagating from slit $j$, and $\psi_{\text{sec}}(x)$ represents the contribution of the secondary coherent diffraction aperture.

---

## 3. Diffraction Models

The simulation engine implements two distinct diffraction regimes:

### A. Analytical Model (Fraunhofer Envelope)
Under the far-field approximation ($L \gg w$), the diffraction envelope of a single slit of width $w$ centered at $x_j$ is given by the Fourier transform of the slit aperture function (a rectangular window). This yields a $\text{sinc}$ envelope:

$$\psi_j(x) = w_j \cdot \text{sinc}\left(\frac{w_j \sin\theta_j}{\lambda}\right) \frac{e^{i k d_j(x)}}{\sqrt{d_j(x)}} \cdot K(\theta_j)$$

where:
- $d_j(x) = \sqrt{(x - x_j)^2 + L^2}$ is the exact Euclidean distance.
- $\sin\theta_j = \frac{x - x_j}{d_j(x)}$ is the exact local angle.
- $K(\theta_j) = 0.5 \cdot (1 + \cos\theta_j)$ is the Kirchhoff obliquity factor.
- $w_j$ is the amplitude weight representing the slit width.

### B. Numerical Model (Band-Limited Angular Spectrum Method - BL-ASM)
To model near-field (Fresnel) diffraction, wide-angle non-paraxial scattering, and sub-pixel slit details without paraxial approximations, the engine solves the Helmholtz equation in the spatial frequency domain using the **Band-limited Angular Spectrum Method (BL-ASM)**.

#### 1. Transfer Function & Fourier Propagation
The wavefield $U(x, y)$ at any propagation distance $y$ along the optical axis is calculated by multiplying the Fourier transform of the initial aperture profile $U(x, 0)$ by the free-space transfer function $H(f_x, y)$ and taking the inverse transform:

$$U(x, y) = \mathcal{F}_x^{-1} \left\{ \mathcal{F}_x\{U(x, 0)\} \times H(f_x, y) \right\}$$

where the exact non-paraxial transfer function is:

$$H(f_x, y) = \exp\left(1j \cdot 2\pi y \sqrt{\frac{1}{\lambda^2} - f_x^2}\right)$$

#### 2. Evanescent Wave Filter
To prevent numerical divergence from decaying complex wavevectors, frequencies where $|f_x| > \frac{1}{\lambda}$ are filtered out:

$$H(f_x, y) = 0 \quad \text{for } |f_x| > \frac{1}{\lambda}$$

#### 3. Band-Limitation Constraint to Prevent Phase Aliasing
To prevent phase aliasing when the transfer function phase oscillates faster than the grid's Nyquist frequency, the frequency range is dynamically limited based on the propagation distance $y$ and the physical width of the computational grid $W_{\text{comp}}$:

$$f_{x, \text{limit}}(y) = \frac{1}{\lambda \sqrt{1 + \left(\frac{2 y}{W_{\text{comp}}}\right)^2}}$$

#### 4. Anti-Aliasing Window (Raised Cosine)
To avoid Gibbs phenomenon oscillations from hard frequency truncation, a smooth raised-cosine window rolls off the frequencies between $0.9 f_{x, \text{limit}}$ and $f_{x, \text{limit}}$:

$$w(f_x) = \begin{cases} 
      1.0 & |f_x| < 0.9 f_{\text{limit}} \\
      0.5 \left(1 + \cos\left(\pi \frac{|f_x| - 0.9 f_{\text{limit}}}{0.1 f_{\text{limit}}}\right)\right) & 0.9 f_{\text{limit}} \le |f_x| \le f_{\text{limit}} \\
      0.0 & |f_x| > f_{\text{limit}}
   \end{cases}$$

---

## 4. Boundary Discretization & Resampling

### A. Sub-Pixel Oversampling
To represent slits smaller than the grid spacing ($w < dx$), the simulator calculates the exact fractional overlap area of the slit with each pixel interval $[x_i - \frac{dx}{2}, x_i + \frac{dx}{2}]$:

$$\text{Fraction}(x_i) = \frac{\max\left(0, \, \min\left(x_i + \frac{dx}{2}, \, x_{\text{right}}\right) - \max\left(x_i - \frac{dx}{2}, \, x_{\text{left}}\right)\right)}{dx}$$

where $x_{\text{left}} = x_j - \frac{w_j}{2}$ and $x_{\text{right}} = x_j + \frac{w_j}{2}$.

### B. Adaptive Grid Resampling
To resolve the high spatial frequencies of narrow slits, the computational grid spacing $dx$ adaptively resamples to satisfy:

$$dx \le \frac{w_{\text{min}}}{4}$$

and the computed fields are complex-linearly interpolated back to the standard display grids.

---

## 5. Inverse Transform Sampling & Bayesian Slit Origin Selection

In Quantum Mechanics, the wave function describes the evolution of probabilities. During measurement, the wave function collapses to a single localized point on the screen. The spatial probability density of the landing coordinate $x$ is given by:

$$P(x) = \frac{I(x)}{\int_{-\infty}^{\infty} I(x') dx'}$$

To simulate this photon-by-photon collapse, the engine uses **Inverse Transform Sampling**:
1. We compute the discrete Cumulative Distribution Function (CDF) across the screen coordinates $x_i$:
   $$CDF(x_i) = \frac{\sum_{n=1}^{i} I(x_n)}{\sum_{n=1}^{M} I(x_n)}$$
2. We draw a uniform random variable $u \sim U(0, 1)$ using an independent thread-safe generator (`default_rng()`).
3. We map $u$ to the coordinate grid via binary search:
   $$x_{\text{sampled}} = x_k \quad \text{where } CDF(x_{k-1}) < u \le CDF(x_k)$$
   This is executed using NumPy's vectorized `searchsorted` utility in $O(N \log M)$ time for $N$ photons and $M$ grid points.

### Bayesian Slit Origin Selection (Conditional Probability)
To eliminate the unphysical "X"-shaped trajectory crossing pattern, the origin slit $x_{\text{start}}$ of a photon landing at coordinate $x_{\text{final}}$ is chosen using conditional probabilities:

$$P(\text{Slit}_j \mid x_{\text{final}}) = \frac{|\psi_j(x_{\text{final}})|^2}{\sum_k |\psi_k(x_{\text{final}})|^2}$$

where $\psi_j$ is the individual wavefunction contribution of slit $j$. This matches Bohmian mechanics trajectories and preserves probability current paths.
