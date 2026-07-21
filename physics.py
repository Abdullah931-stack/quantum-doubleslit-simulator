import numpy as np

# Uniform Epsilon to prevent numerical singularity and division by zero
EPSILON = 1e-12

def wavelength_to_rgb(wavelength_nm):
    """
    Converts a wavelength in nanometers (380 - 780) to an RGB vector [R, G, B].
    Applies gamma correction and edge attenuation to simulate human eye response.
    Returns a numpy array of shape (3,) with values in [0.0, 255.0].
    """
    wl = float(wavelength_nm)
    if wl < 380.0 or wl > 780.0:
        return np.array([0.0, 0.0, 0.0], dtype=np.float64)
        
    if 380.0 <= wl < 440.0:
        r = -(wl - 440.0) / (440.0 - 380.0)
        g = 0.0
        b = 1.0
    elif 440.0 <= wl < 490.0:
        r = 0.0
        g = (wl - 440.0) / (490.0 - 440.0)
        b = 1.0
    elif 490.0 <= wl < 510.0:
        r = 0.0
        g = 1.0
        b = -(wl - 510.0) / (510.0 - 490.0)
    elif 510.0 <= wl < 580.0:
        r = (wl - 510.0) / (580.0 - 510.0)
        g = 1.0
        b = 0.0
    elif 580.0 <= wl < 645.0:
        r = 1.0
        g = -(wl - 645.0) / (645.0 - 580.0)
        b = 0.0
    else:  # 645.0 <= wl <= 780.0
        r = 1.0
        g = 0.0
        b = 0.0

    # Attenuation factor at the edges of human vision
    if 380.0 <= wl < 420.0:
        factor = 0.3 + 0.7 * (wl - 380.0) / (420.0 - 380.0)
    elif 420.0 <= wl < 701.0:
        factor = 1.0
    elif 701.0 <= wl <= 780.0:
        factor = 0.3 + 0.7 * (780.0 - wl) / (780.0 - 701.0)
    else:
        factor = 0.0

    gamma = 0.8
    r = (r * factor) ** gamma if r > 0 else 0
    g = (g * factor) ** gamma if g > 0 else 0
    b = (b * factor) ** gamma if b > 0 else 0

    return np.array([r, g, b], dtype=np.float64) * 255.0

def compute_simulation_data(model_type, params, resolution_mode, check_abort=None):
    """
    Unified Physics API for double/multi-slit experiment.
    Computes static spatial wavefield Phi_static, screen intensity profile I_x,
    screen wave function psi_screen, screen grid x_grid, and base RGB color vector.
    Supports a thread-safe callback `check_abort` for early cancellation.
    
    Parameters:
        model_type (str): 'analytical' or 'numerical'
        params (dict): Physics parameters
            - wavelength_nm (float)
            - L (float) (screen distance in meters)
            - slits (list of dict): List of slits, each {'x': center_in_m, 'w': width_in_m, 'active': bool}
            - sec_active (bool): Whether secondary source is active
            - sec_x (float): Secondary source position in meters
            - sec_w (float): Secondary source width in meters
            - sec_phase (float): Secondary source phase offset in radians
            - sec_amp (float): Secondary source relative amplitude
        resolution_mode (str): 'Low', 'Medium', or 'High'
        check_abort (callable): Function returning True if calculation should abort.
        
    Returns:
        Phi_static (np.ndarray of complex128): Shape (Ny, Nx) spatial wavefield.
        I_x (np.ndarray of float64): Shape (Nx,) intensity profile on the screen.
        psi_screen (np.ndarray of complex128): Shape (Nx,) wave function on the screen.
        x_grid (np.ndarray of float64): Shape (Nx,) screen coordinates.
        base_rgb (np.ndarray of float64): Shape (3,) base RGB vector.
    """
    # 1. Parse parameters
    wavelength = float(params['wavelength_nm']) * 1e-9  # nm to meters
    L = float(params['L'])
    k = 2.0 * np.pi / wavelength
    
    # 2. Get base RGB
    base_rgb = wavelength_to_rgb(params['wavelength_nm'])
    
    # 3. Define grids
    # Resolution grid sizes (increased to prevent spatial undersampling/aliasing of narrow beams)
    res_map = {'Low': 100, 'Medium': 200, 'High': 300}
    n_space = res_map.get(resolution_mode, 120)
    
    # Screen width (scaled dynamically by L to fit patterns nicely)
    screen_width = 0.05 * L  # 5cm * L
    
    # 1D Screen grid (used for plotting and sampling)
    # Higher resolution for plotting (1001 points) to look clean
    n_screen = 1001
    x_grid = np.linspace(-screen_width/2.0, screen_width/2.0, n_screen, dtype=np.float64)
    
    # 2D Space grids: X is horizontal, Y is propagation axis
    # Y = 0 is slits barrier, Y = L is detector screen
    x_space = np.linspace(-screen_width/2.0, screen_width/2.0, n_space, dtype=np.float64)
    y_space = np.linspace(0.0, L, n_space, dtype=np.float64)
    X, Y = np.meshgrid(x_space, y_space)
    
    # Active slits
    active_slits = [s for s in params['slits'] if s['active']]
    
    # 4. Wavefield Calculation
    if model_type == 'analytical':
        # --- 1D Screen ---
        psi_slits = []
        
        # Slits contribution
        for slit in active_slits:
            if check_abort and check_abort(): return None, None, None, None, None
            xj = float(slit['x'])
            wj = float(slit['w'])
            
            d_j = np.sqrt((x_grid - xj)**2 + L**2)
            d_j = np.clip(d_j, a_min=EPSILON, a_max=None)
            
            sin_theta = (x_grid - xj) / d_j
            sinc_env = np.sinc((wj * sin_theta) / wavelength)
            
            K = 0.5 * (1.0 + L / d_j)  # Kirchhoff obliquity factor
            
            # Multiply by slit width wj to ensure amplitude scales physically
            psi_j = wj * sinc_env * np.exp(1j * k * d_j) / np.sqrt(d_j) * K
            psi_slits.append(psi_j)
            
        # Secondary source contribution
        if params['sec_active']:
            if check_abort and check_abort(): return None, None, None, None, None
            sec_x = float(params['sec_x'])
            sec_w = float(params['sec_w'])
            sec_phase = float(params['sec_phase'])
            sec_amp = float(params['sec_amp'])
            
            d_sec = np.sqrt((x_grid - sec_x)**2 + L**2)
            d_sec = np.clip(d_sec, a_min=EPSILON, a_max=None)
            
            sin_theta_sec = (x_grid - sec_x) / d_sec
            sinc_sec = np.sinc((sec_w * sin_theta_sec) / wavelength)
            
            K_sec = 0.5 * (1.0 + L / d_sec)
            
            # Multiply by secondary width sec_w to scale amplitude physically
            psi_sec = sec_amp * sec_w * sinc_sec * np.exp(1j * (k * d_sec + sec_phase)) / np.sqrt(d_sec) * K_sec
            psi_slits.append(psi_sec)
            
        psi_screen = sum(psi_slits) if psi_slits else np.zeros_like(x_grid, dtype=np.complex128)
        I_x = np.abs(psi_screen)**2
        params['psi_slits'] = psi_slits
        
        # --- 2D Space ---
        Phi_static = np.zeros((n_space, n_space), dtype=np.complex128)
        
        # Slits wave field propagation
        for slit in active_slits:
            if check_abort and check_abort(): return None, None, None, None, None
            xj = float(slit['x'])
            wj = float(slit['w'])
            
            r_j = np.sqrt((X - xj)**2 + Y**2)
            r_j = np.clip(r_j, a_min=EPSILON, a_max=None)
            
            sin_theta_2d = (X - xj) / r_j
            sinc_env_2d = np.sinc((wj * sin_theta_2d) / wavelength)
            
            K_2d = 0.5 * (1.0 + Y / r_j)
            
            # Multiply by slit width wj to scale amplitude physically
            Phi_static += wj * sinc_env_2d * np.exp(1j * k * r_j) / np.sqrt(r_j) * K_2d
            
        # Secondary source wave field propagation
        if params['sec_active']:
            if check_abort and check_abort(): return None, None, None, None, None
            sec_x = float(params['sec_x'])
            sec_w = float(params['sec_w'])
            sec_phase = float(params['sec_phase'])
            sec_amp = float(params['sec_amp'])
            
            r_sec = np.sqrt((X - sec_x)**2 + Y**2)
            r_sec = np.clip(r_sec, a_min=EPSILON, a_max=None)
            
            sin_theta_sec_2d = (X - sec_x) / r_sec
            sinc_sec_2d = np.sinc((sec_w * sin_theta_sec_2d) / wavelength)
            
            K_sec_2d = 0.5 * (1.0 + Y / r_sec)
            
            # Multiply by secondary width sec_w to scale amplitude physically
            Phi_static += sec_amp * sec_w * sinc_sec_2d * np.exp(1j * (k * r_sec + sec_phase)) / np.sqrt(r_sec) * K_sec_2d
            
    else:  # 'numerical' (High Fidelity Angular Spectrum Method - ASM)
        if check_abort and check_abort(): return None, None, None, None, None
        # 1. Determine target dx_space to resolve the smallest slit (Adaptive Grid Resampling)
        w_min = 0.0004  # Default 0.4mm
        if active_slits:
            w_min = min(float(s['w']) for s in active_slits)
            
        # Target spacing to satisfy Nyquist sampling for the slit aperture (dx <= w_min / 4)
        dx_target = w_min / 4.0
        
        # --- 1D Screen ---
        dx_screen_default = screen_width / n_screen
        dx_screen = min(dx_screen_default, dx_target)
        
        # Determine required number of grid points to cover screen_width with dx_screen
        # Plus 8x padding to prevent wrap-around boundary reflections (Spatial Zero-Padding)
        M_screen_required = int(np.ceil(screen_width / dx_screen)) * 8
        # Round to the next power of 2 for optimal FFT performance
        M_screen = int(2 ** np.ceil(np.log2(M_screen_required)))
        M_screen = max(M_screen, 8192)  # Maintain minimum 8192 for 8x padding
        
        # Setup padded computational grid for the screen
        x_pad_screen = np.arange(-M_screen/2, M_screen/2) * dx_screen
        
        # Propagate to L using transfer function H
        freqs_screen = np.fft.fftshift(np.fft.fftfreq(M_screen, d=dx_screen))
        k_x_screen = 2.0 * np.pi * freqs_screen
        ky_sq_screen = k**2 - k_x_screen**2
        
        ky_screen = np.zeros_like(ky_sq_screen)
        valid_screen = ky_sq_screen >= 0
        ky_screen[valid_screen] = np.sqrt(ky_sq_screen[valid_screen])
        
        # Phase propagation term
        H_screen = np.zeros_like(ky_screen, dtype=np.complex128)
        H_screen[valid_screen] = np.exp(1j * ky_screen[valid_screen] * L)
        
        # Apply Band-limited Transfer Function (BL-ASM) with smooth raised cosine window
        f_limit_screen = 1.0 / (wavelength * np.sqrt(1.0 + (2.0 * L / (M_screen * dx_screen))**2))
        f_abs_screen = np.abs(freqs_screen)
        
        w_filter_screen = np.ones_like(freqs_screen)
        w_filter_screen[f_abs_screen > f_limit_screen] = 0.0
        
        # Cosine roll-off transition between 0.9 * f_limit and f_limit (Anti-aliasing Window)
        transition_mask_screen = (f_abs_screen >= 0.9 * f_limit_screen) & (f_abs_screen <= f_limit_screen)
        w_filter_screen[transition_mask_screen] = 0.5 * (1.0 + np.cos(np.pi * (f_abs_screen[transition_mask_screen] - 0.9 * f_limit_screen) / (0.1 * f_limit_screen)))
        
        H_screen = H_screen * w_filter_screen
        
        start_idx_screen = (M_screen - n_screen) // 2
        psi_slits = []
        
        # Propagate each active slit individually (Sub-pixel Oversampling)
        for slit in active_slits:
            if check_abort and check_abort(): return None, None, None, None, None
            xj = float(slit['x'])
            wj = float(slit['w'])
            x_left = xj - wj / 2.0
            x_right = xj + wj / 2.0
            
            # Sub-pixel oversampling fraction
            overlap = np.clip(np.minimum(x_pad_screen + dx_screen / 2.0, x_right) - 
                              np.maximum(x_pad_screen - dx_screen / 2.0, x_left), 0.0, None)
            fraction = overlap / dx_screen
            
            U0_j = wj * fraction
            
            A0_j = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(U0_j)))
            A_prop_j = A0_j * H_screen
            U_prop_j = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(A_prop_j)))
            
            psi_slits.append(U_prop_j)
            
        if params['sec_active']:
            if check_abort and check_abort(): return None, None, None, None, None
            sec_x = float(params['sec_x'])
            sec_w = float(params['sec_w'])
            sec_phase = float(params['sec_phase'])
            sec_amp = float(params['sec_amp'])
            x_left_sec = sec_x - sec_w / 2.0
            x_right_sec = sec_x + sec_w / 2.0
            
            overlap_sec = np.clip(np.minimum(x_pad_screen + dx_screen / 2.0, x_right_sec) - 
                                  np.maximum(x_pad_screen - dx_screen / 2.0, x_left_sec), 0.0, None)
            fraction_sec = overlap_sec / dx_screen
            
            U0_sec = sec_amp * sec_w * fraction_sec * np.exp(1j * sec_phase)
            
            A0_sec = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(U0_sec)))
            A_prop_sec = A0_sec * H_screen
            U_prop_sec = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(A_prop_sec)))
            
            psi_slits.append(U_prop_sec)
            
        # Sum individual fields and interpolate from padded coordinate grid back to display grid (x_grid)
        psi_screen_padded = sum(psi_slits) if psi_slits else np.zeros(M_screen, dtype=np.complex128)
        
        psi_screen_real = np.interp(x_grid, x_pad_screen, psi_screen_padded.real)
        psi_screen_imag = np.interp(x_grid, x_pad_screen, psi_screen_padded.imag)
        psi_screen = psi_screen_real + 1j * psi_screen_imag
        I_x = np.abs(psi_screen)**2
        
        # Re-map psi_slits to the display grid as well for Bayesian origin sampling
        psi_slits_mapped = []
        for psi_p in psi_slits:
            if check_abort and check_abort(): return None, None, None, None, None
            psi_r = np.interp(x_grid, x_pad_screen, psi_p.real)
            psi_i = np.interp(x_grid, x_pad_screen, psi_p.imag)
            psi_slits_mapped.append(psi_r + 1j * psi_i)
        params['psi_slits'] = psi_slits_mapped
        
        if check_abort and check_abort(): return None, None, None, None, None
        
        # --- 2D Space ---
        dx_space_default = screen_width / n_space
        dx_space = min(dx_space_default, dx_target)
        
        M_space_required = int(np.ceil(screen_width / dx_space)) * 8
        M_space = int(2 ** np.ceil(np.log2(M_space_required)))
        M_space = max(M_space, 8 * n_space)  # Maintain minimum 8x padding
        
        x_pad_space = np.arange(-M_space/2, M_space/2) * dx_space
        
        # Build initial aperture mask U0_space
        U0_space = np.zeros(M_space, dtype=np.complex128)
        for slit in active_slits:
            if check_abort and check_abort(): return None, None, None, None, None
            xj = float(slit['x'])
            wj = float(slit['w'])
            x_left = xj - wj / 2.0
            x_right = xj + wj / 2.0
            
            overlap = np.clip(np.minimum(x_pad_space + dx_space / 2.0, x_right) - 
                              np.maximum(x_pad_space - dx_space / 2.0, x_left), 0.0, None)
            fraction = overlap / dx_space
            U0_space += wj * fraction
            
        if params['sec_active']:
            if check_abort and check_abort(): return None, None, None, None, None
            sec_x = float(params['sec_x'])
            sec_w = float(params['sec_w'])
            sec_phase = float(params['sec_phase'])
            sec_amp = float(params['sec_amp'])
            x_left_sec = sec_x - sec_w / 2.0
            x_right_sec = sec_x + sec_w / 2.0
            
            overlap_sec = np.clip(np.minimum(x_pad_space + dx_space / 2.0, x_right_sec) - 
                                  np.maximum(x_pad_space - dx_space / 2.0, x_left_sec), 0.0, None)
            fraction_sec = overlap_sec / dx_space
            U0_space += sec_amp * sec_w * fraction_sec * np.exp(1j * sec_phase)
            
        if check_abort and check_abort(): return None, None, None, None, None
        
        # 2. Transform to Spatial Frequency Domain
        A0_space = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(U0_space)))
        freqs_space = np.fft.fftshift(np.fft.fftfreq(M_space, d=dx_space))
        
        # 3. Propagate to all distances in y_space using vectorized 2D broadcasting
        k_x_space = 2.0 * np.pi * freqs_space
        ky_sq_space = k**2 - k_x_space**2
        
        ky_space_arr = np.zeros_like(ky_sq_space)
        valid_space = ky_sq_space >= 0
        ky_space_arr[valid_space] = np.sqrt(ky_sq_space[valid_space])
        
        H_2d = np.zeros((n_space, M_space), dtype=np.complex128)
        H_2d[:, valid_space] = np.exp(1j * ky_space_arr[valid_space][np.newaxis, :] * y_space[:, np.newaxis])
        
        # Apply 2D Band-limited Transfer Function (BL-ASM) with dynamic f_limit for each row y_i
        y_2d = y_space[:, np.newaxis]
        f_abs_2d = np.abs(freqs_space)[np.newaxis, :]
        f_limit_2d = 1.0 / (wavelength * np.sqrt(1.0 + (2.0 * y_2d / (M_space * dx_space))**2))
        
        # Broadcast variables to matching shapes to prevent boolean indexing mismatches
        f_abs_2d_broadcast = np.broadcast_to(f_abs_2d, (n_space, M_space))
        f_limit_2d_broadcast = np.broadcast_to(f_limit_2d, (n_space, M_space))
        
        w_filter_2d = np.ones((n_space, M_space))
        w_filter_2d[f_abs_2d_broadcast > f_limit_2d_broadcast] = 0.0
        
        # Smooth cosine transition mask (Anti-aliasing Window)
        transition_mask_2d = (f_abs_2d_broadcast >= 0.9 * f_limit_2d_broadcast) & (f_abs_2d_broadcast <= f_limit_2d_broadcast)
        w_filter_2d[transition_mask_2d] = 0.5 * (1.0 + np.cos(np.pi * (f_abs_2d_broadcast[transition_mask_2d] - 0.9 * f_limit_2d_broadcast[transition_mask_2d]) / (0.1 * f_limit_2d_broadcast[transition_mask_2d])))
        
        H_2d = H_2d * w_filter_2d
        
        A_prop = A0_space[np.newaxis, :] * H_2d
        
        if check_abort and check_abort(): return None, None, None, None, None
        
        # 4. Take inverse FFT along transverse axis (axis 1)
        U_prop = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(A_prop, axes=1), axis=1), axes=1)
        
        # Interpolate the real and imaginary parts back to the display grid x_space
        Phi_static = np.zeros((n_space, n_space), dtype=np.complex128)
        for i in range(n_space):
            if i % 10 == 0 and check_abort and check_abort(): return None, None, None, None, None
            Phi_real_interp = np.interp(x_space, x_pad_space, U_prop[i, :].real)
            Phi_imag_interp = np.interp(x_space, x_pad_space, U_prop[i, :].imag)
            Phi_static[i, :] = Phi_real_interp + 1j * Phi_imag_interp
    return Phi_static, I_x, psi_screen, x_grid, base_rgb

def sample_photons(x_grid, I_x, num_photons, rng):
    """
    Thread-safe Inverse Transform Sampling for single photon positions.
    Computes Cumulative Distribution Function (CDF) from intensity curve,
    samples uniform values, and performs O(N log M) vectorized binary search.
    
    Parameters:
        x_grid (np.ndarray): Screen grid coordinates.
        I_x (np.ndarray): Screen intensity profile.
        num_photons (int): Number of photons to sample.
        rng (np.random.Generator): Independent random number generator.
        
    Returns:
        np.ndarray: Sampled coordinates on the screen.
    """
    if num_photons <= 0:
        return np.array([], dtype=np.float64)
        
    # Calculate CDF
    cdf = np.cumsum(I_x)
    total_val = cdf[-1]
    if total_val <= EPSILON:
        # uniform fall back if intensity is completely zero
        return rng.uniform(x_grid[0], x_grid[-1], size=num_photons)
        
    cdf /= total_val
    
    # Draw uniform random samples
    u = rng.random(num_photons)
    
    # Binary search to find indices
    indices = np.searchsorted(cdf, u)
    
    # Map back to screen coordinate space
    return x_grid[indices]
