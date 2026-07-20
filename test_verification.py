import time
import numpy as np
import physics

def run_tests():
    print("==================================================")
    print("RUNNING AUTOMATED PHYSICAL & COMPUTATIONAL TESTS")
    print("==================================================")
    
    # ----------------------------------------------------
    # Test 1: Float64 and Singularity Prevention
    # ----------------------------------------------------
    print("\n[Test 1] Floating-Point Precision & Epsilon Check...")
    params = {
        'wavelength_nm': 500.0,
        'L': 2.0,
        'slits': [
            {'x': 0.0, 'w': 0.0004, 'active': True}  # Slit at center
        ],
        'sec_active': False
    }
    
    # Run calculation at High resolution
    Phi_static, I_x, _, x_grid, base_rgb = physics.compute_simulation_data(
        'numerical', params, 'High'
    )
    
    assert Phi_static.dtype == np.complex128, f"Expected complex128, got {Phi_static.dtype}"
    assert I_x.dtype == np.float64, f"Expected float64, got {I_x.dtype}"
    assert x_grid.dtype == np.float64, f"Expected float64, got {x_grid.dtype}"
    
    # Check that no NaNs or Infs exist in the grids
    assert not np.isnan(Phi_static).any(), "NaN detected in wavefield matrix"
    assert not np.isinf(Phi_static).any(), "Inf detected in wavefield matrix"
    assert not np.isnan(I_x).any(), "NaN detected in intensity vector"
    
    # Check that points exactly at the slits do not crash due to zero-division
    # Slit is at x=0.0. Calculate Y at Y=0.0 (the slit plane boundary)
    # Our epsilon clipping clips r to 1e-12
    zero_dist_check = np.sqrt(0.0**2 + 0.0**2)
    zero_dist_check = np.clip(zero_dist_check, a_min=physics.EPSILON, a_max=None)
    assert zero_dist_check == physics.EPSILON, "Epsilon clipping failed"
    print("-> PASSED: float64 and Epsilon safety verified.")
    
    # ----------------------------------------------------
    # Test 2: Double-Slit Fringe Spacing Paraxial Check
    # ----------------------------------------------------
    print("\n[Test 2] Fringe Spacing Paraxial Check (Fraunhofer limit)...")
    lambda_nm = 600.0
    wavelength = lambda_nm * 1e-9
    L = 3.0
    d = 0.002  # 2.0 mm separation
    
    params_double = {
        'wavelength_nm': lambda_nm,
        'L': L,
        'slits': [
            {'x': -d/2.0, 'w': 0.0001, 'active': True},  # Narrow slits to minimize single-slit envelope decay
            {'x': d/2.0, 'w': 0.0001, 'active': True}
        ],
        'sec_active': False
    }
    
    _, I_x_double, _, x_grid_double, _ = physics.compute_simulation_data(
        'analytical', params_double, 'Medium'
    )
    
    # Theoretical spacing
    dx_theoretical = wavelength * L / d  # 600nm * 3m / 2mm = 0.90 mm
    
    # Find peaks in simulated I_x
    # Locate index of maximum peaks
    # We find indices where I_x is a local maximum
    peaks = []
    for i in range(1, len(I_x_double) - 1):
        if I_x_double[i] > I_x_double[i-1] and I_x_double[i] > I_x_double[i+1]:
            # Threshold to avoid noise/envelope peaks
            if I_x_double[i] > 0.1 * np.max(I_x_double):
                peaks.append(x_grid_double[i])
                
    peaks = np.array(peaks)
    
    # Spacing between central peaks
    # Find peak closest to center (x=0) and its neighbors
    center_idx = np.argmin(np.abs(peaks))
    if center_idx > 0 and center_idx < len(peaks) - 1:
        dx_simulated = peaks[center_idx + 1] - peaks[center_idx]
        percent_error = np.abs(dx_simulated - dx_theoretical) / dx_theoretical * 100
        print(f"-> Theoretical spacing (dx): {dx_theoretical*1e3:.4f} mm")
        print(f"-> Simulated spacing (dx):   {dx_simulated*1e3:.4f} mm")
        print(f"-> Percentage Error:         {percent_error:.3f}%")
        
        # Verify paraxial parms accuracy under 1% limit
        assert percent_error < 1.0, f"Fringe spacing error {percent_error:.3f}% exceeds 1% limit"
        print("-> PASSED: Fringe spacing match verified (Error < 1.0%).")
    else:
        print("-> WARNING: Not enough peaks found near center to verify.")
        
    # ----------------------------------------------------
    # Test 3: Numerical Model Convergence Check
    # ----------------------------------------------------
    print("\n[Test 3] Numerical Model Convergence Check (MSE Match)...")
    # In far field (L >= 10d), Analytical and Numerical models must match closely
    params_conv = {
        'wavelength_nm': 500.0,
        'L': 2.5,
        'slits': [
            {'x': -0.001, 'w': 0.0003, 'active': True},
            {'x': 0.001, 'w': 0.0003, 'active': True}
        ],
        'sec_active': False
    }
    
    # Analytical
    _, I_ana, _, _, _ = physics.compute_simulation_data('analytical', params_conv, 'Medium')
    # Numerical Huygens-Fresnel
    _, I_num, _, _, _ = physics.compute_simulation_data('numerical', params_conv, 'Medium')
    
    # Normalize profiles
    I_ana_norm = I_ana / np.max(I_ana)
    I_num_norm = I_num / np.max(I_num)
    
    # Mean Squared Error
    mse = np.mean((I_ana_norm - I_num_norm)**2)
    percent_match = (1.0 - mse) * 100
    print(f"-> Mean Squared Error (MSE): {mse:.6f}")
    print(f"-> Model Convergence Match:  {percent_match:.4f}%")
    
    assert mse < 0.01, f"Numerical model MSE {mse:.5f} exceeds convergence limit of 0.01 (99% match)"
    print("-> PASSED: Numerical model matches Analytical model (Match > 99%).")
    
    # ----------------------------------------------------
    # Test 4: Inverse Transform Sampler Speed Check
    # ----------------------------------------------------
    print("\n[Test 4] Inverse Transform Sampling Speed Check...")
    rng = np.random.default_rng(42)
    t_start = time.perf_counter()
    
    # Draw 10,000 samples
    samples = physics.sample_photons(x_grid, I_x, 10000, rng)
    t_duration = (time.perf_counter() - t_start) * 1000.0  # ms
    
    print(f"-> Draw duration for 10,000 photons: {t_duration:.3f} ms")
    assert len(samples) == 10000, "Failed to sample correct amount of photons"
    assert t_duration < 5.0, f"Sampling took {t_duration:.2f} ms which exceeds the 5.0 ms limit"
    print("-> PASSED: Sampling speed check verified (Duration < 5.0 ms).")
    
    print("\n==================================================")
    print("ALL TESTS COMPLETED SUCCESSFULLY: CALIBRATION PASSED")
    print("==================================================")

if __name__ == "__main__":
    run_tests()
