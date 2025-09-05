import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import re

from scipy.interpolate import CubicSpline

# Import your ProtonBeam model
from src.proton_physics import ProtonBeam

def compare_bragg_peak(file_path, energy_mev, x_extend_factor=1., num_points=1000, spline_points=1000):
    """
    Compare experimental Bragg peak with a simulated curve and spline-fit experimental data.
    
    Parameters:
    - file_path: str, path to CSV or TXT (two columns: depth [mm], dose)
    - energy_mev: float, proton beam energy
    - x_extend_factor: float, factor to extend simulation x-axis beyond experimental max depth
    - num_points: int, number of points in simulation
    - spline_points: int, number of points for the spline interpolation
    """
    # --- Load experimental data ---
    df = pd.read_csv(file_path, delim_whitespace=True, header=None) if file_path.endswith(".txt") \
         else pd.read_csv(file_path)
    
    depth_mm = df.iloc[:, 0].values
    dose_exp = df.iloc[:, 1].values
    depth_cm = depth_mm / 10.0
    
    # Normalize experimental dose
    dose_exp_norm = dose_exp / np.max(dose_exp)
    
    # --- Fit spline to experimental data ---
    spline = CubicSpline(depth_cm, dose_exp_norm)
    depth_spline_cm = np.linspace(np.min(depth_cm), np.max(depth_cm), spline_points)
    dose_spline = spline(depth_spline_cm)
    
    # --- Simulated Bragg peak on extended x-axis ---
    beam = ProtonBeam(medium="water")
    max_depth_cm = np.max(depth_cm) * x_extend_factor
    depth_sim_cm = np.linspace(0, max_depth_cm, num_points)
    sim_dose = beam.calculate_bragg_curve(energy_mev, depth_sim_cm)
    
    # --- Plot ---
    plt.figure(figsize=(8,5))
    plt.plot(depth_cm*10, dose_exp_norm, "o", label="Experimental (normalized)")
    plt.plot(depth_spline_cm*10, dose_spline, "-", lw=2, label="Spline fit")
    plt.plot(depth_sim_cm*10, sim_dose, "--", lw=2, label=f"Simulation ({energy_mev} MeV)")
    plt.xlabel("Depth in water (mm)")
    plt.ylabel("Relative Dose")
    plt.title("Bragg Peak: Experiment vs Simulation")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def compare_bragg_peak(file_path, energy_mev, x_extend_factor=1., num_points=1000, spline_points=1000):
    """
    Compare experimental Bragg peak with a simulated curve and spline-fit experimental data.
    Adds vertical lines at Bragg peak positions.
    """
    # --- Load experimental data ---
    df = pd.read_csv(file_path, delim_whitespace=True, header=None) if file_path.endswith(".txt") \
         else pd.read_csv(file_path)
    
    depth_mm = df.iloc[:, 0].values
    dose_exp = df.iloc[:, 1].values
    depth_cm = depth_mm / 10.0
    
    # Normalize experimental dose
    dose_exp_norm = dose_exp / np.max(dose_exp)
    
    # --- Fit spline to experimental data ---
    spline = CubicSpline(depth_cm, dose_exp_norm)
    depth_spline_cm = np.linspace(np.min(depth_cm), np.max(depth_cm), spline_points)
    dose_spline = spline(depth_spline_cm)
    
    # --- Simulated Bragg peak on extended x-axis ---
    beam = ProtonBeam(medium="water")
    max_depth_cm = np.max(depth_cm) * x_extend_factor
    depth_sim_cm = np.linspace(0, max_depth_cm, num_points)
    sim_dose = beam.calculate_bragg_curve(energy_mev, depth_sim_cm)
    
    # --- Locate Bragg peaks ---
    peak_depth_spline_cm = depth_spline_cm[np.argmax(dose_spline)]
    peak_depth_sim_cm = depth_sim_cm[np.argmax(sim_dose)]
    
    # --- Plot ---
    plt.figure(figsize=(8,5))
    plt.plot(depth_cm*10, dose_exp_norm, "o", label="Experimentale (normalize)")
    plt.plot(depth_spline_cm*10, dose_spline, "-", lw=2, label="Spline fit")
    plt.plot(depth_sim_cm*10, sim_dose, "--", lw=2, label=f"Simulation ({energy_mev} MeV)")
    
    # Vertical lines for Bragg peaks
    plt.axvline(peak_depth_spline_cm*10, color="blue", linestyle=":", lw=1.5, 
                label=f"Exp. pic = {peak_depth_spline_cm*10:.1f} mm")
    plt.axvline(peak_depth_sim_cm*10, color="red", linestyle=":", lw=1.5, 
                label=f"Sim. pic = {peak_depth_sim_cm*10:.1f} mm")
    
    plt.xlabel("Profondeur dans l'eau(mm)")
    plt.ylabel("Dose Relative (a.u.)")
    plt.title("Pic du Bragg: Experience vs Simulation")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_multiple_bragg_peaks(file_list, spline_points=1000, labels=None, norm = 'max'):
    """
    Plot experimental Bragg peaks from several files on the same plot (no simulation).
    
    Parameters:
    - file_list: list of file paths (CSV or TXT, two columns: depth [mm], dose)
    - spline_points: int, number of points for the spline interpolation
    - labels: list of labels for each file (optional)
    """
    plt.figure(figsize=(10, 6))
    def extract_label(file_path):
        base = os.path.basename(file_path)
        match = re.search(r"(.*?mev)", base, re.IGNORECASE)
        if match:
            return match.group(1)
        return base  

    plt.figure(figsize=(10, 6))
    if labels is None:
        labels = [extract_label(f) for f in file_list]
    for idx, file_path in enumerate(file_list):
        df = pd.read_csv(file_path, delim_whitespace=True, header=None) \
            if file_path.endswith(".txt") else pd.read_csv(file_path)
        depth_mm = df.iloc[:, 0].values
        dose_exp = df.iloc[:, 1].values
        depth_cm = depth_mm / 10.0
        if norm == 'max':
            dose_exp_norm = dose_exp / np.max(dose_exp)
        elif norm == 'first':
            dose_exp_norm = dose_exp / dose_exp[0]
        spline = CubicSpline(depth_cm, dose_exp_norm)
        depth_spline_cm = np.linspace(np.min(depth_cm), np.max(depth_cm), spline_points)
        dose_spline = spline(depth_spline_cm)
        # --- Locate Bragg peaks ---
        peak_depth_spline_cm = depth_spline_cm[np.argmax(dose_spline)]
        plt.plot(depth_cm*10, dose_exp_norm, "o", alpha=0.5)
        plt.plot(depth_spline_cm*10, dose_spline, "-", lw=2, label=f"{labels[idx]}, pic at {peak_depth_spline_cm*10:.3f} mm")
        plt.axvline(peak_depth_spline_cm*10, color="red", linestyle=":", lw=1.5)
    plt.xlabel("Profondeur dans l'eau (mm)")
    plt.ylabel("Dose Relative (a.u.)")
    plt.title("Bragg Peaks: Experimental Data")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


