"""
FES → Joint Torque Dataset Generator
======================================
Sweeps pulse width × frequency × subject, runs the Ding 2007 model for each
combination, converts muscle force to isometric knee torque, and saves the
results for training an inverse model (torque → FES parameters).

Outputs
-------
  fes_dataset.npz        — numpy archive (fast ML loading)
  fes_dataset.csv        — scalar summary per sample (human-readable)
  fes_dataset_plots/     — heatmaps and sample torque curves

Usage
-----
  python3 generate_fes_dataset.py

Modify the GRID and SETTINGS sections to change the parameter sweep.
"""

import os
import csv
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from ding_model import run_ding

# ── Experimental setup ────────────────────────────────────────────────────────
KNEE_ANGLE_DEG = -80   # isometric knee angle (degrees)

# Moment arms at KNEE_ANGLE_DEG from OpenSim (metres).
# Positive = extensor contribution.
# Re-run the lookup cell in the notes if you change the angle.
MOMENT_ARMS = {
    "rect_fem_r": 0.03385,
    "vas_int_r":  0.03468,
}
COMBINED_MOMENT_ARM = sum(MOMENT_ARMS.values())   # 0.06853 m

# ── Stimulation grid ──────────────────────────────────────────────────────────
# Frequency is fixed — only pulse width varies.
# Finer PW grid gives the inverse model more resolution to learn from.
PULSE_WIDTHS_US = list(range(50, 505, 1))   # 50, 60, 70 … 500 µs (46 values)
FREQUENCIES_HZ  = [33]                        # fixed at 33 Hz

# Subjects: 1–10 are real Ding 2007 subjects; 'average' is the population mean.
SUBJECTS = list(range(1, 11)) + ['average']

# ── Simulation settings ───────────────────────────────────────────────────────
SIM_DURATION_MS = 2000    # ms — how long each stimulation train runs
DT_MS           = 1.0     # ms — integration timestep (1 ms is accurate enough)
OUTPUT_DT_MS    = 10.0    # ms — output resolution (downsample for storage)

# ── Output paths ──────────────────────────────────────────────────────────────
OUT_DIR      = os.path.expanduser("~/fes_project/fes_dataset_33hz")
NPZ_PATH     = os.path.join(OUT_DIR, "fes_dataset.npz")
CSV_PATH     = os.path.join(OUT_DIR, "fes_dataset.csv")
PLOT_DIR     = os.path.join(OUT_DIR, "plots")


# ─────────────────────────────────────────────────────────────────────────────
def force_to_torque(force_n: np.ndarray) -> np.ndarray:
    """
    Convert Ding muscle force (N) to isometric knee torque (N·m).

    Assumes both FES-targeted muscles (rect_fem_r, vas_int_r) receive the
    same stimulation.  The moment arm is fixed at KNEE_ANGLE_DEG because
    this is an isometric (fixed-joint) setup.

    For a dynamic setup (joint free to move) the moment arm changes with
    angle and you would need the full OpenSim simulation instead.
    """
    return force_n * COMBINED_MOMENT_ARM


def compute_features(time_s: np.ndarray, torque: np.ndarray) -> dict:
    """
    Scalar summary features extracted from a torque curve.

    These are the candidate targets / inputs for the inverse model.
    Add more here if needed (e.g. time constants, fatigue slope).
    """
    peak_idx          = np.argmax(torque)
    peak_torque       = torque[peak_idx]
    time_to_peak_s    = time_s[peak_idx]
    mean_torque       = torque.mean()
    torque_rms        = np.sqrt((torque ** 2).mean())
    # Slope of torque rise: mean rate of change in first 25% of sim
    rise_end          = max(1, len(torque) // 4)
    rise_slope        = np.diff(torque[:rise_end]).mean() / (OUTPUT_DT_MS * 1e-3)
    return {
        "peak_torque_Nm":    round(float(peak_torque),    4),
        "time_to_peak_s":    round(float(time_to_peak_s), 4),
        "mean_torque_Nm":    round(float(mean_torque),    4),
        "torque_rms_Nm":     round(float(torque_rms),     4),
        "rise_slope_Nm_s":   round(float(rise_slope),     4),
    }


def downsample(time_s, values, out_dt_ms):
    """Resample to a coarser time grid by linear interpolation."""
    t_out = np.arange(0, time_s[-1] + 1e-9, out_dt_ms * 1e-3)
    v_out = np.interp(t_out, time_s, values)
    return t_out, v_out


def generate_dataset():
    os.makedirs(OUT_DIR,  exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    grid = list(itertools.product(PULSE_WIDTHS_US, FREQUENCIES_HZ, SUBJECTS))
    total = len(grid)
    print(f"Generating {total} samples  "
          f"({len(PULSE_WIDTHS_US)} PW × {len(FREQUENCIES_HZ)} freq × {len(SUBJECTS)} subjects)")
    print(f"Knee angle  : {KNEE_ANGLE_DEG}°  (isometric)")
    print(f"Moment arm  : {COMBINED_MOMENT_ARM:.5f} m  (rect_fem_r + vas_int_r)")
    print(f"Duration    : {SIM_DURATION_MS} ms  |  dt={DT_MS} ms  |  output dt={OUTPUT_DT_MS} ms\n")

    # Determine output time axis length from the first run
    t_s, excitation, force = run_ding(
        FREQUENCIES_HZ[0], PULSE_WIDTHS_US[0] * 1e-3,
        SIM_DURATION_MS, SUBJECTS[0], DT_MS
    )
    t_out, _ = downsample(t_s, force, OUTPUT_DT_MS)
    n_timesteps = len(t_out)

    # Storage arrays
    params_arr  = np.zeros((total, 3),           dtype=np.float32)  # pw, freq, subject_id
    torque_arr  = np.zeros((total, n_timesteps), dtype=np.float32)  # torque curve
    force_arr   = np.zeros((total, n_timesteps), dtype=np.float32)  # raw Ding force

    csv_rows = []
    csv_header = ["idx", "pw_us", "freq_hz", "subject"] + list(
        compute_features(t_out, np.zeros(n_timesteps)).keys()
    )

    for idx, (pw_us, freq_hz, subject) in enumerate(grid):
        subject_id = 0 if subject == 'average' else int(subject)

        t_s, _, force_n = run_ding(
            frequency_hz   = freq_hz,
            pulse_width_ms = pw_us * 1e-3,
            duration_ms    = SIM_DURATION_MS,
            subject        = subject,
            dt_ms          = DT_MS,
        )
        t_out, force_ds  = downsample(t_s, force_n, OUTPUT_DT_MS)
        torque           = force_to_torque(force_ds)

        params_arr[idx]  = [pw_us, freq_hz, subject_id]
        torque_arr[idx]  = torque.astype(np.float32)
        force_arr[idx]   = force_ds.astype(np.float32)

        features = compute_features(t_out, torque)
        csv_rows.append([idx, pw_us, freq_hz, subject] + list(features.values()))

        if (idx + 1) % 50 == 0 or idx == total - 1:
            print(f"  [{idx+1:3d}/{total}]  PW={pw_us}µs  f={freq_hz}Hz  "
                  f"subj={subject}  peak={features['peak_torque_Nm']:.3f} N·m")

    # Save .npz
    np.savez_compressed(
        NPZ_PATH,
        params      = params_arr,   # shape (N, 3): pw_us, freq_hz, subject_id
        torque      = torque_arr,   # shape (N, T): torque curve in N·m
        force       = force_arr,    # shape (N, T): raw Ding force in N
        time_s      = t_out,        # shape (T,):   shared time axis in seconds
        param_names = np.array(["pw_us", "freq_hz", "subject_id"]),
        knee_angle_deg         = np.float32(KNEE_ANGLE_DEG),
        combined_moment_arm_m  = np.float32(COMBINED_MOMENT_ARM),
    )
    print(f"\nSaved  → {NPZ_PATH}")

    # Save .csv
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        writer.writerows(csv_rows)
    print(f"Saved  → {CSV_PATH}")

    return t_out, params_arr, torque_arr, grid


def plot_heatmaps(params_arr, torque_arr, grid):
    """Peak torque heatmap (PW × frequency) for each subject."""
    print("\nGenerating heatmaps …")
    pw_vals   = sorted(set(PULSE_WIDTHS_US))
    freq_vals = sorted(set(FREQUENCIES_HZ))

    for subject in SUBJECTS:
        subject_id = 0 if subject == 'average' else int(subject)
        mask       = params_arr[:, 2] == subject_id
        sub_params = params_arr[mask]
        sub_torque = torque_arr[mask]

        grid_heat = np.zeros((len(freq_vals), len(pw_vals)))
        for i, (row_p, row_t) in enumerate(zip(sub_params, sub_torque)):
            pw_i   = pw_vals.index(int(row_p[0]))
            freq_i = freq_vals.index(float(row_p[1]))
            grid_heat[freq_i, pw_i] = row_t.max()

        fig, ax = plt.subplots(figsize=(9, 5))
        im = ax.imshow(grid_heat, aspect='auto', origin='lower',
                       cmap='hot', norm=mcolors.PowerNorm(gamma=0.5))
        ax.set_xticks(range(len(pw_vals)));   ax.set_xticklabels(pw_vals)
        ax.set_yticks(range(len(freq_vals))); ax.set_yticklabels(freq_vals)
        ax.set_xlabel("Pulse Width (µs)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(f"Peak Knee Torque (N·m)  —  Subject: {subject}")
        plt.colorbar(im, ax=ax, label="Peak torque (N·m)")
        plt.tight_layout()
        fname = os.path.join(PLOT_DIR, f"heatmap_subject_{subject}.png")
        plt.savefig(fname, dpi=110)
        plt.close()

    print(f"Heatmaps saved → {PLOT_DIR}/")


def plot_sample_curves(t_out, params_arr, torque_arr):
    """Torque curves for average subject, coloured by pulse width."""
    print("Generating sample torque curves …")
    subject_id = 0   # 'average'
    mask       = params_arr[:, 2] == subject_id
    sub_params = params_arr[mask]
    sub_torque = torque_arr[mask]

    # Sort by PW so the colour gradient is meaningful
    order      = np.argsort(sub_params[:, 0])
    sub_params = sub_params[order]
    sub_torque = sub_torque[order]

    cmap   = plt.cm.viridis
    colors = cmap(np.linspace(0, 1, len(sub_params)))

    fig, ax = plt.subplots(figsize=(10, 5))
    for color, params_row, torque_row in zip(colors, sub_params, sub_torque):
        ax.plot(t_out, torque_row, color=color, lw=1.2)

    sm = plt.cm.ScalarMappable(cmap=cmap,
         norm=plt.Normalize(sub_params[:, 0].min(), sub_params[:, 0].max()))
    plt.colorbar(sm, ax=ax, label="Pulse Width (µs)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Torque (N·m)")
    ax.set_title(f"Knee Torque Curves — Average Subject — {FREQUENCIES_HZ[0]} Hz fixed")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    fname = os.path.join(PLOT_DIR, "torque_curves_average.png")
    plt.savefig(fname, dpi=110)
    plt.close()
    print(f"Curves saved   → {fname}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t_out, params_arr, torque_arr, grid = generate_dataset()
    plot_heatmaps(params_arr, torque_arr, grid)
    plot_sample_curves(t_out, params_arr, torque_arr)

    # Quick load verification
    data = np.load(NPZ_PATH, allow_pickle=True)
    print(f"\nDataset summary:")
    print(f"  params shape  : {data['params'].shape}   (samples × [pw_us, freq_hz, subject_id])")
    print(f"  torque shape  : {data['torque'].shape}   (samples × timesteps)")
    print(f"  time axis     : {data['time_s'][0]:.3f}s → {data['time_s'][-1]:.3f}s  "
          f"({len(data['time_s'])} points @ {OUTPUT_DT_MS}ms)")
    print(f"  moment arm    : {float(data['combined_moment_arm_m']):.5f} m")
    print(f"  knee angle    : {float(data['knee_angle_deg']):.0f}°")
    print(f"\nLoading example:")
    print(f"  data = np.load('fes_dataset.npz', allow_pickle=True)")
    print(f"  X = data['params']   # inputs:  (pw_us, freq_hz, subject_id)")
    print(f"  y = data['torque']   # outputs: torque curve per sample")
