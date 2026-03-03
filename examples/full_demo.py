#!/usr/bin/env python3
"""Full demonstration of pydmdeeg including all plotting functions.

Run this script to test all features and generate plots.

Usage:
    python examples/full_demo.py

This will create a 'plots' directory with all generated figures.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from pydmdeeg import DMD


def generate_synthetic_eeg(
    n_trials: int = 10,
    n_channels: int = 8,
    n_times: int = 1000,
    sampling_rate: int = 256,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate synthetic EEG data with realistic oscillations."""
    np.random.seed(42)
    t = np.linspace(0, n_times / sampling_rate, n_times)

    X = np.zeros((n_trials, n_channels, n_times))

    for trial in range(n_trials):
        is_task = trial >= n_trials // 2

        for ch in range(n_channels):
            # Alpha (10 Hz) - stronger in rest
            alpha_amp = 1.0 if not is_task else 0.5
            alpha = alpha_amp * np.sin(2 * np.pi * 10 * t + np.random.rand() * 2 * np.pi)

            # Beta (20 Hz) - stronger in task
            beta_amp = 0.5 if not is_task else 1.0
            beta = beta_amp * np.sin(2 * np.pi * 20 * t + np.random.rand() * 2 * np.pi)

            # Spatial weighting
            spatial_weight = 0.5 + 0.5 * np.sin(ch * np.pi / n_channels)

            X[trial, ch, :] = spatial_weight * (alpha + beta) + 0.3 * np.random.randn(n_times)

    y = np.array([0] * (n_trials // 2) + [1] * (n_trials - n_trials // 2))
    channels = [f"Ch{i + 1}" for i in range(n_channels)]

    return X, y, channels


def main():
    # Create output directory for plots
    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)

    print("=" * 60)
    print("pydmdeeg Full Demo - Testing All Features")
    print("=" * 60)

    # 1. Generate data
    print("\n[1/7] Generating synthetic EEG data...")
    X, y, channels = generate_synthetic_eeg(
        n_trials=10,
        n_channels=8,
        n_times=1000,
        sampling_rate=256,
    )
    print(f"      Data shape: {X.shape}")
    print(f"      Labels: {y}")

    # 2. Initialize DMD
    print("\n[2/7] Initializing DMD...")
    dmd = DMD(
        X=X,
        y=y,
        channels=channels,
        dt=1 / 256,
        win_size=128,
        overlap=64,
        datascale="norm",
        algorithm="exact",
        stacking_factor=1,
    )
    print(f"      Windows per trial: {dmd.info['numws_per_trial']}")

    # 3. Run decomposition
    print("\n[3/7] Running DMD decomposition...")
    dmd.DMD_win()
    print(f"      Total modes: {len(dmd.results)}")
    print(f"      Results columns: {list(dmd.results.columns)[:5]}...")

    # 4. Compute statistics
    print("\n[4/7] Computing mode statistics...")
    stats = dmd.mode_stats(fbands=[[8, 12], [13, 30]], labels=[0, 1])
    for band, df in stats.items():
        print(f"      {band} Hz: mean = {df.loc['mean'].mean():.4f}")

    # 5. Plot: Channel statistics heatmap
    print("\n[5/7] Plotting channel statistics...")
    fig1 = dmd.plot_statsCH()
    fig1.suptitle("Channel Statistics by Frequency Band", fontsize=14)
    fig1.tight_layout()
    fig1.savefig(f"{plot_dir}/01_channel_stats.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"      Saved: {plot_dir}/01_channel_stats.png")

    # 6. Plot: Channel amplitude and error
    print("\n[6/7] Plotting amplitude and reconstruction error...")
    fig2 = dmd.plot_ChAmpErr(labels=[0, 1])
    fig2.suptitle("Channel Amplitude and Reconstruction Error", fontsize=14)
    fig2.tight_layout()
    fig2.savefig(f"{plot_dir}/02_amp_error.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"      Saved: {plot_dir}/02_amp_error.png")

    # 7. Plot: DMD power spectrum
    print("\n[7/7] Plotting DMD spectra...")

    # Power spectrum (|Psi|^2)
    g1 = dmd.plot_frRPsi(labels=[0, 1])
    g1.figure.suptitle("DMD Power Spectrum by Condition", fontsize=14, y=1.02)
    g1.figure.savefig(f"{plot_dir}/03_power_spectrum.png", dpi=150, bbox_inches="tight")
    plt.close(g1.figure)
    print(f"      Saved: {plot_dir}/03_power_spectrum.png")

    # Lambda spectrum
    g2 = dmd.plot_frRLam(labels=[0, 1])
    g2.figure.suptitle("Eigenvalue Spectrum by Condition", fontsize=14, y=1.02)
    g2.figure.savefig(f"{plot_dir}/04_lambda_spectrum.png", dpi=150, bbox_inches="tight")
    plt.close(g2.figure)
    print(f"      Saved: {plot_dir}/04_lambda_spectrum.png")

    # Additional analysis
    print("\n" + "=" * 60)
    print("Additional Analysis")
    print("=" * 60)

    # Extract data for specific bands
    print("\nAlpha band (8-12 Hz) comparison:")
    alpha_rest = dmd.get_PSI(fband=[8, 12], labels=[0])
    alpha_task = dmd.get_PSI(fband=[8, 12], labels=[1])
    psi_cols = [c for c in alpha_rest.columns if c.startswith("PSI_")]
    print(f"  Rest: {len(alpha_rest)} modes, mean |Ψ|: {alpha_rest[psi_cols].abs().mean().mean():.4f}")
    print(f"  Task: {len(alpha_task)} modes, mean |Ψ|: {alpha_task[psi_cols].abs().mean().mean():.4f}")

    print("\nBeta band (13-30 Hz) comparison:")
    beta_rest = dmd.get_PSI(fband=[13, 30], labels=[0])
    beta_task = dmd.get_PSI(fband=[13, 30], labels=[1])
    print(f"  Rest: {len(beta_rest)} modes, mean |Ψ|: {beta_rest[psi_cols].abs().mean().mean():.4f}")
    print(f"  Task: {len(beta_task)} modes, mean |Ψ|: {beta_task[psi_cols].abs().mean().mean():.4f}")

    # Trial selection test
    print("\nTesting trial selection...")
    dmd_subset = dmd.select_trials([0, 1, 2], return_copy=True)
    print(f"  Original trials: {dmd.results['trial'].nunique()}")
    print(f"  Subset trials: {dmd_subset.results['trial'].nunique()}")

    # Reconstruction error summary
    print("\nReconstruction quality:")
    print(f"  Mean Frobenius error: {dmd.AmpCh_Err['FroErW'].mean():.4f}")
    print(f"  Std Frobenius error: {dmd.AmpCh_Err['FroErW'].std():.4f}")

    print("\n" + "=" * 60)
    print(f"All plots saved to '{plot_dir}/' directory")
    print("=" * 60)


if __name__ == "__main__":
    main()
