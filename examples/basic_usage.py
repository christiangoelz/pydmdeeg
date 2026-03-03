#!/usr/bin/env python3
"""Basic usage example for pydmdeeg.

This script demonstrates how to use the DMD class to decompose
synthetic EEG-like data into spatio-temporal coherent patterns.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt

from pydmdeeg import DMD


def generate_synthetic_eeg(
    n_trials: int = 10,
    n_channels: int = 8,
    n_times: int = 1000,
    sampling_rate: int = 256,
    alpha_freq: float = 10.0,
    beta_freq: float = 20.0,
    noise_level: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate synthetic EEG-like data with alpha and beta oscillations.

    Parameters
    ----------
    n_trials : int
        Number of trials.
    n_channels : int
        Number of channels.
    n_times : int
        Number of time points per trial.
    sampling_rate : int
        Sampling rate in Hz.
    alpha_freq : float
        Alpha band frequency in Hz.
    beta_freq : float
        Beta band frequency in Hz.
    noise_level : float
        Standard deviation of Gaussian noise.

    Returns
    -------
    X : ndarray
        EEG data of shape (n_trials, n_channels, n_times).
    y : ndarray
        Trial labels.
    channels : list
        Channel names.
    """
    np.random.seed(42)
    t = np.linspace(0, n_times / sampling_rate, n_times)

    X = np.zeros((n_trials, n_channels, n_times))

    for trial in range(n_trials):
        # Different conditions: rest (0) vs task (1)
        is_task = trial >= n_trials // 2

        for ch in range(n_channels):
            # Alpha oscillation (stronger in rest)
            alpha_amp = 1.0 if not is_task else 0.5
            alpha = alpha_amp * np.sin(2 * np.pi * alpha_freq * t + np.random.rand() * 2 * np.pi)

            # Beta oscillation (stronger in task)
            beta_amp = 0.5 if not is_task else 1.0
            beta = beta_amp * np.sin(2 * np.pi * beta_freq * t + np.random.rand() * 2 * np.pi)

            # Add channel-specific spatial weighting
            spatial_weight = 0.5 + 0.5 * np.sin(ch * np.pi / n_channels)

            # Combine components
            X[trial, ch, :] = spatial_weight * (alpha + beta) + noise_level * np.random.randn(n_times)

    # Labels: first half rest (0), second half task (1)
    y = np.array([0] * (n_trials // 2) + [1] * (n_trials - n_trials // 2))

    # Channel names
    channels = [f"Ch{i + 1}" for i in range(n_channels)]

    return X, y, channels


def main():
    """Main function demonstrating DMD analysis."""
    print("=" * 60)
    print("pydmdeeg - Dynamic Mode Decomposition for EEG Analysis")
    print("=" * 60)

    # Generate synthetic data
    print("\n1. Generating synthetic EEG data...")
    n_trials, n_channels, n_times = 10, 8, 1000
    sampling_rate = 256

    X, y, channels = generate_synthetic_eeg(
        n_trials=n_trials,
        n_channels=n_channels,
        n_times=n_times,
        sampling_rate=sampling_rate,
    )

    print(f"   - Trials: {n_trials} ({sum(y == 0)} rest, {sum(y == 1)} task)")
    print(f"   - Channels: {n_channels}")
    print(f"   - Time points: {n_times}")
    print(f"   - Sampling rate: {sampling_rate} Hz")
    print(f"   - Duration: {n_times / sampling_rate:.2f} seconds")

    # Initialize DMD
    print("\n2. Initializing DMD...")
    dmd = DMD(
        X=X,
        y=y,
        channels=channels,
        dt=1 / sampling_rate,
        win_size=128,
        overlap=64,
        datascale="norm",
        algorithm="exact",
        stacking_factor=1,
    )

    print(f"   - Window size: {dmd.info['win_size']} samples ({dmd.info['win_size'] / sampling_rate * 1000:.0f} ms)")
    print(f"   - Overlap: {dmd.info['overlap']} samples")
    print(f"   - Windows per trial: {dmd.info['numws_per_trial']}")
    print(f"   - Algorithm: {dmd.info['algorithm']}")
    print(f"   - Data scaling: {dmd.info['datascale']}")

    # Run DMD decomposition
    print("\n3. Running DMD decomposition...")
    dmd.DMD_win()

    print(f"\n   - Total modes extracted: {len(dmd.results)}")
    print(f"   - Frequency range: {dmd.results['Mu'].min():.2f} to {dmd.results['Mu'].max():.2f} Hz")

    # Compute mode statistics for frequency bands
    print("\n4. Computing mode statistics...")
    frequency_bands = [
        [8, 12],  # Alpha band
        [13, 30],  # Beta band
    ]
    stats = dmd.mode_stats(fbands=frequency_bands)

    for band_name, band_stats in stats.items():
        print(f"\n   {band_name} Hz band:")
        mean_power = band_stats.loc["mean"].mean()
        std_power = band_stats.loc["std"].mean()
        print(f"   - Mean power: {mean_power:.4f} ± {std_power:.4f}")

    # Extract mode data for specific bands
    print("\n5. Extracting alpha band modes...")
    alpha_psi = dmd.get_PSI(fband=[8, 12])
    print(f"   - Alpha modes found: {len(alpha_psi)}")

    # Condition comparison
    print("\n6. Comparing conditions...")
    alpha_rest = dmd.get_PSI(fband=[8, 12], labels=[0])
    alpha_task = dmd.get_PSI(fband=[8, 12], labels=[1])
    beta_rest = dmd.get_PSI(fband=[13, 30], labels=[0])
    beta_task = dmd.get_PSI(fband=[13, 30], labels=[1])

    psi_cols = [c for c in alpha_rest.columns if c.startswith("PSI_")]

    print(f"\n   Alpha band (8-12 Hz):")
    print(f"   - Rest: {len(alpha_rest)} modes, mean |Ψ|: {alpha_rest[psi_cols].abs().mean().mean():.4f}")
    print(f"   - Task: {len(alpha_task)} modes, mean |Ψ|: {alpha_task[psi_cols].abs().mean().mean():.4f}")

    print(f"\n   Beta band (13-30 Hz):")
    print(f"   - Rest: {len(beta_rest)} modes, mean |Ψ|: {beta_rest[psi_cols].abs().mean().mean():.4f}")
    print(f"   - Task: {len(beta_task)} modes, mean |Ψ|: {beta_task[psi_cols].abs().mean().mean():.4f}")

    # Reconstruction error
    print("\n7. Reconstruction quality...")
    mean_error = dmd.AmpCh_Err["FroErW"].mean()
    std_error = dmd.AmpCh_Err["FroErW"].std()
    print(f"   - Mean Frobenius error: {mean_error:.4f} ± {std_error:.4f}")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
