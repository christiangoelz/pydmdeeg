# pydmdeeg

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**Dynamic Mode Decomposition for EEG Signal Analysis**

A Python library for extracting spatio-temporal coherent patterns from EEG data using Dynamic Mode Decomposition (DMD). This method enables data-driven decomposition of neural recordings into interpretable oscillatory modes.

## Features

- **Two DMD algorithms**: Exact and standard implementations based on Tu et al. (2014)
- **Sliding window analysis**: Configurable window size and overlap for time-resolved decomposition
- **Delay embedding**: Shift-stacking technique for enhanced temporal dynamics capture
- **SVD truncation**: Optional hard thresholding (optht) or fixed-rank truncation for noise reduction
- **Data scaling**: Multiple preprocessing options (centering, normalization, z-scoring)
- **Rich visualization**: Built-in plotting for DMD spectra, channel statistics, and reconstruction errors
- **Multi-trial support**: Handle epoched EEG data with trial labels for condition comparisons

## Installation

### From PyPI (recommended)

```bash
pip install pydmdeeg
```

### From source

```bash
git clone https://github.com/christiangoelz/pydmdeeg.git
cd pydmdeeg
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from pydmdeeg import DMD

# Generate synthetic EEG-like data
# Shape: (n_trials, n_channels, n_timepoints)
n_trials, n_channels, n_times = 10, 64, 1000
sampling_rate = 256  # Hz

# Create data with oscillatory components
t = np.linspace(0, n_times/sampling_rate, n_times)
X = np.zeros((n_trials, n_channels, n_times))
for trial in range(n_trials):
    for ch in range(n_channels):
        # Alpha (10 Hz) and beta (20 Hz) oscillations with noise
        X[trial, ch, :] = (
            np.sin(2 * np.pi * 10 * t) +
            0.5 * np.sin(2 * np.pi * 20 * t) +
            0.3 * np.random.randn(n_times)
        )

# Trial labels (e.g., 0 = rest, 1 = task)
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# Channel names
channels = [f'Ch{i+1}' for i in range(n_channels)]

# Initialize DMD
dmd = DMD(
    X=X,
    y=y,
    channels=channels,
    dt=1/sampling_rate,
    win_size=128,
    overlap=64,
    algorithm='exact',
    datascale='norm'
)

# Run decomposition
dmd.DMD_win()

# Extract alpha band (8-12 Hz) mode statistics
alpha_stats = dmd.mode_stats(fbands=[[8, 12]], labels=[0, 1])
print(alpha_stats)

# Get mode magnitudes for specific frequency band
psi = dmd.get_PSI(fband=[8, 12], labels=[1])
```

## API Reference

### DMD Class

```python
DMD(
    X,                    # EEG data: (n_trials, n_channels, n_times) or (n_channels, n_times)
    y,                    # Trial labels: (n_trials,)
    channels,             # Channel names: list of strings
    dt,                   # Sampling interval: 1/sampling_rate
    stacking_factor=0,    # Delay embedding factor (0 = no stacking)
    win_size=100,         # Window size in samples
    overlap=0,            # Window overlap in samples
    datascale='norm',     # Scaling: 'none', 'centre', 'norm', 'centre_norm'
    algorithm='exact',    # DMD algorithm: 'exact' or 'standard'
    truncation={'method': None, 'keep': None}  # SVD truncation settings
)
```

### Key Methods

| Method | Description |
|--------|-------------|
| `DMD_win()` | Run DMD decomposition across sliding windows |
| `mode_stats(fbands, labels)` | Compute statistics for modes in frequency bands |
| `get_PSI(fband, labels)` | Extract mode magnitudes for frequency band |
| `get_AMP(fband, labels)` | Extract mode amplitudes for frequency band |
| `select_trials(selector)` | Select subset of trials |
| `plot_frRPsi(labels)` | Plot DMD power spectrum |
| `plot_frRLam(labels)` | Plot eigenvalue spectrum |
| `plot_statsCH()` | Plot channel statistics heatmap |
| `plot_ChAmpErr(labels)` | Plot reconstruction error |

## Mathematical Background

Dynamic Mode Decomposition approximates the linear operator **A** that best maps consecutive time snapshots:

**X'** ≈ **AX**

where **X** and **X'** are time-shifted data matrices. The eigenvalues of **A** (λ) encode oscillation frequencies and growth/decay rates, while eigenvectors (Φ) represent spatial patterns.

For EEG analysis:
- **Frequencies**: μ = Im(log(λ)) / (2πΔt)
- **Spatial modes**: Φ shows channel contributions to each oscillatory pattern
- **Amplitudes**: |Φ| weighted by initial conditions

## Publications Using This Method

This implementation has been validated in peer-reviewed neuroscience research:

1. Goelz C et al. (2021). Classification of visuomotor tasks based on electroencephalographic data depends on age-related differences in brain activity patterns. *Neural Networks*. [DOI](https://doi.org/10.1016/j.neunet.2021.04.029)

2. Goelz C et al. (2021). Electrophysiological signatures of dedifferentiation differ between fit and less fit older adults. *Cognitive Neurodynamics*. [DOI](https://doi.org/10.1007/s11571-020-09656-9)

3. Goelz C et al. (2018). Improved Neural Control of Movements Manifests in Expertise-Related Differences in Force Output and Brain Network Dynamics. *Frontiers in Physiology*. [DOI](https://doi.org/10.3389/fphys.2018.01540)

4. Vieluf S et al. (2018). Age- and Expertise-Related Differences of Sensorimotor Network Dynamics during Force Control. *Neuroscience*. [DOI](https://doi.org/10.1016/j.neuroscience.2018.07.025)

## References

- Brunton BW et al. (2016). Extracting spatial-temporal coherent patterns in large-scale neural recordings using dynamic mode decomposition. *J Neurosci Methods*. [DOI](https://doi.org/10.1016/j.jneumeth.2015.10.010)
- Tu JH et al. (2014). On dynamic mode decomposition: Theory and applications. *J Computational Dynamics*. [DOI](https://doi.org/10.3934/jcd.2014.1.391)
- Donoho D & Gavish M (2014). The Optimal Hard Threshold for Singular Values is 4/√3. *IEEE Trans Information Theory*. [DOI](https://doi.org/10.1109/TIT.2014.2323359)

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check .

# Run type checking
mypy pydmdeeg
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{pydmdeeg,
  author = {Goelz, Christian},
  title = {pydmdeeg: Dynamic Mode Decomposition for EEG Analysis},
  url = {https://github.com/christiangoelz/pydmdeeg},
  version = {0.2.0},
  year = {2024}
}
```
