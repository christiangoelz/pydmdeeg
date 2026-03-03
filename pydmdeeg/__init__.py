"""pydmdeeg - Dynamic Mode Decomposition for EEG signal analysis.

This package provides tools for decomposing EEG data into spatio-temporal
coherent patterns using Dynamic Mode Decomposition (DMD).

Example
-------
>>> import numpy as np
>>> from pydmdeeg import DMD
>>> X = np.random.randn(10, 64, 1000)
>>> y = np.array([0]*5 + [1]*5)
>>> channels = [f'Ch{i}' for i in range(64)]
>>> dmd = DMD(X, y, channels, dt=1/256)
>>> dmd.DMD_win()
"""

__version__ = "0.2.0"
__author__ = "Christian Goelz"
__all__ = ["DMD"]

from .DMD import DMD
