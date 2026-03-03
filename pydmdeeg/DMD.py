#!/usr/bin/env python3
"""Dynamic Mode Decomposition for EEG signal analysis.

This module provides the DMD class for decomposing EEG data into
spatio-temporal coherent patterns using Dynamic Mode Decomposition.

Author: Christian Goelz
"""

from __future__ import annotations

import copy as cp
import sys
import warnings
from math import floor
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.linalg import cond, eig, norm, pinv
from optht import optht
from scipy.linalg import svd

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.figure import Figure
    from numpy.typing import NDArray


class DMD:
    """EEG signal decomposition using Dynamic Mode Decomposition (DMD).

    This class decomposes EEG data into spatio-temporal coherent patterns.
    DMD approximates the linear operator that best maps consecutive time
    snapshots, extracting oscillatory modes with associated frequencies
    and spatial patterns.

    Parameters
    ----------
    X : ndarray
        EEG data with shape (n_epochs, n_channels, n_times) for epoched data
        or (n_channels, n_times) for continuous data.
    y : ndarray
        Trial labels with shape (n_epochs,). Required for epoched data.
    channels : list of str
        Channel names, must match n_channels dimension of X.
    dt : float
        Sampling interval (1/sampling_rate) in seconds.
    stacking_factor : int, default=0
        Stacking factor h for delay embedding. See Brunton et al. (2016).
        If 0, no stacking is applied.
    win_size : int, default=100
        Window size in samples for sliding window DMD.
    overlap : int, default=0
        Overlap between consecutive windows in samples.
    datascale : {'none', 'centre', 'centre_norm', 'norm'}, default='norm'
        Data scaling method:
        - 'none': No scaling
        - 'centre': Zero mean (subtract mean)
        - 'norm': Amplitude normalization (divide by std)
        - 'centre_norm': Z-score normalization (subtract mean, divide by std)
    algorithm : {'exact', 'standard'}, default='exact'
        DMD algorithm variant. See Tu et al. (2014).
    truncation : dict, optional
        SVD truncation settings:
        - {'method': 'optht'}: Optimal hard thresholding (Gavish & Donoho, 2014)
        - {'method': 'cut', 'keep': n}: Keep n SVD modes

    Attributes
    ----------
    X : ndarray
        The (possibly scaled) input data.
    y : ndarray
        Trial labels.
    dt : float
        Sampling interval.
    scaled : bool
        Whether data has been scaled.
    info : dict
        Configuration and metadata.
    results : DataFrame
        DMD results including modes, frequencies, and amplitudes.
    AmpCh_Err : DataFrame
        Channel amplitudes and reconstruction errors.
    Stats : dict
        Mode statistics (after calling mode_stats).

    References
    ----------
    .. [1] Brunton BW, Johnson LA, Ojemann JG, Kutz JN (2016).
           Extracting spatial-temporal coherent patterns in large-scale
           neural recordings using dynamic mode decomposition.
           J Neurosci Methods 258:1-15.
    .. [2] Tu JH et al. (2014). On dynamic mode decomposition: Theory
           and applications. J Computational Dynamics 1(2):391-421.
    .. [3] Gavish M, Donoho DL (2014). The optimal hard threshold for
           singular values is 4/sqrt(3). IEEE Trans Inf Theory 60(8):5040-5053.

    Examples
    --------
    >>> import numpy as np
    >>> from pydmdeeg import DMD
    >>> # Create synthetic data
    >>> X = np.random.randn(10, 64, 1000)
    >>> y = np.array([0]*5 + [1]*5)
    >>> channels = [f'Ch{i}' for i in range(64)]
    >>> dmd = DMD(X, y, channels, dt=1/256, win_size=128)
    >>> dmd.DMD_win()
    >>> stats = dmd.mode_stats([[8, 12]])
    """

    def __init__(
        self,
        X: NDArray[np.floating[Any]] | None = None,
        y: NDArray[np.integer[Any]] | None = None,
        channels: list[str] | None = None,
        dt: float | None = None,
        stacking_factor: int = 0,
        win_size: int = 100,
        overlap: int = 0,
        datascale: str = "norm",
        algorithm: str = "exact",
        truncation: dict[str, Any] | None = None,
    ) -> None:
        if truncation is None:
            truncation = {"method": None, "keep": None}

        # Validate inputs
        _check_option("datascale", datascale, ["none", "centre", "centre_norm", "norm"])
        _check_option("algorithm", algorithm, ["exact", "standard"])

        if X is None:
            warnings.warn("No data provided, specify data matrix", stacklevel=2)
        elif y is None and len(X.shape) == 3:
            raise ValueError("No trial labels provided for epoched data")
        elif channels is None:
            raise ValueError("No channel information provided")

        # Validate data dimensions
        if X is not None and channels is not None:
            n_chan = len(channels)
            chan_num = X.shape[1] if len(X.shape) == 3 else X.shape[0]
            if n_chan != chan_num:
                raise ValueError(
                    f"Number of channels doesn't match: got {chan_num} in data, "
                    f"but {n_chan} channel names provided"
                )
            trials = X.shape[0] if len(X.shape) == 3 else 1
            len_trial = int(X.size / n_chan / trials)
            numws = int(floor(len_trial - overlap) / (win_size - overlap))
        else:
            trials = 0
            len_trial = 0
            numws = 0
            n_chan = 0

        # Ensure stacking_factor is at least 1
        if stacking_factor < 1:
            stacking_factor = 1

        self.X = X
        self.y = y
        self.dt = dt
        self.scaled = False
        self.info: dict[str, Any] = {
            "trials": trials,
            "dt": dt,
            "trials_pts": len_trial,
            "channels": channels,
            "n_chan": n_chan,
            "numws_per_trial": numws,
            "stacking_factor": stacking_factor,
            "win_size": win_size,
            "overlap": overlap,
            "datascale": datascale,
            "algorithm": algorithm,
            "truncation": truncation,
        }
        self.results: pd.DataFrame | None = None
        self.AmpCh_Err: pd.DataFrame | None = None
        self.Stats: dict[str, pd.DataFrame] | None = None

    def DMD_win(self) -> DMD:
        """Run DMD decomposition in sliding windows.

        Computes DMD for each window across all trials, extracting
        modes, frequencies (Mu), eigenvalues (Lambda), and amplitudes.

        Returns
        -------
        self : DMD
            Returns self with results and AmpCh_Err attributes populated.
        """
        if not self.scaled:
            self._scale_input()

        X = self.X
        y = self.y
        ws = self.info["win_size"]
        ol = self.info["overlap"]
        trials = self.info["trials"]
        numws = self.info["numws_per_trial"]
        n_chan = self.info["n_chan"]
        channels = self.info["channels"]

        Window: list[NDArray[np.floating[Any]]] = []
        Trial: list[NDArray[np.floating[Any]]] = []
        Label: list[NDArray[np.floating[Any]]] = []
        Lambda: list[NDArray[np.complexfloating[Any, Any]]] = []
        Mu: list[NDArray[np.floating[Any]]] = []
        Psi: list[NDArray[np.complexfloating[Any, Any]]] = []
        Amp: list[NDArray[np.complexfloating[Any, Any]]] = []
        AppErr: list[NDArray[np.floating[Any]]] = []
        FroErW: list[float] = []
        AmpCh: list[NDArray[np.floating[Any]]] = []
        err_lab: list[Any] = []

        sys.stdout.write(
            f"Decomposing data in windows of {ws} samples with {ol} samples overlap:\n"
        )

        for triali in range(trials):
            if trials > 1:
                Xep = X[triali, :, :]
            else:
                Xep = X

            # Print progress
            t = (triali + 1) / trials
            sys.stdout.write("\r")
            sys.stdout.write(f"[{'=' * int(20 * t):<20s}] {100 * t:.0f}%")
            sys.stdout.flush()

            for win in range(numws):
                start = win * (ws - ol)
                end = (win + 1) * ws - win * ol
                XYwin = Xep[:, start:end]

                psi, lam, Xhat, Xaug, condXaug, z0, amp = _DMD_comp(
                    XYwin,
                    self.info["stacking_factor"],
                    self.info["algorithm"],
                    self.info["truncation"],
                )

                apperr = np.amax(
                    abs(XYwin[:, : (ws - ol)] - Xhat.real[:n_chan, : (ws - ol)]), axis=1
                )
                froerw = norm(
                    XYwin[:, : (ws - ol)] - Xhat.real[:n_chan, : (ws - ol)], "fro"
                ) / norm(XYwin[:, : (ws - ol)], "fro")
                ampch = np.amax(XYwin[:, : (ws - ol)], axis=1) - np.amin(
                    XYwin[:, : (ws - ol)], axis=1
                )

                AppErr.append(apperr)
                FroErW.append(froerw)
                AmpCh.append(ampch)
                err_lab.append(y[triali])

                nrModes = psi.shape[1]
                wind = np.repeat(win, nrModes).reshape(nrModes, 1).T
                trial = np.repeat(triali, nrModes).reshape(nrModes, 1).T
                lab = np.repeat(y[triali], nrModes).reshape(nrModes, 1).T
                lam = lam.reshape(1, len(lam))
                mu = ((np.log(lam) / self.info["dt"]).imag) / (2 * np.pi)

                Window.append(wind)
                Label.append(lab)
                Trial.append(trial)
                Lambda.append(lam)
                Mu.append(mu)
                Psi.append(psi[:n_chan, :])
                Amp.append(amp[:n_chan, :])

        # Create DataFrame with results
        Amp_arr = np.concatenate(Amp, axis=1).T
        Psi_arr = np.concatenate(Psi, axis=1).T
        Lambda_arr = np.concatenate(Lambda, axis=1).T
        Mu_arr = np.concatenate(Mu, axis=1).T
        Window_arr = np.concatenate(Window, axis=1).T
        Trial_arr = np.concatenate(Trial, axis=1).T
        Label_arr = np.concatenate(Label, axis=1).T

        cols = ["PSI_" + item for item in channels] + ["AMP_" + item for item in channels]
        data = np.c_[Psi_arr, Amp_arr]
        df = pd.DataFrame(columns=cols, data=data)
        df["Lambda"] = Lambda_arr
        df["Mu"] = Mu_arr
        df["win"] = Window_arr
        df["trial"] = Trial_arr
        df["label"] = Label_arr
        self.results = df

        # Create DataFrame with Amplitude + Error
        AppErr_arr = np.r_[AppErr]
        AmpCh_arr = np.r_[AmpCh]
        FroErW_arr = np.asarray(FroErW)

        cols2 = ["AmpCh_" + item for item in channels] + ["AppErr_" + item for item in channels]
        data2 = np.c_[AmpCh_arr, AppErr_arr]
        df2 = pd.DataFrame(columns=cols2, data=data2)
        df2["FroErW"] = FroErW_arr
        df2["label"] = err_lab
        self.AmpCh_Err = df2

        sys.stdout.write("\n")
        return self

    def mode_stats(
        self,
        fbands: Sequence[Sequence[float]],
        labels: list[int] | None = None,
        unit_length: bool = False,
        mode: str = "PSI",
    ) -> dict[str, pd.DataFrame]:
        """Calculate descriptive statistics of DMD modes.

        Parameters
        ----------
        fbands : sequence of [lower, upper] pairs
            Frequency bands of interest, e.g., [[8, 12], [13, 30]].
        labels : list of int, optional
            Trial labels to include. If None, uses all labels.
        unit_length : bool, default=False
            If True, normalize modes to unit length before computing stats.
        mode : {'PSI', 'AMP'}, default='PSI'
            Use mode magnitudes ('PSI') or amplitudes ('AMP').

        Returns
        -------
        Stats : dict
            Dictionary mapping frequency band strings to DataFrames
            with descriptive statistics.
        """
        Stats: dict[str, pd.DataFrame] = {}
        if labels is None:
            labels = list(set(self.y))

        for band in fbands:
            df = self._bands(band[0], band[1], labels)
            cols = [col for col in df.columns if mode in col]
            df = df[cols].abs()

            if unit_length:
                normfact = np.sqrt(np.square(df).sum(axis=1))
                df = df.divide(normfact, axis=0)

            Stats[f"{band[0]}-{band[1]}"] = df.describe()

        self.Stats = Stats
        return Stats

    # Plotting methods

    def plot_statsCH(self) -> Figure:
        """Plot channel statistics heatmaps.

        Requires mode_stats() to have been called first.

        Returns
        -------
        fig : Figure
            Matplotlib figure with mean, median, Q1, Q3 heatmaps.
        """
        stats = self.Stats
        channels = self.info["channels"]

        mean_band = []
        median_band = []
        q1 = []
        q2 = []
        fbands = list(stats.keys())

        for band in stats.keys():
            mean_band.append(stats[band].loc["mean"].values)
            median_band.append(stats[band].loc["50%"].values)
            q1.append(stats[band].loc["25%"].values)
            q2.append(stats[band].loc["75%"].values)

        mean = np.c_[mean_band]
        median = np.c_[median_band]
        q1_arr = np.c_[q1]
        q2_arr = np.c_[q2]

        fig, ax = plt.subplots(2, 2, figsize=(25, 12))
        ax1 = sns.heatmap(mean, ax=ax[0, 0], cmap="YlOrBr")
        ax2 = sns.heatmap(median, ax=ax[0, 1], cmap="YlOrBr")
        ax3 = sns.heatmap(q1_arr, ax=ax[1, 0], cmap="YlOrBr")
        ax4 = sns.heatmap(q2_arr, ax=ax[1, 1], cmap="YlOrBr")

        ax1.set_xticklabels(channels, rotation=90, size=6)
        ax2.set_xticklabels(channels, rotation=90, size=6)
        ax3.set_xticklabels(channels, rotation=90, size=6)
        ax4.set_xlabel("Channels")
        ax4.set_xticklabels(channels, rotation=90, size=6)

        ax1.set_ylabel("Mean")
        ax1.set_yticklabels(fbands, size=6)
        ax2.set_ylabel("Median")
        ax2.set_yticklabels(fbands, size=6)
        ax3.set_ylabel("Q1")
        ax3.set_yticklabels(fbands, size=6)
        ax4.set_ylabel("Q2")
        ax4.set_yticklabels(fbands, size=6)

        return fig

    def plot_ChAmpErr(self, labels: list[int] | None = None) -> Figure:
        """Plot channel amplitude and reconstruction error.

        Parameters
        ----------
        labels : list of int, optional
            Trial labels to include. If None, uses all labels.

        Returns
        -------
        fig : Figure
            Matplotlib figure with amplitude and error plots.
        """
        if labels is None:
            labels = list(set(self.y))

        df = self.AmpCh_Err.copy()
        df = df[df["label"].isin(labels)]

        channels = list(self.info["channels"])
        channels.append("labels")
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(2, 2, figsize=(25, 12))

        # Subplot 1: AMPCH
        a = [col for col in df.columns if "AmpCh" in col]
        a.append("label")
        ampch = df[a]
        ampch.columns = channels
        ampch1 = pd.melt(ampch, "labels", var_name="chan")
        ax1 = sns.stripplot(
            y="value",
            x="chan",
            data=ampch1,
            dodge=True,
            alpha=0.25,
            zorder=1,
            color="g",
            ax=ax[0, 0],
        )
        ax1.set_ylabel("max(EEG amp.)/ window")
        ax1.set_xticklabels(channels[:-1], rotation=90, size=6)

        # Subplot 2: FroErW
        ax2 = sns.lineplot(x=df.index, y="FroErW", data=df, color="b", ax=ax[0, 1])
        mean = np.repeat(df["FroErW"].mean(), len(df))
        df["mean"] = mean
        sns.lineplot(x=df.index, y="mean", data=df, color="r", ax=ax[0, 1])
        ax2.set_xlabel("Time window")

        # Subplot 3: Ampch heatmap
        ampch2 = np.array(ampch.values)
        ax3 = sns.heatmap(ampch2[:, :-1], ax=ax[1, 0], cmap="YlOrBr")
        ax3.set_xticklabels(channels[:-1], rotation=90, size=6)
        ax3.set_ylabel("max(EEG amp.)/ window")

        # Subplot 4: Error heatmap
        b = [col for col in df.columns if "Err" in col]
        b.append("label")
        err = df[b]
        err.columns = channels
        err_arr = np.array(err.values)
        ax4 = sns.heatmap(err_arr[:, :-1], ax=ax[1, 1], cmap="YlOrBr")
        ax4.set_xticklabels(channels[:-1], rotation=90, size=6)
        ax4.set_ylabel("|X-X'|/ window")

        return fig

    def plot_frRPsi(self, labels: list[int] | None = None) -> sns.FacetGrid:
        """Plot DMD power spectrum.

        Parameters
        ----------
        labels : list of int, optional
            Trial labels to include. If None, uses all labels.

        Returns
        -------
        g : FacetGrid
            Seaborn FacetGrid with power spectrum plots.
        """
        if labels is None:
            labels = list(set(self.y))

        channels = list(self.info["channels"])
        channels.append("label")
        channels.append("Mu")

        df = self.results
        df = df[(df["Mu"] > 0)].abs()
        df = df[df["label"].isin(labels)]

        a = [col for col in df.columns if "PSI" in col]
        a.append("label")
        a.append("Mu")
        psi = df[a].abs()
        psi.columns = channels
        psi["power"] = np.square(psi.loc[:, channels[:-2]]).sum(axis=1)
        psi["zero"] = np.repeat(0, len(psi))

        g = sns.FacetGrid(psi, col="label", col_wrap=2, height=8, aspect=0.5)
        g.map(sns.scatterplot, "Mu", "power", alpha=0.5, color="k")
        g.map(plt.vlines, "Mu", "zero", "power", alpha=0.8, color="k")

        g.set_ylabels(r"|$\Phi$|$^2$")
        g.set_xlabels("Frequency (Hz)")
        g.set(xlim=(0, None), ylim=(0, None))
        return g

    def plot_frRLam(self, labels: list[int] | None = None) -> sns.FacetGrid:
        """Plot eigenvalue spectrum.

        Parameters
        ----------
        labels : list of int, optional
            Trial labels to include. If None, uses all labels.

        Returns
        -------
        g : FacetGrid
            Seaborn FacetGrid with eigenvalue plots.
        """
        df = self.results

        if labels is None:
            labels = list(set(self.y))

        df = df[df["label"].isin(labels)]
        df = df[(df["Mu"] > 0)].abs()
        df = df[["Mu", "Lambda", "label"]]

        g = sns.FacetGrid(df, col="label", height=8, aspect=0.5)
        g.map(sns.scatterplot, "Mu", "Lambda", alpha=0.5, color="g")

        g.set_ylabels(r"|$\lambda$|")
        g.set_xlabels("Frequency (Hz)")
        g.set(xlim=(0, None), ylim=(0, None))
        return g

    def get_PSI(
        self,
        fband: Sequence[float],
        labels: list[int] | None = None,
        unit_length: bool = False,
    ) -> pd.DataFrame:
        """Extract mode magnitudes for a frequency band.

        Parameters
        ----------
        fband : [lower, upper]
            Frequency band bounds in Hz.
        labels : list of int, optional
            Trial labels to include. If None, uses all labels.
        unit_length : bool, default=False
            If True, normalize modes to unit length.

        Returns
        -------
        PSI : DataFrame
            Mode magnitudes with Mu, win, trial, and label columns.
        """
        return self._get_data(fband, "PSI", unit_length, labels)

    def get_AMP(
        self,
        fband: Sequence[float],
        labels: list[int] | None = None,
        unit_length: bool = False,
    ) -> pd.DataFrame:
        """Extract mode amplitudes for a frequency band.

        Parameters
        ----------
        fband : [lower, upper]
            Frequency band bounds in Hz.
        labels : list of int, optional
            Trial labels to include. If None, uses all labels.
        unit_length : bool, default=False
            If True, normalize modes to unit length.

        Returns
        -------
        AMP : DataFrame
            Mode amplitudes with Mu, win, trial, and label columns.
        """
        return self._get_data(fband, "AMP", unit_length, labels)

    def select_trials(self, selector: list[int], return_copy: bool = True) -> DMD | pd.DataFrame:
        """Select subset of trials from results.

        Parameters
        ----------
        selector : list of int
            Trial indices to select.
        return_copy : bool, default=True
            If True, return a copy of DMD object with selected trials.
            If False, modify in place and return the filtered DataFrame.

        Returns
        -------
        DMD or DataFrame
            DMD object copy or filtered DataFrame depending on return_copy.
        """
        dmd_cp = cp.deepcopy(self)
        split = [dmd_cp.results[dmd_cp.results["trial"] == t] for t in selector]

        if not return_copy:
            self.results = pd.concat(split)
            return self.results
        else:
            dmd_cp.results = pd.concat(split)
            return dmd_cp

    # Private methods

    def _scale_input(self) -> DMD:
        """Scale input data according to datascale setting."""
        datascale = self.info["datascale"]
        n_channels = self.info["n_chan"]
        n_trials = self.info["trials"]
        l_trials = self.info["trials_pts"]
        X = self.X
        X = X.transpose(1, 0, 2).reshape(n_channels, -1)

        if datascale == "centre":
            X -= np.mean(X, axis=1).reshape(len(X), 1)

        elif datascale == "centre_norm":
            X -= np.mean(X, axis=1).reshape(len(X), 1)
            X /= np.std(X, axis=1, ddof=1).reshape(len(X), 1)

        elif datascale == "norm":
            X /= np.std(X, axis=1, ddof=1).reshape(len(X), 1)

        if len(self.X.shape) == 3:
            X = X.reshape(n_channels, n_trials, l_trials).transpose(1, 0, 2)

        self.X = X
        self.scaled = True
        return self

    def _bands(
        self,
        lower_bound: float,
        upper_bound: float,
        labels: list[int] | None = None,
    ) -> pd.DataFrame:
        """Filter results to modes within frequency band.

        Parameters
        ----------
        lower_bound : float
            Lower frequency bound (Hz). Must be > 0.
        upper_bound : float
            Upper frequency bound (Hz).
        labels : list of int, optional
            Trial labels to include.

        Returns
        -------
        df_bands : DataFrame
            Filtered results.
        """
        df = self.results

        if labels is None:
            labels = list(set(df.label))

        df = df[df["label"].isin(labels)]

        # Exclude 0 Hz modes
        if lower_bound == 0:
            lower_bound += 1e-8

        df_bands = df[(df["Mu"] >= lower_bound) & (df["Mu"] < upper_bound)]
        return df_bands

    def _get_data(
        self,
        fband: Sequence[float],
        mode: str,
        unit_length: bool,
        labels: list[int] | None = None,
    ) -> pd.DataFrame:
        """Extract mode data for frequency band.

        Parameters
        ----------
        fband : [lower, upper]
            Frequency band bounds.
        mode : {'AMP', 'PSI'}
            Mode type to extract.
        unit_length : bool
            Whether to normalize to unit length.
        labels : list of int, optional
            Trial labels to include.

        Returns
        -------
        data : DataFrame
            Extracted mode data.
        """
        df = self._bands(fband[0], fband[1], labels)
        cols = [col for col in df.columns if mode in col]
        data = cp.deepcopy(df[cols])

        if unit_length:
            normfact = np.sqrt(np.square(data).sum(axis=1))
            data = data.divide(normfact, axis=0)

        data["Mu"] = df.Mu
        data["win"] = df.win
        data["trial"] = df.trial
        data["label"] = df.label

        return data


# Module-level functions


def _X_aug_h(X: NDArray[np.floating[Any]], h: int) -> NDArray[np.floating[Any]]:
    """Shift-stack data matrix with stacking factor h.

    Parameters
    ----------
    X : ndarray
        Data matrix of shape (n_channels, n_times).
    h : int
        Stacking factor for delay embedding.

    Returns
    -------
    Xaug : ndarray
        Shift-stacked data matrix.
    """
    m = X.shape[1] + 1
    n = X.shape[0]
    Xaug = np.zeros((h * n, m - h))
    Xvec = np.concatenate(X.T, axis=0)

    for i in range(Xaug.shape[1]):
        Xaug[:, i] = Xvec[i * n : h * n + i * n]

    return Xaug


def _DMD_comp(
    XY: NDArray[np.floating[Any]],
    h: int,
    algorithm: str,
    truncation: dict[str, Any] | None = None,
) -> tuple[
    NDArray[np.complexfloating[Any, Any]],
    NDArray[np.complexfloating[Any, Any]],
    NDArray[np.complexfloating[Any, Any]],
    NDArray[np.floating[Any]],
    float,
    NDArray[np.complexfloating[Any, Any]],
    NDArray[np.complexfloating[Any, Any]],
]:
    r"""Compute DMD decomposition.

    Computes the DMD basis of a data matrix XY \in R^(n,m), where
    n is the dimension (channels) and m is the number of snapshots.

    Parameters
    ----------
    XY : ndarray
        Data matrix of shape (n_channels, n_times).
    h : int
        Stacking factor for delay embedding.
    algorithm : {'exact', 'standard'}
        DMD algorithm variant.
    truncation : dict, optional
        SVD truncation settings.

    Returns
    -------
    Psi : ndarray
        DMD modes.
    lam : ndarray
        DMD eigenvalues.
    Xhat : ndarray
        Reconstructed data.
    Xaug : ndarray
        Augmented data matrix.
    condXaug : float
        Condition number of Xaug.
    z0 : ndarray
        Initial mode amplitudes.
    Amp : ndarray
        Mode amplitudes.
    """
    if truncation is None:
        truncation = {"method": None, "keep": None}

    X = XY[:, :-1]
    Y = XY[:, 1:]

    Xaug = _X_aug_h(X, h)
    Yaug = _X_aug_h(Y, h)

    condXaug = cond(Xaug)

    U, S, Vt = svd(Xaug, full_matrices=False, lapack_driver="gesvd")
    V = Vt.T

    if truncation["method"] is not None:
        # Define how many modes to keep
        if truncation["method"] == "optht":
            r = optht(Xaug, sv=S, sigma=None)
        elif truncation["method"] == "cut":
            if truncation["keep"] is None:
                raise ValueError(
                    "Please specify how many SVD modes to keep when truncation is 'cut'"
                )
            r = truncation["keep"]
        else:
            r = len(S)

        # Keep r modes
        U = U[:, :r]
        S = S[:r]
        V = V[:, :r]

    # Compute DMD using matrix operations
    S1 = np.diag(S**-1)
    S0 = np.diag(S**-0.5)
    S2 = np.diag(S**0.5)

    A = S0 @ (U.T @ Yaug) @ (V @ S1) @ S2
    lam, W_hat = eig(A)
    W = S2 @ W_hat

    # Standard or exact DMD
    if algorithm == "standard":
        Psi = U @ W
    else:  # exact
        Psi = Yaug @ (V @ (S1 @ W))

    z0 = pinv(Psi) @ Xaug[:, 0]
    Xhat = np.zeros((Psi.shape[0], XY.shape[1]), dtype=np.complex128)
    Xhat[:, 0] = np.ravel(Xaug[:, 0])

    for i in range(1, XY.shape[1]):
        Xhat[:, i] = np.ravel(Psi @ np.diag(lam**i) @ z0)

    Amp = Psi @ np.diag(np.ravel(z0))

    return Psi, lam, Xhat, Xaug, condXaug, z0, Amp


def _check_option(
    parameter: str,
    value: Any,
    allowed_values: list[Any],
    extra: str = "",
) -> bool:
    """Check parameter value against allowed options.

    Parameters
    ----------
    parameter : str
        Parameter name for error message.
    value : any
        Value to check.
    allowed_values : list
        Allowed values.
    extra : str, optional
        Extra context for error message.

    Returns
    -------
    bool
        True if valid.

    Raises
    ------
    ValueError
        If value not in allowed_values.
    """
    if value in allowed_values:
        return True

    extra = " " + extra if extra else extra
    msg = (
        f"Invalid value for the '{parameter}' parameter{extra}. "
        "{options}, but got {value!r} instead."
    )

    if len(allowed_values) == 1:
        options = f"The only allowed value is {allowed_values[0]!r}"
    else:
        options = "Allowed values are "
        options += ", ".join([f"{v!r}" for v in allowed_values[:-1]])
        options += f" and {allowed_values[-1]!r}"

    raise ValueError(msg.format(options=options, value=value))
