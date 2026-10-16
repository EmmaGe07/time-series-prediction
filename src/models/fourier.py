"""Fourier-ratio diagnostic from inflation trig.ipynb; forecasting is not implemented."""
import numpy as np


def fourier_ratio(values):
    spectrum = np.fft.fft(np.asarray(values, dtype=float))
    norm = np.linalg.norm(spectrum)
    return float(np.abs(spectrum).sum() / norm) if norm else 0.0
