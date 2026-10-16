"""Random trig reconstruction from trig poly.txt; not a forecasting model."""
import numpy as np

def random_trig_poly_from_thm115(f, k, rng):
    """
    f: array length N, viewed as f: Z_N -> C
    k: number of sampled atoms
    rng: np.random.Generator
    Returns: (freqs, coeffs, FR)
    """
    f = np.asarray(f, dtype=np.complex128)
    N = f.size

    fhat = np.fft.fft(f, norm="ortho")      # unitary DFT
    w = np.abs(fhat)
    S = w.sum()                              # ||fhat||_1
    l2 = np.linalg.norm(fhat)                # ||fhat||_2
    FR = float(S / l2) if l2 != 0 else 0.0   # Fourier ratio

    if S == 0:
        return np.array([], dtype=int), np.array([], dtype=np.complex128), FR

    p = w / S
    ms = rng.choice(N, size=k, replace=True, p=p)

    # Each term contributes (S/k) * sgn(fhat[m]) * N^{-1/2} * chi(mx)
    scale = (S / k) * (N ** (-0.5))

    coeff_map = {}
    for m in ms:
        z = fhat[m]
        if z == 0:
            continue
        phase = z / abs(z)                   # sgn(fhat[m])
        coeff_map[m] = coeff_map.get(m, 0.0) + scale * phase

    freqs = np.fromiter(coeff_map.keys(), dtype=int, count=len(coeff_map))
    coeffs = np.array([coeff_map[m] for m in freqs], dtype=np.complex128)
    return freqs, coeffs, FR

def eval_trig_poly(freqs, coeffs, N):
    x = np.arange(N)
    P = np.zeros(N, dtype=np.complex128)
    for m, a in zip(freqs, coeffs):
        P += a * np.exp(2j * np.pi * m * x / N)   # chi(mx)
    return P

def choose_k_from_eta(f, eta):
    """
    Uses the theorem’s scaling: k ≳ (FR(f)^2 - 1)/eta^2
    """
    f = np.asarray(f, dtype=np.complex128)
    fhat = np.fft.fft(f, norm="ortho")
    l2 = np.linalg.norm(fhat)
    FR = (np.abs(fhat).sum() / l2) if l2 != 0 else 0.0
    k =  int(np.ceil(max(0.0, FR**2 - 1.0) / (eta**2)))
    return max(k, 1), float(FR)
