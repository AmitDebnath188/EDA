# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import find_peaks

def baseline_als(eda_signal, lam=1e3, p=0.001, niter=10):
    """
    Asymmetric Least Squares baseline estimation for EDA tonic/phasic decomposition
    Reference: Eilers & Boelens (2005)
    
    Parameters
    ----------
    eda_signal : array
        Input EDA signal
    lam : float
        Smoothness parameter (higher = smoother baseline)
    p : float
        Asymmetry parameter (0 < p < 1)
    niter : int
        Number of iterations
        
    Returns
    -------
    tonic : array
        Estimated tonic (baseline) component
    phasic : array
        Phasic component = eda_signal - tonic
    """
    L = len(eda_signal)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    w = np.ones(L)

    # Check the sparse difference matrix
    # print("Difference matrix shape:", D.shape)

    for i in range(niter):
        # Build weighted diagonal matrix
        W = sparse.spdiags(w, 0, L, L)
        
        # Solve the weighted least squares problem
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w * eda_signal)

        # Update weights asymmetrically: p where eda > baseline, (1-p) otherwise
        w = p * (eda_signal > z) + (1.0 - p) * (eda_signal <= z)

        # # Optional: inspect intermediate tonic estimate
        # if i % 2 == 0:  # every 2 iterations
        #     print(f"Iteration {i}: tonic min={z.min():.3f}, max={z.max():.3f}")

    # Extract tonic and phasic
    tonic = z
    phasic = eda_signal - tonic

    # # Inspect
    # print("Tonic length:", len(tonic))
    # print("Phasic length:", len(phasic))

    return tonic, phasic

def eda_phasic(eda_signal, lam=1e3, p=0.001, niter=10, sampling_rate=None, method=None, **kwargs):
    """Wrapper function to return tonic and phasic components."""
    tonic, phasic = baseline_als(eda_signal, lam=lam, p=p, niter=niter)
    return pd.DataFrame({"EDA_Tonic": tonic, "EDA_Phasic": phasic})
