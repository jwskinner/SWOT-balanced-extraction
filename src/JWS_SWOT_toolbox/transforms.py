# use the julia ones dont even bother with these, too slow
import numpy as np 

def sqrtint(S, k):
    """
    Compute ∫_{k0}^∞ S(κ)/sqrt(κ^2 - k0^2) dκ
    Inputs S, k are 1D arrays sliced to start at the lower limit k0 = k[0].
    Used in the Abel transforms 
    """
    S = np.asarray(S, dtype=float)
    k = np.asarray(k, dtype=float)
    n = len(S)

    I = 0.0
    if np.isclose(k[0], 0.0):
        I += S[1]
        for i in range(1, n - 1):  # i = 1 .. n-2 (0-based)
            denom = (k[i] - k[i+1])
            I += (S[i+1] - S[i]) + (S[i+1]*k[i] - S[i]*k[i+1]) * np.log(k[i+1]/k[i]) / denom
    else:
        k0 = k[0]
        r = np.sqrt(np.maximum(k**2 - k0**2, 0.0))
        for i in range(0, n - 1):  # i = 0 .. n-2
            denom = (k[i] - k[i+1])
            # r[i] = sqrt(k[i]^2 - k0^2)
            term1 = (S[i+1] - S[i]) * (r[i] - r[i+1])
            num_log = (k[i]   - r[i])
            den_log = (k[i+1] - r[i+1])
            # guard tiny roundoff (should be positive by construction for k>k0)
            val = np.log(num_log / den_log)
            term2 = (S[i+1]*k[i] - S[i]*k[i+1]) * val
            I += (term1 + term2) / denom
    return I

def abel(Sr, k):
    """
    Forward Abel transform:
        S1(k) = (2/π) ∫_k^∞ Sr(kk)/sqrt(κ^2 - k^2) dkk
    """
    print("Forward Abel \n")
    Sr = np.asarray(Sr, dtype=float)
    k  = np.asarray(k,  dtype=float)
    n = len(k)
    assert len(Sr) == n and n >= 3
    S1 = np.zeros(n, dtype=float)
    for i in range(n):
        S1[i] = (2.0/np.pi) * sqrtint(Sr[i:], k[i:])
    return S1

def iabel(S1, kappa):
    """
    Inverse Abel transform:
        Sr(κ) = -κ ∫_κ^∞ S1'(k)/sqrt(k^2 - κ^2) dk
    with 3-point one-sided / centered finite differences.
    """
    print("Inverse Abel \n")
    S1    = np.asarray(S1, dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    n = len(kappa)
    assert len(S1) == n and n >= 3, "Need S1, k of same length (>=3)."

    dκ = kappa[1] - kappa[0]
    # 3-point one-sided at ends, centered in the interior:
    S1p = np.empty(n, dtype=float)
    S1p[0]  = (-3.0*S1[0] + 4.0*S1[1] - S1[2]) / (2.0*dκ)
    S1p[1:-1] = (S1[2:] - S1[:-2]) / (2.0*dκ)
    S1p[-1] = (S1[-3] - 4.0*S1[-2] + 3.0*S1[-1]) / (2.0*dκ)

    Sr = np.zeros(n, dtype=float)
    for i in range(n):
        Sr[i] = -kappa[i] * sqrtint(S1p[i:], kappa[i:])
    return Sr
