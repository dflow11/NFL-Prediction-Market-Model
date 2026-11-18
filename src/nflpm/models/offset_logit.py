"""Model B: logit p = logit(kalshi_mid) + w·x, L2 penalty λ‖w‖² on every weight (no evidence → B = A)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def fit(x: np.ndarray, offset: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    if len(y) == 0:
        return np.zeros(x.shape[1])

    def loss(w: np.ndarray) -> tuple[float, np.ndarray]:
        z = offset + x @ w
        nll = np.sum(np.logaddexp(0, z) - y * z)  # stable −log-likelihood
        return nll + lam * w @ w, x.T @ (expit(z) - y) + 2 * lam * w

    return minimize(loss, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B").x


def predict(w: np.ndarray, x: np.ndarray, offset: np.ndarray) -> np.ndarray:
    return expit(offset + x @ w)
