"""Generic linear Kalman filter.

No domain-specific logic -- any linear-Gaussian tracking problem picks its
own F/H/Q/R matrices and reuses this. (Currently used by lane_tracker.py to
track a lane line's x-intercepts at two reference rows.)
"""
import numpy as np


class KalmanFilter:
    """Standard predict/update linear Kalman filter."""

    def __init__(
        self,
        x0: np.ndarray,
        P0: np.ndarray,
        F: np.ndarray,
        H: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
    ) -> None:
        self.x = x0.astype(np.float64)
        self.P = P0.astype(np.float64)
        self.F = F.astype(np.float64)
        self.H = H.astype(np.float64)
        self.Q = Q.astype(np.float64)
        self.R = R.astype(np.float64)

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def innovation(self, z: np.ndarray) -> tuple:
        """Residual and residual covariance for a candidate measurement,
        without committing it -- lets a caller gate/reject the measurement
        (e.g. via mahalanobis()) before deciding whether to update()."""
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        return y, S

    def mahalanobis(self, z: np.ndarray) -> float:
        """How many standard deviations away z is from this filter's
        current prediction, accounting for the prediction's own
        uncertainty -- the gate for rejecting an outlier measurement."""
        y, S = self.innovation(z)
        return float(np.sqrt(y.T @ np.linalg.solve(S, y)))

    def update(self, z: np.ndarray) -> np.ndarray:
        y, S = self.innovation(z)
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(len(self.x)) - K @ self.H) @ self.P
        return self.x
