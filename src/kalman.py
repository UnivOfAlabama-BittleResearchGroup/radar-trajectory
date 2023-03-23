from typing import Tuple
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
import numpy as np
from filterpy.common import Q_continuous_white_noise, Q_discrete_white_noise

import numpy as np


class KalmanConstantAcceleration:
    P_NOISE = 0.5
    MEASUREMENT_NOISE = 1
    SHAPE = (6, 6)

    def __init__(self, dt, x0):
        self.dt = dt
        self.x = np.array([*x0[:2], 1, *x0[2:], 1])

        # this is the kalman gain
        self.P = np.eye(6) * 3

        # cache an identity matrix for convenience
        self.I = np.eye(6)

        # state transition matrix
        self.F = np.array(
            [
                [1.0, 0.1, 0.5 * self.dt**2, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.1, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.1, 0.5 * self.dt**2],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.1],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )

        # process noise. This is a 6x6 matrix, but we only need the top left 3x3
        # and bottom right 3x3 blocks to be non-zero. The rest are zeros.
        self.Q = Q_discrete_white_noise(
            dim=3, dt=self.dt, var=self.P_NOISE, block_size=2
        )

        # measurement matrix
        self.H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            ]
        )

        # measurement noise. We should calculate this from the GPS data
        self.R = np.array(
            [
                [5, 0.0, 0.0, 0.0],
                [0.0, 3, 0.0, 0.0],
                [0.0, 0.0, 3, 0.0],
                [0.0, 0.0, 0.0, 5],
            ]
        )

        self.history = {
            "x": [],
            "P": [],
        }

    @property
    def shape(self) -> tuple:
        return self.F.shape

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.linalg.multi_dot((self.F, self.P, self.F.T)) + self.Q

        return self.x

    def update(self, z: np.array) -> np.ndarray:
        # tracking error
        y = z - np.dot(self.H, self.x)

        # Kalman gain
        S = np.linalg.multi_dot((self.H, self.P, self.H.T)) + self.R

        K = np.linalg.multi_dot((self.P, self.H.T, np.linalg.inv(S)))

        # update state
        self.x = self.x + K @ y

        # update covariance
        self.P = np.linalg.multi_dot((self.I - K @ self.H, self.P))

        # update the history
        self._update_history()

        return self.x

    def _update_history(self) -> None:
        self.history["x"].append(self.x)
        self.history["P"].append(self.P)

    def batch_update(self, zs: np.ndarray) -> np.ndarray:
        preds = []
        for z in zs:
            self.predict()
            preds.append(self.update(z))
        return np.array(preds)

    def rts_smooth(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Performs Rauch-Tung-Striebel smoothing on a set of
        state estimates output from a Kalman filter.

        A forward pass is performed to compute the smoothed estimates,
        followed by a backward pass to compute the smoothed estimates.
        """

        # allocate smoothed estimates
        xs = self.history["x"].copy()
        P = self.history["P"].copy()
        Pp = self.history["P"].copy()
        if "Q" in self.history:
            Qs = self.history["Q"].copy()
        else:
            Qs = [self.Q for _ in range(len(P))]

        # Create a smoother gain matrix
        K = [np.zeros_like(P[0]) for _ in range(len(P))]

        # perform the reverse pass
        for i in range(len(xs) - 2, -1, -1):
            # compute smoother gain
            Pp[i] = self.F @ P[i] @ self.F.T + Qs[i + 1]

            # get the smoother gain
            K[i] = P[i] @ self.F.T @ np.linalg.inv(Pp[i])
            
            
            xs[i] += K[i] @ (xs[i + 1] - self.F @ xs[i])
            P[i] += K[i] @ (P[i + 1] - Pp[i]) @ K[i].T

        return xs, P


class KalmanConstantAccelerationFadingMemory(KalmanConstantAcceleration):

    """
    I could just use filterpy,
    but I want to understand what's going on under the hood

    Returns:
        _type_: _description_
    """

    

    def __init__(self, dt, x0, alpha=1.08):
        super().__init__(dt, x0)

        self.alpha = alpha
        self._q_count = 0

        self.history['Q'] = []

    def update(self, z) -> np.ndarray:
        # tracking error
        y = z - np.dot(self.H, self.x)

        # Kalman gain
        S = np.linalg.multi_dot((self.H, self.P, self.H.T)) + self.R

        K = np.linalg.multi_dot((self.P, self.H.T, np.linalg.inv(S)))

        # update state
        self.x = self.x + K @ y

        # update covariance
        # with fading memory from https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/14-Adaptive-Filtering.ipynb
        self.P = self.alpha**2 * np.linalg.multi_dot((self.I - K @ self.H, self.P))

        # update the history
        self._update_history()

        return self.x
    
    def _update_history(self) -> None:
        super()._update_history()

        self.history["Q"].append(self.Q)


# create a function to automatically Kalman filter the
