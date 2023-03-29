from math import atan2
from typing import List, Tuple, Union
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
import numpy as np
from filterpy.common import Q_continuous_white_noise, Q_discrete_white_noise

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def normalize_radians(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Normalizes an angle to the range [-pi, pi)."""
    return (x + np.pi) % (2 * np.pi) - np.pi


class KalmanConstantAcceleration:
    P_NOISE = 1
    MEASUREMENT_NOISE = 1
    SHAPE = (6, 6)

    def __init__(self, dt, x0):
        self.dt = dt
        self.x = np.array([*x0[:2], 1, *x0[2:], 1])

        # this is the kalman gain
        self.P = np.eye(6)

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
        self.Q = Q_continuous_white_noise(dim=3, dt=self.dt, block_size=2)

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
                [0.0, 5, 0.0, 0.0],
                [0.0, 0.0, 5, 0.0],
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

        self.history["Q"] = []

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
class CTRAModel:
    pos_x = 0
    pos_y = 1
    theta = 2
    velocity = 3
    acceleration = 4
    yaw_rate = 5

    dim_x = 6

    def __init__(self, dt: float, measured_vars_pos: Tuple[int] = (0, 1, 3)) -> None:
        self.dt = dt
        self.measured_vars_pos = measured_vars_pos
        self.dim_z = len(measured_vars_pos)
        # build H matrix
        self._H = np.zeros((len(measured_vars_pos), 6))
        for i, pos in enumerate(measured_vars_pos):
            self._H[i, pos] = 1

    @property
    def R(
        self,
    ) -> np.ndarray:
        return (
            np.array(
                [
                    [4.0, 0.0, 0, 0],
                    [0.0, 4.0, 0, 0],
                    [0.0, 0.0, 1, 0.0],
                    [0, 0, 0, 3],
                ]
            )
            ** 2
        )
        # return np.array(
        #     [
        #         [2, 0.0, 0.0],
        #         [0.0, 2, 0.0],
        #         # [0.0, 0.0, 0.8, 0.0],
        #         [0.0, 0.0, 1],
        #     ]
        # ) ** 2

    def F(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        """State transition function

        Args:
            x (np.ndarray): state vector

        Returns:
            np.ndarray: state vector
        """
        # calculate x
        # check if the yaw_rate is near zero
        if np.abs(x[self.yaw_rate]) < 1e-4:
            new_x = (
                x[self.velocity] * self.dt + (x[self.acceleration] * self.dt**2) / 2
            ) * np.cos(x[self.theta]) + x[self.pos_x]
            new_y = (
                x[self.velocity] * self.dt + (x[self.acceleration] * self.dt**2) / 2
            ) * np.sin(x[self.theta]) + x[self.pos_y]

            new_yaw_rate = 0.0001
        else:
            a1 = 1 / x[self.yaw_rate] ** 2
            a2 = (
                x[self.velocity] * x[self.yaw_rate]
                + x[self.acceleration] * x[self.yaw_rate] * self.dt
            )
            cos_ = np.cos(x[self.theta] + x[self.yaw_rate] * self.dt)
            sin_ = np.sin(x[self.theta] + x[self.yaw_rate] * self.dt)

            new_x = (
                a1
                * (
                    a2 * sin_
                    + x[self.acceleration] * (cos_ - np.cos(x[self.theta]))
                    - x[self.velocity] * x[self.yaw_rate] * np.sin(x[self.theta])
                )
                + x[self.pos_x]
            )

            new_y = (
                a1
                * (
                    -1 * a2 * cos_
                    + x[self.acceleration] * (sin_ - np.sin(x[self.theta]))
                    + x[self.velocity] * x[self.yaw_rate] * np.cos(x[self.theta])
                )
                + x[self.pos_y]
            )

            new_yaw_rate = x[self.yaw_rate]

        new_theta = x[self.yaw_rate] * self.dt + x[self.theta]
        new_velocity = x[self.acceleration] * self.dt + x[self.velocity]
        new_x = np.array(
            [new_x, new_y, new_theta, new_velocity, x[self.acceleration], new_yaw_rate]
        )

        # normalize theta
        new_x[self.theta] = normalize_radians(new_x[self.theta])

        return new_x

    def H(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        return self._H @ x

    # # # @property
    # def Q(self, *args, **kwargs) -> np.ndarray:
    #     q = np.zeros((6, 6))
    #     q[self.pos_x, self.pos_x] = 0.5 * 8.8 * self.dt**2
    #     q[self.pos_y, self.pos_y] = 0.5 * 8.8 * self.dt**2
    #     q[self.theta, self.theta] = 4 * self.dt
    #     q[self.velocity, self.velocity] = 8.8 * self.dt
    #     q[self.acceleration, self.acceleration] = 2.0 * self.dt
    #     q[self.yaw_rate, self.yaw_rate] = 0.1 * self.dt

    #     return q
    # return np.diag([
    #     0.01,
    #     0.01,
    #     0.2,
    #     0.01,
    #     1,
    #     1
    # ]) * 1

    def residual_func(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Residual function

        Args:
            x1 (np.ndarray): state vector
            x2 (np.ndarray): state vector

        Returns:
            np.ndarray: residual vector
        """
        y = x1 - x2
        # y[self.theta] = (y[self.theta] + np.pi) % (2 * np.pi) - np.pi
        return y

    def unscented_transform(
        self, sigmas, Wm, Wc, noise_cov=None, *args, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Unscented transform function

        Args:
            sigmas (np.ndarray): sigma points
            Wm (np.ndarray): weights for the mean
            Wc (np.ndarray): weights for the covariance

        Returns:
            Tuple[np.ndarray, np.ndarray]: mean and covariance
        """
        kmax, n = sigmas.shape

        # sigma points should not have negative velocity
        # sigmas[:, self.velocity] = np.abs(sigmas[:, self.velocity])

        # x = np.dot(Wm, sigmas)  # dot = \Sigma^n_1 (W[k]*Xi[k])
        sum_sin, sum_cos = 0.0, 0.0

        x = np.zeros(n)
        for i in range(len(sigmas)):
            s = sigmas[i]
            x[:] += s[:] * Wm[i]
            # x[1] += s[1] * Wm[i]
            sum_sin += np.sin(s[self.theta]) * Wm[i]
            sum_cos += np.cos(s[self.theta]) * Wm[i]
            x[self.theta] = atan2(sum_sin, sum_cos)
        x[self.theta] = np.arctan2(sum_sin, sum_cos)

        # P = np.dot(y.T, np.dot(np.diag(Wc), y))
        # # # P += self.Q(x)
        P = np.zeros((n, n))
        for k in range(kmax):
            y = sigmas[k] - x
            # normalize the angle
            y[self.theta] = normalize_radians(y[self.theta])
            P += Wc[k] * np.outer(y, y)

        if noise_cov is not None:
            P += self.Q(x)

        return x, P

    def Q(self, x: np.ndarray) -> np.ndarray:
        # from 10.1109/SDF.2019.8916654

        sigma_a = 5  # the jerk noise. Units are m^2 / s^5
        sigma_w = 2  # the psd for yaw acceleration. Units of this are rad^2 / s^3

        v_k = x[self.velocity]
        theta_k = x[self.theta]

        return np.array(
            [
                [
                    (
                        sigma_w**2 * v_k**2 * np.sin(theta_k) ** 2
                        + sigma_a**2 * np.cos(theta_k) ** 2
                    )
                    * self.dt**5
                    / 20,
                    (sigma_a**2 - sigma_w**2 * v_k**2)
                    * self.dt**5
                    / 20
                    * np.sin(theta_k)
                    * np.cos(theta_k),
                    sigma_a**2 * self.dt**4 / 8 * np.cos(theta_k),
                    -(sigma_w**2) * self.dt**4 / 8 * v_k * np.sin(theta_k),
                    -(sigma_w**2) * self.dt**3 / 6 * v_k * np.sin(theta_k),
                    sigma_a**2 * self.dt**3 / 6 * np.cos(theta_k),
                ],
                [
                    (sigma_a**2 - sigma_w**2 * v_k**2)
                    * self.dt**5
                    / 20
                    * np.sin(theta_k)
                    * np.cos(theta_k),
                    (
                        sigma_w**2 * v_k**2 * np.cos(theta_k) ** 2
                        + sigma_a**2 * np.sin(theta_k) ** 2
                    )
                    * self.dt**5
                    / 20,
                    sigma_a**2 * self.dt**4 / 8 * np.sin(theta_k),
                    sigma_w**2 * self.dt**4 / 8 * v_k * np.cos(theta_k),
                    sigma_w**2 * self.dt**3 / 6 * v_k * np.cos(theta_k),
                    sigma_a**2 * self.dt**3 / 6 * np.sin(theta_k),
                ],
                [
                    sigma_a**2 * self.dt**4 / 8 * np.cos(theta_k),
                    sigma_a**2 * self.dt**4 / 8 * np.sin(theta_k),
                    sigma_a**2 * self.dt**3 / 3,
                    0,
                    0,
                    sigma_a**2 * self.dt**2 / 2,
                ],
                [
                    -(sigma_w**2) * self.dt**4 / 8 * v_k * np.sin(theta_k),
                    sigma_w**2 * self.dt**4 / 8 * v_k * np.cos(theta_k),
                    0,
                    sigma_w**2 * self.dt**3 / 3,
                    sigma_w**2 * self.dt**2 / 2,
                    0,
                ],
                [
                    -(sigma_w**2) * self.dt**3 / 6 * v_k * np.sin(theta_k),
                    sigma_w**2 * self.dt**3 / 6 * v_k * np.cos(theta_k),
                    0,
                    sigma_w**2 * self.dt**2 / 2,
                    sigma_w**2 * self.dt,
                    0,
                ],
                [
                    sigma_a**2 * self.dt**3 / 6 * np.cos(theta_k),
                    sigma_a**2 * self.dt**3 / 6 * np.sin(theta_k),
                    sigma_a**2 * self.dt**2 / 2,
                    0,
                    0,
                    sigma_a**2 * self.dt,
                ],
            ]
        )  # + np.eye(6) * 1e-6
