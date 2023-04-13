import copy

# from math import atan2
from typing import List, Tuple, Union
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
import numpy as np
from filterpy.common import Q_continuous_white_noise, Q_discrete_white_noise

import numpy as np
from scipy.linalg import cho_factor, cho_solve
import polars as pl


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

    def __init__(self, dt: float, measured_vars_pos: Tuple[int] = (0, 1, 2, 3)) -> None:
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
                    [3, 0.0, 0.0, 0.0],
                    [0.0, 3, 0.0, 0.0],
                    [0.0, 0.0, 0.2, 0.0],
                    [0.0, 0.0, 0.0, 1],
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
        if np.abs(x[self.yaw_rate]) < 1e-5:
            new_x = (
                x[self.velocity] * self.dt + (x[self.acceleration] * self.dt**2) / 2
            ) * np.cos(x[self.theta]) + x[self.pos_x]
            new_y = (
                x[self.velocity] * self.dt + (x[self.acceleration] * self.dt**2) / 2
            ) * np.sin(x[self.theta]) + x[self.pos_y]

            new_yaw_rate = 0.001
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
        res = self._H @ x
        # wrap the angle
        res[self.theta] = normalize_radians(res[self.theta])
        return res
        # return self._H @ x

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
        print("I shouldn't be here")
        kmax, n = sigmas.shape

        x = self.state_mean(sigmas, Wm)

        P = np.zeros((n, n))
        for k in range(kmax):
            y = self.residual_fn(sigmas[k], x)
            P += Wc[k] * np.outer(y, y)

        if noise_cov is not None:
            P += noise_cov
        return x, P

    # def Q(self, )

    def Q(self, x: np.ndarray) -> np.ndarray:
        # from 10.1109/SDF.2019.8916654

        sigma_a = 5  # the jerk noise. Units are m^2 / s^5
        sigma_w = 3  # the psd for yaw acceleration. Units of this are rad^2 / s^3

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

    def state_mean_gen(self, shape):
        def state_mean(sigmas, Wm):
            x = np.zeros(shape)
            sum_sin = np.sum(np.dot(np.sin(sigmas[:, self.theta]), Wm))
            sum_cos = np.sum(np.dot(np.cos(sigmas[:, self.theta]), Wm))
            # sum the rest of the states
            x = np.dot(Wm, sigmas)
            x[self.theta] = np.arctan2(sum_sin, sum_cos)
            return x

        return state_mean

    def residual_fn(self, a, b):
        y = a - b
        y[self.theta] = normalize_radians(y[self.theta])
        return y

    def build_filter(self, measurements: np.ndarray) -> UnscentedKalmanFilter:
        kf = ModifiedUnscentedKalmanFilter(
            model=self,
            dim_x=self.dim_x,
            dim_z=self.dim_z,
            dt=self.dt,
            fx=self.F,
            hx=self.H,
            points=MerweScaledSigmaPoints(
                self.dim_x,
                alpha=0.1,
                beta=2,
                kappa=3 - self.dim_x,
            ),
            residual_x=self.residual_fn,
            residual_z=self.residual_fn,
            x_mean_fn=self.state_mean_gen(self.dim_x),
            z_mean_fn=self.state_mean_gen(self.dim_z),
        )

        kf.x = np.r_[measurements[0], 0.01, 0.01]
        kf.P = np.diag([0.1 for _ in range(self.dim_x)])  # * 1e-3
        kf.R = self.R

        # initialize the Q matrix
        kf.Q = self.Q(kf.x)
        # kf.inv = np.linalg.pinv

        return kf


class ModifiedUnscentedKalmanFilter(UnscentedKalmanFilter):
    def __init__(self, model: CTRAModel, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = model

    def predict(self, dt=None, UT=None, fx=None, **fx_args):
        self.Q = self.model.Q(self.x)
        return super().predict(dt, UT, fx, **fx_args)

    def update(self, z, R=None, UT=None, hx=None, **hx_args):
        return super().update(z, R, UT, hx, **hx_args)

    def batch_filter(self, zs, Rs=None, dts=None, saver=None):
        # skip passing the UT argument to the unscented transform
        return super().batch_filter(zs, Rs, dts, saver)

    def predict_future(
        self, time_steps: int, override_yaw_rate: bool = True
    ) -> np.ndarray:
        predicted_states = []
        slice_vars = list(self.model.measured_vars_pos)
        for _ in range(time_steps):
            self.predict()
            if override_yaw_rate:
                self.x[self.model.yaw_rate] = 0
            super().update(self.x[slice_vars])
            predicted_states.append(self.x)
        return np.array(predicted_states)


def polarized_unscented_kalman_filter(
    df: pl.DataFrame,
    model: CTRAModel,
    prediction_steps: int = 50,
    override_yaw_rate: bool = True,
) -> pl.DataFrame:
    measurements = df[
        ["epoch_time", "utm_x", "utm_y", "direction", "f32_velocityInDir_mps"]
    ].to_numpy()

    # create a new copy of the model
    model = copy.deepcopy(model)

    kf = model.build_filter(
        measurements[:, 1:],
    )
    try:
        res, Ps = kf.batch_filter(measurements[:, 1:])
        measurements[:, 1:] = res[:, list(model.measured_vars_pos)]
        predicted_states = kf.predict_future(
            prediction_steps, override_yaw_rate=override_yaw_rate
        )[:, list(model.measured_vars_pos)]
    except np.linalg.LinAlgError:
        print("Failed to filter {0}", df["object_id"].take(0).to_list()[0])
        return pl.DataFrame()

    # add time to the states
    _t = model.dt * 1000
    predicted_states = np.hstack(
        (
            np.arange(
                measurements[-1, 0] + _t,
                measurements[-1, 0] + (prediction_steps * _t) + _t,
                _t,
            ).reshape(-1, 1),
            predicted_states,
        )
    )
    stack = np.vstack((measurements, predicted_states))

    return pl.DataFrame(
        stack,
        schema=[
            "epoch_time",
            "utm_x",
            "utm_y",
            "direction",
            "f32_velocityInDir_mps",
        ],
    ).with_columns(
        [
            pl.lit(df[c].take(0)).alias(c)
            for c in df.columns
            if c
            not in [
                "epoch_time",
                "utm_x",
                "utm_y",
                "direction",
                "f32_velocityInDir_mps",
            ]
        ]
    )
