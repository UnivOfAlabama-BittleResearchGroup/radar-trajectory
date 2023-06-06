import numpy as np
import scipy





class KalmanFilter(object):
    """
    This is the Kalman filter fuses two sensors (camera and radar).
    The state = [x, y, x', y'], [x, y] is the position in NE-world coordinates
    F: state transition matrix --> (4,4)
    H: measurement_c matrix(2,4), measurement_r matrix(2,4)
    X: mean --> [x, y, x', y'] --> (4,) vector
    P: covariance --> (x_dim, x_dim) --> (4,4)
    Q: process noise --> (x_dim, x_dim) --> (4,4)
    R: measurement noise --> (z_dim, z_dim) --> (2,4), z --> measurement
    """

    count = 0

    def __init__(self, pos):
        """
        Initialises a tracker using initial camera bounding box
        state_dim=4, sensor_1_dim(camera)=2, sensor_2_dim(radar)=2
        """
        dt = 1  # time step
        x_dim, c_dim, r_dim = 4, 2, 2

        # create Kalman filter model matrices, F and H
        self._motion_mat = np.eye(x_dim, x_dim)  # F=(4,4)
        for i in range(2):
            self._motion_mat[i, c_dim + i] = dt

        self._update_1_mat = np.eye(c_dim, x_dim)  # H_1=(2,4)
        self._update_2_mat = np.eye(r_dim, x_dim)  # H_2=(2,4)

        # create state space, mean and covariance --> X and P
        std_weight_position = 1.0 / 20
        std_weight_velocity = 1.0 / 160

        #         self.mean = np.r_[pos, np.zeros(2)]        # X=[x,y,x',y'],(4,)
        self.mean = np.r_[
            pos, np.array([1.06, 0.28])
        ]  # X=[x,y,x',y'],(4,) 25mph(1.06,0.28)

        std = [
            1 * std_weight_position * self.mean[0],  # x
            1 * std_weight_position * self.mean[1],  # y
            2 * std_weight_velocity * self.mean[0],  # x'
            2 * std_weight_velocity * self.mean[1],
        ]  # y'
        self.covariance = np.diag(np.square(std))  # P=(4x4)

        # create the process and measurement noise --> Q and R
        std_pos = [
            std_weight_position * self.mean[0],  # x
            std_weight_position * self.mean[1],
        ]  # y
        std_vel = [
            std_weight_velocity * self.mean[0],  # x'
            std_weight_velocity * self.mean[1],
        ]  # y'
        self.motion_cov = np.diag(np.square(std_pos + std_vel))  # Q=(4x4)

        std_1 = [
            std_weight_position * self.mean[0],  # x
            std_weight_position * self.mean[1],
        ]  # y
        self.innovation_cov_1 = np.diag(np.square(std_1))  # R_1=(2x2), camera

        std_2 = [
            std_weight_position * self.mean[0],  # x
            std_weight_position * self.mean[1],
        ]  # y
        self.innovation_cov_2 = np.diag(np.square(std_2))  # R_2=(2x2), radar

        # other paramters
        self.id = KalmanFilter.count
        KalmanFilter.count += 1

        self.time_since_update = 0  # the time since last update (>max_age: disap obj)
        self.hit_streak = 0  # update times (>min_hits: new obj)
        self.credit = 0  ####

    def predict(self):
        """
        Run Kalman filter prediction step.
        Fomula:
                X_t = F * X_(t-1), (4,)=(4x4)*(4,)
                P_t = F * P_(t-1) * F.T + Q, (4x4)=(4x4)*(4x4)*(4x4)+(4x4)

        Returns(output)
        -------
        mean, covariance: ndarray, ndarray
        Returns the mean vector(4,) and covariance matrix(4x4) of the predicted state.
        Unobserved velocities are initialized to 0 mean.
        """
        self.credit = 0
        # get predicted mean and covariance
        # X_t = F * X_(t-1), (4,)
        self.mean = np.dot(self._motion_mat, self.mean)
        # P_t = F * P_(t-1) * F.T + Q, (4x4)
        self.covariance = (
            np.linalg.multi_dot((self._motion_mat, self.covariance, self._motion_mat.T))
            + self.motion_cov
        )

        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.credit += 1

        return np.array(self.mean[:2]).reshape((1, 2))

    def update_c(self, cams):
        """
        1.Project predicted state distribution to camera measurement space.
        Fomula:
                u_1 = H_1 * X_t          # (2,)=(2x4)*(4,) --> mean_1
                E_1 = H_1 * P_t * H_1.T  # (2x2)=(2x4)*(4x4)*(4x2) --> covariance_1

        2.Run Kalman filter correction step (fuse with the measurement, camera).
        Fomula:
                K = (P_t*H_1.T)/(H_1*P_t*H_1.T + R)  # Kalman gain
                X_t' = X_t + K*(bbox - H_1*X_t)      # mean_update_with_camera
                P_t' = P_t - K*H_1*P_t               # variance_update_with_camera

        Paramters(inputs)
        ---------
        c_position: camera measurement, [x,y]

        """
        # set parameters
        self.time_since_update = 0
        self.hit_streak += 1
        self.credit += 1

        # project the state(x) to camera space
        # mean, [x,y], u_1 = H_1 * X_t, (2x4)(4,)=(2,)
        mean_1 = np.dot(self._update_1_mat, self.mean)
        # E_1 = H_1 * P_t * H_1.T, (2x2)
        covariance_1 = np.linalg.multi_dot(
            (self._update_1_mat, self.covariance, self._update_1_mat.T)
        )

        # update the measurement_1 (camera), projected_mean = mean_1
        projected_cov = covariance_1 + self.innovation_cov_1  # E_1 + R_1, (2x2)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )
        # K, (4x2)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(self.covariance, self._update_1_mat.T).T,
            check_finite=False,
        ).T
        innovation = cams - mean_1  # y = z - u_1, (2,)
        # X_update1 = X + y*K.T, (4,)=(4,)+(2,)(2x4)
        self.mean = self.mean + np.dot(innovation, kalman_gain.T)
        # P_update1 = P - K*(E_1+R_1)*K.T
        self.covariance = self.covariance - np.linalg.multi_dot(
            (kalman_gain, projected_cov, kalman_gain.T)
        )

        return np.array(self.mean[:2]).reshape((1, 2))

    def update_r(self, rads):
        # set parameters
        self.time_since_update = 0
        self.hit_streak += 1
        self.credit += 1

        # project the state(x) to radar space
        # mean, [x,y], u_2 = H_2 * X_t, (2x4)(4,)=(2,)
        mean_2 = np.dot(self._update_2_mat, self.mean)
        # E_2 = H_2 * P_t * H_2.T, (2x2)
        covariance_2 = np.linalg.multi_dot(
            (self._update_2_mat, self.covariance, self._update_2_mat.T)
        )

        ###########################################################
        # update the measurement_2 (radar), projected_mean = mean_2
        theta = 14.5
        X = rads[0] * np.cos(np.radians(theta)) + rads[1] * np.sin(np.radians(theta))
        b_line = 65
        if X < b_line:
            projected_cov = covariance_2 + self.innovation_cov_2  # E_2 + R_2, (2x2)
        else:
            projected_cov = covariance_2 + 0.75 * self.innovation_cov_2  ####
        ###########################################################

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )
        
        # K, (4x2)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(self.covariance, self._update_2_mat.T).T,
            check_finite=False,
        ).T
        
        innovation = rads - mean_2  # y = z - u_1, (2,)
        
        # X_update1 = X + y*K.T, (4,)=(4,)+(2,)(2x4)
        self.mean = self.mean + np.dot(innovation, kalman_gain.T)
        
        # P_update1 = P - K*(E_1+R_1)*K.T
        self.covariance = self.covariance - np.linalg.multi_dot(
            (kalman_gain, projected_cov, kalman_gain.T)
        )

    def get_state(self):
        """
        Returns the current bounding box estimate, [x,y], (1,2)
        """
        return np.array(self.mean[:2]).reshape((1, 2))
