import numpy as np

# Class KalmanFilter
# This class implements a Kalman filter for a 2D state vector
class KalmanFilter:
    # Constructor
    def __init__(self, dt=0.1, u_x=1, u_y=1, std_acc=1, x_sdt_meas=0.1, y_sdt_meas=0.1):
        # Define sampling time
        self.dt = dt

        # Input variables
        self.u = np.matrix([[u_x], [u_y]])

        # State matrix
        self.x = np.matrix([[0], [0], [0], [0]])

        # Transition matrix
        self.A = np.matrix([[1, 0, self.dt, 0],
                            [0, 1, 0, self.dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

        # Control input matrix
        self.B = np.matrix([[(self.dt**2)/2, 0],
                            [0, (self.dt**2)/2],
                            [self.dt, 0],
                            [0, self.dt]])

        # Measurement mapping matrix
        self.H = np.matrix([[1, 0, 0, 0],
                            [0, 1, 0, 0]])

        # Measurement noise covariance matrix
        self.R = np.matrix([[x_sdt_meas**2, 0],
                            [0, y_sdt_meas**2]])

        # Define the process noise covariance matrix
        self.Q = np.matrix([[(self.dt**4)/4, 0, (self.dt**3)/2, 0],
                            [0, (self.dt**4)/4, 0, (self.dt**3)/2],
                            [(self.dt**3)/2, 0, self.dt**2, 0],
                            [0, (self.dt**3)/2, 0, self.dt**2]]) * std_acc**2

        # Define the state covariance matrix
        self.P = np.eye(self.A.shape[1])

    # Predict the next state
    def predict(self):
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
    
    # Update the state based on measurements
    def update(self, z):
        # Calculate the Kalman gain
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # Update the state estimate
        z = np.array([[z[0]], [z[1]]])
        self.x = np.round(self.x + np.dot(K, (z - np.dot(self.H, self.x))))

        # Update the covariance
        I = np.eye(self.H.shape[1])
        self.P = (I - (K * self.H)) * self.P