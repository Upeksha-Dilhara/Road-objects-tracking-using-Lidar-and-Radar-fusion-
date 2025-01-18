import numpy as np
import matplotlib.pyplot as plt

# Function calculate process jacobian
def calculate_jacobian(x_t, dt):
    v = x_t[2]
    yaw = x_t[3]
    yaw_rate = x_t[4]
    
    if abs(yaw_rate) > 1e-3:  # Non-zero yaw rate
        S = -np.sin(yaw) + np.sin(yaw + yaw_rate * dt)
        C = np.cos(yaw) - np.cos(yaw + yaw_rate * dt)
        F = (v * dt / yaw_rate) * np.cos(yaw + yaw_rate * dt) - (v / yaw_rate**2) * S
        G = (v * dt / yaw_rate) * np.sin(yaw + yaw_rate * dt) - (v / yaw_rate**2) * C
    else:  # Zero yaw rate (straight-line motion)
        S = 0  # S becomes 0
        C = 0  # C becomes 0
        F = v * dt * np.cos(yaw)  # Simplified term
        G = v * dt * np.sin(yaw)  # Simplified term
    
    # Construct the Jacobian matrix
    G_t_plus_1 = np.array([
        [1, 0, S / (yaw_rate if abs(yaw_rate) > 1e-3 else 1), -v * C / (yaw_rate if abs(yaw_rate) > 1e-3 else 1), F],
        [0, 1, C / (yaw_rate if abs(yaw_rate) > 1e-3 else 1),  v * S / (yaw_rate if abs(yaw_rate) > 1e-3 else 1), G],
        [0, 0, 1,            0,                0],
        [0, 0, 0,            1,                dt],
        [0, 0, 0,            0,                1]
    ])
    
    return G_t_plus_1


# Function calculate radar measurement jacobian 
def radar_jacobian(x_t_plus_1_pred):
    px = x_t_plus_1_pred[0]
    py = x_t_plus_1_pred[1]
    v = x_t_plus_1_pred[2]
    yaw = x_t_plus_1_pred[3]

    P = np.sqrt(px**2 + py**2)
    if P < 1e-3:  # Avoid division by zero
        raise ValueError("Range is too small to compute the Jacobian.")
    
    P1 = py * np.cos(yaw) - px * np.sin(yaw)
    P2 = px * np.cos(yaw) + py * np.sin(yaw)

    # Construct the Jacobian matrix
    H_t_plus_1_radar = np.array([
        [px / P, py / P, 0, 0, 0],
        [-py / P**2, px / P**2, 0, 0, 0],
        [(py * v * P1) / P, -(px * v * P1) / P, P2 / P, v * P1 / P, 0]
    ])

    return H_t_plus_1_radar


# Time settings
dt = 0.1  # Time step (s)
t = np.arange(0, 50 + dt, dt)  # Simulation duration

# Ground truth values for velocity and yaw rate(Constant)
v = 5 + 2 * np.sin(0.2 * t)  # Velocity varying as a sine wave
yaw_rate_t = 0.3 * np.sin(0.2 * t)

# Initialize arrays for states variables 
px_gt = np.zeros_like(t)  # Position X
py_gt = np.zeros_like(t)  # Position Y
yaw_gt = np.zeros_like(t)  # Yaw angle

# Ground truth state variable at each time step
for k in range(0, len(t)-1):
    if abs(yaw_rate_t[k]) > 1e-3:  # If yaw rate is non-zero
        px_gt[k + 1] = px_gt[k] + (v[k] / yaw_rate_t[k]) * (np.sin(yaw_gt[k] + yaw_rate_t[k] * dt) - np.sin(yaw_gt[k]))
        py_gt[k + 1] = py_gt[k] + (v[k] / yaw_rate_t[k]) * (-np.cos(yaw_gt[k] + yaw_rate_t[k] * dt) + np.cos(yaw_gt[k]))
    else:  # If yaw rate is zero
        px_gt[k + 1] = px_gt[k] + v[k] * np.cos(yaw_gt[k]) * dt
        py_gt[k + 1] = py_gt[k] + v[k] * np.sin(yaw_gt[k]) * dt

    yaw_gt[k+1] = yaw_gt[k] + yaw_rate_t[k] * dt 
    yaw_gt[k+1] = (yaw_gt[k+1] + np.pi) % (2 * np.pi) - np.pi

x_ground_truth = np.zeros((5, len(t)))
x_ground_truth[0, :] = px_gt  # Position X
x_ground_truth[1, :] = py_gt  # Position Y
x_ground_truth[2, :] = v      # Velocity (constant)
x_ground_truth[3, :] = yaw_gt  # Yaw angle
x_ground_truth[4, :] = yaw_rate_t  # Yaw rate (constant

# Process noise parameters 
sigma_a = 2.0  # Standard deviation of linear acceleration (m/s^2)
sigma_ax = 3.0  # Standard deviation of x-axis acceleration (m/s^2)
sigma_ay = 1.5  # Standard deviation of y-axis acceleration (m/s^2)
sigma_yawdd = 0.9  # Standard deviation of yaw acceleration (rad/s^2)
sigma_yawrate = 0.2  # Standard deviation of yaw rate (rad/s)


# Initialize process noise covariance matrix
dt2 = dt ** 2
dt3 = dt ** 3
dt4 = dt ** 4
R =  np.array([
        [dt4 / 4 * sigma_ax**2, 0, dt3 / 2 * sigma_a**2, 0, 0],
        [0, dt4 / 4 * sigma_ay**2, dt3 / 2 * sigma_ay**2, 0, 0],
        [dt3 / 2 * sigma_ax**2, dt3 / 2 * sigma_ay**2, dt2 * sigma_a**2, 0, 0],
        [0, 0, 0, dt2 * sigma_yawrate**2, 0],
        [0, 0, 0, 0, dt2 * sigma_yawdd**2]
    ])


# Sensor noise parameters
sigma_px = 0.1  # LiDAR position noise
sigma_py = 0.1 
sigma_rho = 0.05  # Radar range noise
sigma_phi = 0.1  # Radar angle noise
sigma_rho_dot = 0.05  # Radar range rate noise

# Initialize measurement noise covariance matrices for Lidar and Radar
Q_lidar = np.diag([sigma_px**2, sigma_py**2])  # LiDAR measurement noise
Q_radar = np.diag([sigma_rho**2, sigma_phi**2, sigma_rho_dot**2])  # Radar measurement noise


# Generate noisy sensor measurements for Lidar in each time step 
px_lidar = px_gt + np.random.randn(len(px_gt)) * sigma_px  
py_lidar = py_gt + np.random.randn(len(py_gt)) * sigma_py  

# Generate noisy sensor measurements for Radar in each time step
rho_radar = np.sqrt(px_gt**2 + py_gt**2) + np.random.randn(len(px_gt)) * sigma_rho
phi_radar = np.arctan2(py_gt, px_gt) + np.random.randn(len(px_gt)) * sigma_phi
rho_dot_radar = (px_gt * v * np.cos(yaw_gt) + py_gt * v * np.sin(yaw_gt)) / rho_radar + np.random.randn(len(px_gt)) * sigma_rho_dot

# Initial state and covariance
x_t = np.array([px_lidar[0], py_lidar[0], 0, 0, 0])  # Initial state /x_0
Sigma_t= np.eye(5)  # Initial covariance / Sigma_0
Sigma_t[0][0] = 1.0
Sigma_t[1][1] = 1.0
Sigma_t[2][2] = np.sqrt(1000)
Sigma_t[3][3] = np.sqrt(1000)
Sigma_t[4][4] = np.sqrt(1000)


# Storage for EKF results
x_estimates = np.zeros((5, len(t)))
x_estimates[:, 0] = x_t
# EKF Loop
for k in range(0, len(t)-1):
    x_t = x_estimates[:,k]
    Px_t = x_t[0]
    Py_t = x_t[1]
    v_t = x_t[2]
    yaw_t = x_t[3]
    yaw_rate_t = x_t[4]

    ##############Predition step 
    if abs(yaw_rate_t) > 1e-3:
        Px_t_plus_1_pred = Px_t + (v_t / yaw_rate_t) * (np.sin(yaw_t + yaw_rate_t * dt) - np.sin(yaw_t))
        Py_t_plus_1_pred = Py_t + (v_t / yaw_rate_t) * (-np.cos(yaw_t + yaw_rate_t * dt) + np.cos(yaw_t))
    else:
        Px_t_plus_1_pred = Px_t + v_t * np.cos(yaw_t) * dt
        Py_t_plus_1_pred = Py_t + v_t * np.sin(yaw_t) * dt
    v_t_plus_1_pred = v_t
    yaw_t_plus_1_pred = yaw_t + yaw_rate_t * dt
    yaw_t_plus_1_pred = (yaw_t_plus_1_pred + np.pi) % (2 * np.pi) - np.pi
    yaw_rate_t_plus_1_pred = yaw_rate_t
    x_t_plus_1_pred = np.array([Px_t_plus_1_pred, Py_t_plus_1_pred, v_t_plus_1_pred, yaw_t_plus_1_pred, yaw_rate_t_plus_1_pred])

    # Jacobian of the motion model at  x_t
    G_t_plus_1 = calculate_jacobian(x_t, dt )

    # Predict covariance
    Sigma_t_plus_1_pred = G_t_plus_1 @ Sigma_t @ G_t_plus_1.T + R

    ############# Update step

    z_t_plus_1_pred = np.array([Px_t_plus_1_pred, Py_t_plus_1_pred])
    #Jacobian for lidar measurment at x_t_plus_1_pred
    H_t_plus_1 = np.array([[1, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0]])
    z_t_plus_1 = np.array([px_lidar[k+1], py_lidar[k+1]])
    Q = Q_lidar

    # Kalman gain
    S_t_plus_1 = H_t_plus_1 @ Sigma_t_plus_1_pred @ H_t_plus_1.T + Q
    K_t_plus_1 = Sigma_t_plus_1_pred @ H_t_plus_1.T @ np.linalg.inv(S_t_plus_1)

    # Update state and covariance
    x_t_plus_1 = x_t_plus_1_pred + K_t_plus_1 @ (z_t_plus_1 -z_t_plus_1_pred)
    Sigma_t_plus_1 = (np.eye(len(K_t_plus_1)) - K_t_plus_1 @ H_t_plus_1) @ Sigma_t_plus_1_pred
    # Store results
    x_estimates[:, k+1] = x_t_plus_1

# print(x_estimates.shape)

# Create a 2x2 grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# Plot 1: EKF Position Estimation
axes[0, 0].plot(x_ground_truth[0, :], x_ground_truth[1, :], label="Ground Truth Position", color="r",linewidth="1")
axes[0, 0].plot(x_estimates[0, :], x_estimates[1, :], label="EKF Estimate Position", color="b", linewidth="1")
axes[0, 0].set_xlabel("Position X (m)")
axes[0, 0].set_ylabel("Position Y (m)")
axes[0, 0].set_title("EKF Trajectory Estimation")
axes[0, 0].grid()
axes[0, 0].legend()

# Plot 2: EKF Velocity Estimation
axes[0, 1].plot(t, x_estimates[2, :], label="EKF Estimate Velocity", color="b", linewidth="1")
axes[0, 1].plot(t, x_ground_truth[2, :], label="Ground Truth Velocity", color="r", linewidth="1")
axes[0, 1].set_xlabel("Time")
axes[0, 1].set_ylabel("Velocity (m/s)")
axes[0, 1].set_title("EKF Velocity Estimation")
axes[0, 1].grid()
axes[0, 1].legend()

# Plot 3: EKF Yaw Angle Estimation
axes[1, 0].plot(t, x_estimates[3, :], label="EKF Estimate Yaw Angle", color="b", linewidth="1")
axes[1, 0].plot(t, x_ground_truth[3, :], label="Ground Truth Yaw Angle", color="r", linewidth="1")
axes[1, 0].set_xlabel("Time")
axes[1, 0].set_ylabel("Yaw Angle (rad)")
axes[1, 0].set_title("EKF Yaw Angle Estimation")
axes[1, 0].grid()
axes[1, 0].legend()

# # Plot 4: EKF Yaw Angle Rate Estimation
axes[1, 1].plot(t, x_estimates[4, :], label="EKF Estimate Yaw Angle Rate", color="b", linewidth="1")
axes[1, 1].plot(t, x_ground_truth[4, :], label="Ground Truth Yaw Angle Rate", color="r", linewidth="1")
axes[1, 1].set_xlabel("Time")
axes[1, 1].set_ylabel("Yaw Angle Rate (rad/s)")
axes[1, 1].set_title("EKF Yaw Angle Rate Estimation")
axes[1, 1].grid()
axes[1, 1].legend()
# Adjust layout for better spacing
plt.tight_layout()
plt.show()


# Compute RMSE 
rmse = np.sqrt(np.mean((x_estimates - x_ground_truth)**2, axis=1))
assert rmse.shape[0] == x_estimates.shape[0], "Mismatch in RMSE dimensions and state variables."
state_variables = ['px', 'py', 'v', 'yaw', 'yaw rate']  

print("Root Mean Square Error (RMSE):")
for i, state in enumerate(state_variables):
    print(f"{state}: {rmse[i]:.4f}")

 