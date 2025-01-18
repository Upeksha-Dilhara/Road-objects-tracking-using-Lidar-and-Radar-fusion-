import numpy as np
import matplotlib.pyplot as plt

# Time parameters
dt = 0.1  # Time step (s)
t = np.arange(0, 50 + dt, dt)  # Simulation duration

# Ground truth values for velocity and yaw rate
v = 5 + 2 * np.sin(0.2 * t)  # Velocity varying as a sine wave
yaw_rate_t = 0.3 * np.sin(0.2 * t)  # Sinusoidal yaw rate for figure-eight path

# Initialize arrays for state variables
px_gt = np.zeros_like(t)  # Position X
py_gt = np.zeros_like(t)  # Position Y
yaw_gt = np.zeros_like(t)  # Yaw angle

# Ground truth state variable at each time step
for k in range(0, len(t) - 1):
    if abs(yaw_rate_t[k]) > 1e-3:  # If yaw rate is non-zero
        px_gt[k + 1] = px_gt[k] + (v[k] / yaw_rate_t[k]) * (
            np.sin(yaw_gt[k] + yaw_rate_t[k] * dt) - np.sin(yaw_gt[k])
        )
        py_gt[k + 1] = py_gt[k] + (v[k] / yaw_rate_t[k]) * (
            -np.cos(yaw_gt[k] + yaw_rate_t[k] * dt) + np.cos(yaw_gt[k])
        )
    else:  # If yaw rate is zero
        px_gt[k + 1] = px_gt[k] + v[k] * np.cos(yaw_gt[k]) * dt
        py_gt[k + 1] = py_gt[k] + v[k] * np.sin(yaw_gt[k]) * dt

    yaw_gt[k + 1] = yaw_gt[k] + yaw_rate_t[k] * dt
    yaw_gt[k + 1] = (yaw_gt[k + 1] + np.pi) % (2 * np.pi) - np.pi  # Wrap to [-π, π]

# Plot the path
plt.figure(figsize=(8, 6))
plt.plot(t, v, 'b')
plt.xlabel('P_x')
plt.ylabel('P_y')
plt.title('Figure-Eight Path with Varying Velocity')
plt.grid()
plt.axis('equal')
plt.show()
