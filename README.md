# Road-Object Tracking using Lidar and Radar Fusion

## Overview
This project implements a sensor fusion approach using Lidar and Radar data to enhance road-object tracking in Advanced Driver Assistance Systems (ADAS). The fusion algorithm is built on an Extended Kalman Filter (EKF), leveraging the strengths of both sensor types to achieve accurate position, velocity, and orientation estimations.

## Features
- **Sensor Fusion**: Combines Lidar's positional accuracy and Radar's velocity data using EKF.
- **Simulation Environment**: Python-based simulation of object tracking in a dynamic environment.
- **Performance Metrics**: Evaluation using Root Mean Square Error (RMSE) to validate the accuracy of state estimations.
- **Visualization**: Real-time plots for trajectory, velocity, yaw angle, and yaw rate.

## System Architecture
The system uses the following components:
- **State Variables**:
  - Position: \( P_x, P_y \)
  - Velocity: \( v \)
  - Yaw Angle: \( phi \)
  - Yaw Rate: \( phi_dot \)
- **Motion Model**: Predicts the object's next state based on the current state and motion dynamics.
- **Measurement Model**: Processes Lidar and Radar measurements for state updates.

### Block Diagram
![Screenshot 2024-12-10 193630](https://github.com/user-attachments/assets/c46980d1-2e5d-4aac-9eb9-39eed7ea2581)


## Implementation Details
### Extended Kalman Filter
1. **Initialization**:
   - State vector: \( x_0 \)
   - Covariance matrix: \( \Sigma_0 \)
2. **Prediction Step**:
   - Predicts the next state using the motion model.
3. **Update Step**:
   - Incorporates Lidar or Radar measurements to refine predictions.

### Noise Parameters
| Parameter                  | Value      |
|----------------------------|------------|
| Longitudinal Acceleration  | 2.0 \( m/s^2 \) |
| Yaw Acceleration           | 0.9 \( rad/s^2 \) |
| Lidar Position Noise (X, Y)| 0.1 \( m \) |
| Radar Velocity Noise       | 0.05 \( m/s \) |

### Performance Metrics
RMSE values for state variables after tuning:
![image](https://github.com/user-attachments/assets/a381b8f3-437e-4f35-a93d-e72846456196)


## Results and Visualizations
- Fusion of Lidar and Radar reduces RMSE across all state variables compared to using individual sensors.
- Example simulations:
  - **Lidar Only**: Significant deviations in velocity estimation.
  ![lidar_only](https://github.com/user-attachments/assets/32375bfe-4c52-407c-b374-49a4d6d3c04f)
  - **Radar Only**: Large positional inaccuracies.
  ![radar_only](https://github.com/user-attachments/assets/efe29eae-6ab0-480e-9d00-bbd5effbb32e) 
  - **Fusion**: Balanced accuracy in position and velocity.
  ![Lidar and radar](https://github.com/user-attachments/assets/04786506-ae76-408f-aad9-f04bdfd45771)


## Software and Tools
- **Programming Language**: Python
- **Libraries**: NumPy (matrix computations), Matplotlib (visualization)
- **Algorithms**: Extended Kalman Filter


## Future Work
- Optimization of noise parameters for improved accuracy.
- Integration with real-world datasets for validation.
- Extending the framework to include additional sensor types (e.g., cameras).

## References
- Farag, W., "Road-object tracking for autonomous driving using Lidar and Radar fusion," *Journal of Electrical Engineering*, 2020.
- Chavez-Garcia, R. O., & Aycard, O., "Multiple Sensor Fusion and Classification for Moving Object Detection and Tracking," *IEEE Transactions on Intelligent Transportation Systems*.

---
Feel free to contribute by submitting issues or pull requests!
