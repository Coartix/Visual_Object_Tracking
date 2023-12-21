# Object Tracking Project with Kalman Filter Integration

## Overview
This project enhances an existing object tracking system, which utilizes Intersection over Union (IoU) and the Hungarian algorithm, by integrating Kalman Filters. This integration significantly improves the tracking accuracy, especially in predicting the future positions of objects.

## Features
- **Kalman Filter for Predictive Tracking**: Integrates Kalman Filters to predict the trajectory of each object, providing more accurate tracking, especially in situations where objects move unpredictably or get occluded.
- **Enhanced Trajectory Visualization**: The system not only tracks objects in real-time but also predicts and visualizes their future positions, offering a clearer understanding of object movement patterns.
- **Robustness in Complex Scenarios**: The use of Kalman Filters makes the tracking system more robust to variations in object motion and temporary occlusions.
- **Looking in forward frames**: The system can look forward in time to predict the future positions of objects, which is useful for planning and decision-making in autonomous driving.
