# Object Tracking Project with Kalman Filter Integration

## Overview
This project enhances an existing object tracking system, which utilizes Intersection over Union (IoU) and the Hungarian algorithm, by integrating Kalman Filters. This integration significantly improves the tracking accuracy, especially in predicting the future positions of objects.

## Features
- **Kalman Filter for Predictive Tracking**: Integrates Kalman Filters to predict the trajectory of each object, providing more accurate tracking, especially in situations where objects move unpredictably or get occluded.
- **Enhanced Trajectory Visualization**: The system not only tracks objects in real-time but also predicts and visualizes their future positions, offering a clearer understanding of object movement patterns.
- **Robustness in Complex Scenarios**: The use of Kalman Filters makes the tracking system more robust to variations in object motion and temporary occlusions.

## Prerequisites
- Python 3.x
- OpenCV
- NumPy
- Pandas
- SciPy (for the Hungarian algorithm)

## Usage
1. Ensure a sequence of images is placed in a designated directory (e.g., `data/img1/`).
2. Have a detections file (`det.txt`) prepared with the specified format.
3. Run the main script to process the images, with the Kalman Filter enhancing the tracking process.
4. The system will display real-time tracking along with predictions, and output a video file (`output_with_tracking.mp4`).

## Input Data Format
The detections file (`det.txt`) should include the following columns:
- `frame`: Frame number.
- `id`: Object ID (used for initial frame initialization).
- `bb_left`, `bb_top`, `bb_width`, `bb_height`: Bounding box coordinates.
- `conf`: Detection confidence score.

## Output
- **Video File**: `output_with_tracking.mp4`, displaying real-time tracking of objects with predictive paths, IDs, and confidence scores.

## Integration of Kalman Filter
The Kalman Filter integration allows for multi-step prediction of object positions, enhancing the system's capability to handle challenging scenarios where objects move erratically or become temporarily occluded. This predictive approach adds a layer of sophistication to the tracking algorithm, making it more adaptable and accurate.
