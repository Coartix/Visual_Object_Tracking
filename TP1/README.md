# Object Tracking with Kalman Filter

## Overview

This project implements object tracking using a Kalman Filter, integrated with OpenCV for processing video streams. It tracks the motion of an object in a video, predicts its future position, and corrects these predictions based on new observations.

## Features

- **Real-Time Object Detection**: A green circle is drawn around the detected object in each frame, indicating its current position.
- **Predicted Position**: A blue rectangle marks the predicted next position of the object, as calculated by the Kalman Filter before the measurement update.
- **Estimated Position**: A red rectangle indicates the estimated position of the object, updated by the Kalman Filter after measurement.
- **Trajectory Tracking**: The path of the object is traced in yellow, showing the movement trajectory over time.

## Prerequisites

To run this project, the following are required:
- Python 3.10 - 3.11
- OpenCV (cv2) for video processing.
- NumPy for numerical calculations.

## Usage

Run the object tracking system with the following command:

```bash
python objTracking.py <path_to_video>
