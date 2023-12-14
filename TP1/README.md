# Object Tracking with Kalman Filter

## Overview

This project implements object tracking using a Kalman Filter, integrated with OpenCV for processing video streams. It's designed to track the motion of an object in a video, predicting its future position and correcting these predictions based on new observations.

## What it Shows on Video

- **Detected Object**: A green circle is drawn around the detected object in each frame.
- **Predicted Position**: A red circle indicates the predicted next position of the object, as calculated by the Kalman Filter.

## Prerequisites

To run this project, ensure you have the following installed:
- Python 3.x: The programming language used for the script.
- OpenCV (cv2): A library for computer vision tasks, used for video processing.
- NumPy: A library for numerical operations, used for matrix calculations in the Kalman Filter.

## Usage

1. **Run the Script**: Use the following command to process a video file:
   ```bash
   python objTracking.py <path_to_video>
