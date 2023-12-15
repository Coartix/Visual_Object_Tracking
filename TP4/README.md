# Object Tracking Project

## Overview
This project implements an advanced object tracking system using Intersection over Union (IoU) and the Hungarian algorithm. It processes a sequence of images, precisely tracks objects based on their bounding boxes, and visualizes their movement over time.

## Features
- **IoU-Based Tracking with Hungarian Algorithm**: Enhances tracking accuracy by optimally associating detections to tracks using the Hungarian algorithm, which improves assignment efficiency in complex scenarios.
- **Unique ID Assignment**: Maintains consistent IDs for each tracked object, even in challenging situations like occlusions or closely moving objects.
- **Confidence Score Display**: Shows the detection confidence score for each object, providing insight into the detection reliability.
- **Trajectory Visualization**: Draws the path of each object, illustrating its movement throughout the sequence of frames.
- **Video Output with Enhanced Tracking**: Generates a video that showcases the advanced tracking capabilities with annotations.

## Prerequisites
- Python 3.x
- OpenCV
- NumPy
- Pandas
- SciPy (for the Hungarian algorithm)

## Usage
1. Place your sequence of images in a directory (e.g., `data/img1/`).
2. Prepare a detections file (`det.txt`) in the specified format.
3. Run the main script to process the images and apply the enhanced tracking algorithm.
4. View the real-time tracking results and find the generated video file (`output_with_tracking.mp4`).

## Input Data Format
The detections file (`det.txt`) should include:
- `frame`: Frame number.
- `id`: Object ID (initial frame initialization).
- `bb_left`, `bb_top`, `bb_width`, `bb_height`: Bounding box coordinates.
- `conf`: Detection confidence score.
- `x`, `y`, `z`: World coordinates (unused in this 2D challenge).

## Output
- **Video File**: `output_with_tracking.mp4`, displaying tracked objects, IDs, confidence scores, bounding boxes, and trajectories.
- **Tracking File**: A text file containing tracking results in a format similar to the ground truth, updated with unique IDs assigned to each track, enhancing data analysis and verification.

# Changes from TP2  

## Enhancements with the Hungarian Algorithm
The integration of the Hungarian algorithm significantly improves the tracking accuracy, especially in scenes with multiple moving objects. By optimizing detection-to-track associations, it reduces ID switch errors and enhances the overall robustness of the tracking system.

## Tracking File for Analysis
The system generates a detailed tracking file, which is invaluable for post-analysis, allowing for a thorough examination of the tracking performance and object behavior throughout the video.
