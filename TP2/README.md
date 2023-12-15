# Object Tracking Project

## Overview
This project implements an object tracking system using Intersection over Union (IoU) based tracking. It processes a sequence of images, tracks objects based on their bounding boxes, and visualizes their movement over time.

## Features
- **IoU-Based Tracking**: Tracks objects across frames using the IoU metric.
- **Unique ID Assignment**: Assigns unique IDs to each tracked object for easy identification.
- **Confidence Score Display**: Shows the confidence score for each detection.
- **Trajectory Visualization**: Draws the trajectory of each object, showing its path through the sequence of frames.
- **Video Output**: Compiles the processed frames into a video with tracking information.

## Prerequisites
- Python 3.x
- OpenCV
- NumPy
- Pandas   


## Usage
1. Place your sequence of images in a directory (e.g., `data/img1/`).
2. Ensure you have a detections file (`det.txt`) with the format as specified.
3. Run the main script to process the images and track objects  


4. The script will display the processed frames and save the output as a video file.

## Input Data Format
The detections file (`det.txt`) should contain the following columns:
- `frame`: Frame number.
- `id`: Object ID (used for the first frame's initialization).
- `bb_left`, `bb_top`, `bb_width`, `bb_height`: Bounding box coordinates.
- `conf`: Detection confidence score.
- `x`, `y`, `z`: World coordinates (not used in this 2D challenge).

## Output
The output is a video file (`output_with_tracking.mp4`) showing the tracked objects with their IDs, confidence scores, bounding boxes, and trajectories.

