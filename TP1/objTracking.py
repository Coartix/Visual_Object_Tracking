from KalmanFilter import KalmanFilter
from Detector import detect
import cv2
import sys
from random import randint
import numpy as np


square_size = 30  # Length of the square's side
square_color = (0, 0, 255)  # Red color
thickness = 2  # Thickness of the square's outline
transparency = 0.5  # Transparency of the square

def add_transparent_square(image, center):
    overlay = image.copy()
    top_left = (center[0] - square_size // 2, center[1] - square_size // 2)
    bottom_right = (center[0] + square_size // 2, center[1] + square_size // 2)
    cv2.rectangle(overlay, top_left, bottom_right, square_color, -1)

    # Blend the overlay with the original image
    return cv2.addWeighted(overlay, transparency, image, 1 - transparency, 0)

# Create main function that takes the path to the video as input
def main(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    delay = int(1000 / fps)

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_with_detections.mp4', fourcc, fps, (width, height))

    # Kalman Filter
    kf = KalmanFilter(dt=0.1, u_x=0, u_y=0, std_acc=1, x_sdt_meas=0.1, y_sdt_meas=0.1)

    # First frame
    success, frame = cap.read()
    if not success:
        print('Failed to read video')
        sys.exit(1)
    center = detect(frame)
    center = center[0]
    center = tuple(center.astype(int).ravel())
    cv2.circle(frame, center, 5, (0, 255, 0), -1)
    cv2.imshow('frame', frame)

    # Initialize the state
    kf.x = np.matrix([[center[0]], [center[1]], [0], [0]])
    predicted_center = tuple(int(val) for val in kf.x[:2, 0].A1)
    cv2.circle(frame, predicted_center, 5, (0, 0, 255), -1)
    cv2.imshow('frame', frame)
    out.write(frame)

    trajectory = []
    # For each frame, detect the objects and predict the next state then update the state
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print('Failed to read video')
            sys.exit(1)
        center = detect(frame)
        if len(center) > 1:
            print('More than one object detected')
            sys.exit(1)
        center = center[0]
        center = tuple(center.astype(int).ravel())
        cv2.circle(frame, center, 5, (0, 255, 0), -1)

        kf.predict()
        predicted_center = tuple(int(val) for val in kf.x[:2, 0].A1)
        top_left_predicted = (predicted_center[0] - square_size // 2, predicted_center[1] - square_size // 2)
        bottom_right_predicted = (predicted_center[0] + square_size // 2, predicted_center[1] + square_size // 2)
        # Draw a blue rectangle for the predicted position
        cv2.rectangle(frame, top_left_predicted, bottom_right_predicted, (255, 0, 0), thickness)
        
        
        kf.update(center)
        estimated_center = tuple(int(val) for val in kf.x[:2, 0].A1)
        top_left_estimated = (estimated_center[0] - square_size // 2, estimated_center[1] - square_size // 2)
        bottom_right_estimated = (estimated_center[0] + square_size // 2, estimated_center[1] + square_size // 2)

        # Draw a red rectangle for the estimated position
        cv2.rectangle(frame, top_left_estimated, bottom_right_estimated, (0, 0, 255), thickness)
        
        # Draw the trajectory
        trajectory.append(estimated_center)
        for i in range(1, len(trajectory)):
            cv2.line(frame, trajectory[i - 1], trajectory[i], (0, 255, 255), 2)
        
        out.write(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

        # Detect end of video
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            print('End of video')
            break
        
    # Release the video capture object
    cap.release()
    # Close all windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python objTracking.py <video_path>')
        sys.exit(1)

    video_path = sys.argv[1]
    if video_path is None:
        print('No video path supplied')
        sys.exit(0)
    if not isinstance(video_path, str):
        print('Video path is not a string')
        sys.exit(0)
    if not video_path.endswith('.mp4') and not video_path.endswith('.avi'):
        print('Video path is not a path to a video file')
        sys.exit(0)

    main(video_path)
