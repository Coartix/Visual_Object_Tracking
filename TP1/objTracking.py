from KalmanFilter import KalmanFilter
from Detector import detect
import cv2
import sys
from random import randint
import numpy as np

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
        kf.update(center)
        predicted_center = tuple(int(val) for val in kf.x[:2, 0].A1)
        cv2.circle(frame, predicted_center, 5, (0, 0, 255), -1)
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
