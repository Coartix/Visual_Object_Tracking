import pandas as pd
import os
import numpy as np
import cv2
import yaml
import sys
import time

from utils import draw_tracking_result, save_tracking_results, load_detections
from tracks import track_iou, create_similarity_matrix, associate_detections_to_tracks, update_tracks, initialize_new_track, look_forward, init_model
from KalmanFilter import KalmanFilter

def read_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main(config_path):
    # Load configuration
    config = read_config(config_path)

    # Initialize tracking variables
    tracks_history = {}
    frame_number = 1

    # Initialize model for visual features
    init_model(config['use_resnet'])

    # Load detections
    detections_df = load_detections(config['det_file_path'])

    # Prepare output video writer
    fourcc = cv2.VideoWriter_fourcc(*config['fourcc'])
    out = cv2.VideoWriter(config['output_video'], fourcc, config['fps'], tuple(config['frame_size']))
    delay = int(1000 / config['fps'])

    if os.path.isfile(config['tracking_file']):
        open(config['tracking_file'], 'w').close()
    else:
        open(config['tracking_file'], 'x')

    total_time = 0  # Variable to store total processing time
    total_frames = 0  # Variable to store total number of frames processed

    #### Process each image frame ####
    for filename in sorted(os.listdir(config['image_frames_path'])):
        if filename.endswith(".jpg"):
            frame_path = os.path.join(config['image_frames_path'], filename)
            frame = cv2.imread(frame_path)

            start_time = time.time()

            tracks_history = look_forward(
                detections_df,
                tracks_history,
                frame_number,
                frame,
                config['image_frames_path'],
                nb_step=config['nb_step'],
                sigma=config['sigma'],
                conf_threshold=config['CONF_THRESH']
            )

            end_time = time.time()
            frame_time = end_time - start_time
            total_time += frame_time
            total_frames += 1

            frame_with_tracking = draw_tracking_result(frame, tracks_history[frame_number])
            save_tracking_results(tracks_history[frame_number], config['tracking_file'], frame_number)

            cv2.imshow('Tracking', frame_with_tracking)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break
            out.write(frame_with_tracking)

            frame_number += 1

    # Release resources
    out.release()
    cv2.destroyAllWindows()

    if total_time > 0:
        mean_fps = total_frames / total_time
        print("Mean FPS over the entire computation: {:.2f}".format(mean_fps))
    else:
        print("No frames were processed.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python final.py <path_to_config.yaml>")
        sys.exit(1)
    main(sys.argv[1])