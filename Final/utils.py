import pandas as pd
import os
import numpy as np
import cv2
import random

def get_color_by_id(track_id):
    random.seed(track_id)
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def load_detections(det_file_path):
    """
        Load the detections from the detection file into a Pandas DataFrame.
    """
    # Define the column names for the DataFrame
    columns = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z']
    
    # Read the detection file into a DataFrame
    df = pd.read_csv(det_file_path, header=None, names=columns)
    df['conf'] = df['conf'].astype(float)
    return df

def draw_tracking_result(frame, tracks):
    """
    Draw the tracking result on the frame.
    Parameters:
        frame: current frame
        tracks: list of tracks
    """
    for track in tracks:
        # Generate a distinct color for each track ID
        track_color = get_color_by_id(track['id'])

        # Draw bounding box and ID
        x, y, w, h = track['box']
        similarity_score = track.get('similarity_score', 0)  # Get the stored similarity score
        label = f'ID: {track["id"]}, Score: {similarity_score:.2f}'  # Display the similarity score
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), track_color, 2)
        cv2.putText(frame, label, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, track_color, 2, cv2.LINE_AA)

        # Draw tracking path
        for i in range(1, len(track['positions'])):
            cv2.line(frame, track['positions'][i - 1], track['positions'][i], track_color, 2)
    
    return frame


def save_tracking_results(tracks, output_file_path, frame_number):
    """
        Save the tracking results to the output file.
        Format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
        Parameters:
            tracks: list of tracks
            output_file_path: path to the output file
            frame_number: current frame number
        
    """
    with open(output_file_path, 'a') as file:
        for track in tracks:
            x, y, w, h = track['box']
            conf = 1 #track['conf']
            id = track['id']
            file.write(f'{frame_number},{id},{x},{y},{w},{h},{conf},-1,-1,-1\n')