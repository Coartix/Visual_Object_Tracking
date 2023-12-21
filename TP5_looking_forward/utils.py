import pandas as pd
import os
import numpy as np
import cv2


def draw_tracking_result(frame, tracks):
    """
        Draw the tracking result on the frame.
        Parameters:
            frame: current frame
            tracks: list of tracks
    """
    for track in tracks:
        # Draw bounding box and ID
        x, y, w, h = track['box']
        conf = track['conf']
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 255), 2)
        label = f'ID: {track["id"]}, Conf: {conf:.2f}'
        cv2.putText(frame, label, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)

        # Draw tracking path
        for i in range(1, len(track['positions'])):
            cv2.line(frame, track['positions'][i - 1], track['positions'][i], (0, 255, 0), 2)
    
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
            conf = track['conf']
            id = track['id']
            file.write(f'{frame_number},{id},{x},{y},{w},{h},{conf},-1,-1,-1\n')