import pandas as pd
import os
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

next_track_id = 1

def load_detections(det_file_path):
    # Define the column names for the DataFrame
    columns = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z']
    
    # Read the detection file into a DataFrame
    df = pd.read_csv(det_file_path, header=None, names=columns)
    
    return df


def track_iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[0] + box1[2], box2[0] + box2[2])
    y_bottom = min(box1[1] + box1[3], box2[1] + box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou


def create_similarity_matrix(current_detections, previous_tracks):
    similarity_matrix = np.zeros((len(current_detections), len(previous_tracks)))
    for i, current_det in enumerate(current_detections):
        for j, previous_track in enumerate(previous_tracks):
            similarity_matrix[i][j] = track_iou(current_det, previous_track['box'])
    return similarity_matrix

def associate_detections_to_tracks(similarity_matrix, sigma_iou):
    if similarity_matrix.size == 0:
        return [], list(range(similarity_matrix.shape[0])), list(range(similarity_matrix.shape[1]))
    
    # Convert IoU similarity matrix to a cost matrix for the Hungarian algorithm
    cost_matrix = 1 - similarity_matrix
    det_indices, track_indices = linear_sum_assignment(cost_matrix)

    matches = []
    unmatched_detections = list(range(similarity_matrix.shape[0]))
    unmatched_tracks = list(range(similarity_matrix.shape[1]))

    for det_idx, track_idx in zip(det_indices, track_indices):
        if similarity_matrix[det_idx, track_idx] >= sigma_iou:
            matches.append((det_idx, track_idx))
            unmatched_detections.remove(det_idx)
            unmatched_tracks.remove(track_idx)

    return matches, unmatched_detections, unmatched_tracks


def update_tracks(matches, unmatched_tracks, unmatched_detections, current_detections, current_confidences, previous_tracks, frame_number):
    global next_track_id
    # Update matched tracks
    for i, j in matches:
        box = current_detections[i]
        center = (int(box[0] + box[2] / 2), int(box[1] + box[3] / 2))
        previous_tracks[j]['box'] = box
        previous_tracks[j]['conf'] = current_confidences[i]
        previous_tracks[j]['frames'].append(frame_number)
        previous_tracks[j]['positions'].append(center)

    # Add new tracks for unmatched detections
    for i in unmatched_detections:
        new_track = {
            'id': next_track_id,
            'box': current_detections[i],
            'conf': current_confidences[i],  # Store the confidence
            'frames': [frame_number],
            'positions': [(int(current_detections[i][0] + current_detections[i][2] / 2), int(current_detections[i][1] + current_detections[i][3] / 2))]
        }
        previous_tracks.append(new_track)
        next_track_id += 1  # Increment the ID for the next new track

    # Remove unmatched tracks
    for i in reversed(unmatched_tracks):
        del previous_tracks[i]
    return previous_tracks

def draw_tracking_result(frame, tracks):
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
    # Format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
    # Append the tracking results to the output file
    with open(output_file_path, 'a') as file:
        for track in tracks:
            x, y, w, h = track['box']
            conf = track['conf']
            id = track['id']
            file.write(f'{frame_number},{id},{x},{y},{w},{h},{conf},-1,-1,-1\n')