import pandas as pd
import os
import numpy as np
import cv2

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
    matches, unmatched_detections, unmatched_tracks = [], [], []

    # If no detections or no tracks, return empty lists
    if similarity_matrix.size == 0:
        unmatched_detections = list(range(similarity_matrix.shape[0]))
        unmatched_tracks = list(range(similarity_matrix.shape[1]))
        return matches, unmatched_detections, unmatched_tracks
    
    forbidden_idx = []

    for det_idx, row in enumerate(similarity_matrix):
        track_idx = row.argmax()
        # Choose the best iou inside the row while respecting forbidden indices and without changing values inside the row
        max_iou = row[track_idx]
        while len(forbidden_idx) < len(row) and track_idx in forbidden_idx:
            row[track_idx] = -1
            track_idx = row.argmax()
            max_iou = row[track_idx]
        if max_iou > sigma_iou:
            matches.append((det_idx, track_idx))
            forbidden_idx.append(track_idx)
        else:
            unmatched_detections.append(det_idx)

    matched_tracks = [track_idx for _, track_idx in matches]
    unmatched_tracks = [i for i in range(similarity_matrix.shape[1]) if i not in matched_tracks]

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