import pandas as pd
import os
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from KalmanFilter import KalmanFilter

next_track_id = 1

def load_detections(det_file_path):
    """
        Load the detections from the detection file into a Pandas DataFrame.
    """
    # Define the column names for the DataFrame
    columns = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z']
    
    # Read the detection file into a DataFrame
    df = pd.read_csv(det_file_path, header=None, names=columns)
    return df


def track_iou(box1, box2):
    """
        Compute the IoU between two bounding boxes.
    """
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

def get_predicted_box(kf, w, h):
    """
    Get the predicted bounding box from the Kalman Filter state.
    """
    predicted_state = kf.x
    x_center, y_center = predicted_state[0, 0], predicted_state[1, 0]
    return [x_center - w / 2, y_center - h / 2, w, h]

def create_similarity_matrix(current_detections, previous_tracks):
    similarity_matrix = np.zeros((len(current_detections), len(previous_tracks)))
    for i, current_det in enumerate(current_detections):
        for j, previous_track in enumerate(previous_tracks):
            predicted_box = get_predicted_box(previous_track['kf'], current_det[2], current_det[3])
            similarity_matrix[i][j] = track_iou(current_det, predicted_box)
    return similarity_matrix

def associate_detections_to_tracks(similarity_matrix, sigma_iou):
    """
        Associate the detections to the tracks.
        Parameters:
            similarity_matrix: similarity matrix between the current detections and the previous tracks
            sigma_iou: IoU threshold
    """
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

def initialize_new_track(det, conf, frame_number):
    """
        Initialize a new track.
        Parameters:
            det: detection
            conf: confidence of the detection
            frame_number: current frame number
    """
    global next_track_id
    kf = KalmanFilter()
    kf.x = np.matrix([[det[0] + det[2] / 2], [det[1] + det[3] / 2], [0], [0]])
    new_track = {
        'id': next_track_id,
        'box': det,
        'conf': conf,
        'frames': [frame_number],
        'positions': [(int(det[0] + det[2] / 2), int(det[1] + det[3] / 2))],
        'kf': kf
    }
    next_track_id += 1
    return new_track

def update_track(track, det, conf, frame_number):
    # Update Kalman Filter with the new detection
    track['kf'].update([det[0] + det[2] / 2, det[1] + det[3] / 2])
    # Update the track's state with the Kalman Filter's state
    estimated_pos = track['kf'].x[:2]
    track['box'] = [estimated_pos[0, 0] - det[2] / 2, estimated_pos[1, 0] - det[3] / 2, det[2], det[3]]
    track['frames'].append(frame_number)
    track['positions'].append((int(estimated_pos[0, 0]), int(estimated_pos[1, 0])))
    track['conf'] = conf

def update_tracks(matches, unmatched_tracks, unmatched_detections, current_detections, current_confidences, previous_tracks, frame_number):
    """
        Update the tracks based on the matches and unmatched detections.
        Parameters:
            matches: list of matched detections and tracks
            unmatched_tracks: list of unmatched tracks
            unmatched_detections: list of unmatched detections
            current_detections: list of current detections
            current_confidences: list of confidences for the current detections
            previous_tracks: list of previous tracks
            frame_number: current frame number
    """
    global next_track_id

    # Update matched tracks with new detections
    for det_index, track_index in matches:
        det = current_detections[det_index]
        conf = current_confidences[det_index]
        update_track(previous_tracks[track_index], det, conf, frame_number)

    # Handle unmatched detections (new tracks)
    for det_index in unmatched_detections:
        det = current_detections[det_index]
        conf = current_confidences[det_index]
        new_track = initialize_new_track(det, conf, frame_number)
        previous_tracks.append(new_track)

    # Remove unmatched tracks
    previous_tracks[:] = [track for index, track in enumerate(previous_tracks) if index not in unmatched_tracks]

    return previous_tracks

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