import pandas as pd
import numpy as np
import cv2
import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms

from scipy.optimize import linear_sum_assignment
from KalmanFilter import KalmanFilter
from copy import deepcopy
from PIL import Image

#### Global variables ####

next_track_id = 1
resnet_model = None
preprocess = None
deep_similarities = []

#### Model to extract visual features from the detections ####

def init_model():
    global resnet_model, preprocess
    # Load pre-trained ResNet
    resnet_model = models.resnet18(pretrained=True)
    resnet_model.eval()  # Set to evaluation mode

    # Define preprocessing for input images
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def extract_deep_features(img, bbox):
    x, y, w, h = bbox
    crop_img = img[int(y):int(y+h), int(x):int(x+w)]
    if crop_img.size == 0:
        return torch.zeros((1, 1000))  # Size depends on the model's output features
    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    crop_img = Image.fromarray(crop_img)
    input_tensor = preprocess(crop_img)
    input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

    with torch.no_grad():
        output = resnet_model(input_batch)
    return output

#### IOU matching ####

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

def compute_similarity(iou_score, deep_features1, deep_features2):
    """
        Compute the similarity between two detections.
    """
    deep_similarity = torch.nn.functional.cosine_similarity(deep_features1, deep_features2).item()

    deep_similarities.append(iou_score * deep_similarity)
    return iou_score * deep_similarity


def create_similarity_matrix(current_detections, previous_tracks, frame):
    """
        Create the similarity matrix between the current detections and the previous tracks.
        Parameters:
            current_detections: list of current detections
            previous_tracks: list of previous tracks
    """
    similarity_matrix = np.zeros((len(current_detections), len(previous_tracks)))
    for i, current_det in enumerate(current_detections):
        # Extract features for the current detection
        deep_features1 = extract_deep_features(frame, current_det)

        for j, previous_track in enumerate(previous_tracks):
            predicted_box = get_predicted_box(previous_track['kf'], current_det[2], current_det[3])
            iou_score = track_iou(current_det, predicted_box)

            # Use the stored features from the track
            deep_features2 = previous_track['deep_features']

            similarity_score = compute_similarity(iou_score, deep_features1, deep_features2)
            similarity_matrix[i][j] = similarity_score
    return similarity_matrix

def associate_detections_to_tracks(similarity_matrix, sigma=0.5):
    """
        Associate the detections to the tracks.
        Parameters:
            similarity_matrix: similarity matrix between the current detections and the previous tracks            sigma_iou: IoU threshold
            sigma: Similarity threshold
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
        if similarity_matrix[det_idx, track_idx] >= sigma:
            matches.append((det_idx, track_idx))
            unmatched_detections.remove(det_idx)
            unmatched_tracks.remove(track_idx)

    return matches, unmatched_detections, unmatched_tracks

def initialize_new_track(det, conf, frame_number, frame):
    """
        Initialize a new track.
        Parameters:
            det: detection
            conf: confidence of the detection
            frame_number: current frame number
            frame: current frame
    """
    global next_track_id
    kf = KalmanFilter()
    kf.x = np.matrix([[det[0] + det[2] / 2], [det[1] + det[3] / 2], [0], [0]])

    # Extract deep features and color histogram for the initial detection
    deep_features = extract_deep_features(frame, det)
    new_track = {
        'id': next_track_id,
        'box': det,
        'conf': conf,
        'frames': [frame_number],
        'positions': [(int(det[0] + det[2] / 2), int(det[1] + det[3] / 2))],
        'kf': kf,
        'deep_features': deep_features
    }
    next_track_id += 1
    return new_track

def update_track(track, det, conf, frame_number, frame):
    # Update Kalman Filter with the new detection
    track['kf'].update([det[0] + det[2] / 2, det[1] + det[3] / 2])
    # Update the track's state with the Kalman Filter's state
    estimated_pos = track['kf'].x[:2]
    track['box'] = [estimated_pos[0, 0] - det[2] / 2, estimated_pos[1, 0] - det[3] / 2, det[2], det[3]]
    track['frames'].append(frame_number)
    track['positions'].append((int(estimated_pos[0, 0]), int(estimated_pos[1, 0])))
    track['conf'] = conf

    # Update deep features for the track
    track['deep_features'] = extract_deep_features(frame, track['box'])

def update_tracks(matches,
                  unmatched_tracks,
                  unmatched_detections,
                  current_detections,
                  current_confidences,
                  previous_tracks,
                  frame_number,
                    frame):
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
            frame: current frame
    """
    # Update matched tracks with new detections
    for det_index, track_index in matches:
        det = current_detections[det_index]
        conf = current_confidences[det_index]
        update_track(previous_tracks[track_index], det, conf, frame_number, frame)

    # Handle unmatched detections (new tracks)
    for det_index in unmatched_detections:
        det = current_detections[det_index]
        conf = current_confidences[det_index]
        new_track = initialize_new_track(det, conf, frame_number, frame)
        previous_tracks.append(new_track)

    # Remove unmatched tracks
    previous_tracks[:] = [track for index, track in enumerate(previous_tracks) if index not in unmatched_tracks]

    return previous_tracks

#### Forward tracking ####

def look_forward(det_df,
                 tracks_history,
                 frame_number,
                 frame,
                 frames_path,
                 nb_step = 10,
                 sigma = 0.5,
                 conf_threshold = 20.0):
    """
        Perform tracking on the given detection file, the goal of looking forward
        is to predict the position of the detected objects in the future and keep
        track of its ID so that it is not replaced by another.

        Parameters:
            det_df: DataFrame containing the detections
            tracks_history: dictionnary with as key the frame number and as value the list of tracks
            frame_number: current frame number
            frame: current frame
            frames_path: path to the folder containing the frames
            nb_step: number of steps to look forward
            sigma_iou: IoU threshold
            conf_threshold: confidence threshold for the detections
    """
    if nb_step < 1:
        raise ValueError('nb_step must be greater than 0')

    if frame_number == 1:
        # Loop over the first nb_step images
        for i, frame_filename in enumerate(sorted(os.listdir(frames_path))[:nb_step], start=1):
            frame_path = os.path.join(frames_path, frame_filename)
            frame = cv2.imread(frame_path)

            # Get the detections for the current frame
            current_detections = det_df[det_df['frame'] == i]
            current_detections = current_detections[current_detections['conf'] >= conf_threshold]
            current_boxes = current_detections[['bb_left', 'bb_top', 'bb_width', 'bb_height']].values
            current_confidences = current_detections['conf'].values

            if i == frame_number:
                tracks_history[i] = []
                for det, conf in zip(current_boxes, current_confidences):
                    new_track = initialize_new_track(det, conf, i, frame)
                    tracks_history[i].append(new_track)
            else:
                for tracks_idx in range(frame_number, i):
                    for track in tracks_history[tracks_idx]:
                        track['kf'].predict()
                
                tracks = deepcopy(tracks_history[i - 1])
                for j in range(frame_number, i):
                    for track in tracks_history[j]:
                        tracks.append(track)
                
                similarity_matrix = create_similarity_matrix(current_boxes, tracks, frame)
                matches, unmatched_detections, unmatched_tracks = associate_detections_to_tracks(similarity_matrix, sigma)
                tracks_history[i] = update_tracks(matches,
                                                  unmatched_tracks,
                                                  unmatched_detections,
                                                  current_boxes,
                                                  current_confidences,
                                                  tracks,
                                                  i,
                                                  frame)

    else:
        # Current detections from last frame in the history
        current_detections = det_df[det_df['frame'] == frame_number + nb_step - 1]
        current_detections = current_detections[current_detections['conf'] >= conf_threshold]
        current_boxes = current_detections[['bb_left', 'bb_top', 'bb_width', 'bb_height']].values
        current_confidences = current_detections['conf'].values

        last_track_idx = frame_number + nb_step - 2
        for tracks_idx in range(frame_number, last_track_idx):
            for track in tracks_history[tracks_idx]:
                track['kf'].predict()

        tracks = deepcopy(tracks_history[last_track_idx])     
        for j in range(frame_number, frame_number + nb_step - 1):
            for track in tracks_history[j]:
                tracks.append(track)

        similarity_matrix = create_similarity_matrix(current_boxes, tracks, frame)
        matches, unmatched_detections, unmatched_tracks = associate_detections_to_tracks(similarity_matrix, sigma)
        tracks_history[frame_number + nb_step - 1] = update_tracks(matches,
                                                                   unmatched_tracks,
                                                                   unmatched_detections,
                                                                   current_boxes,
                                                                   current_confidences,
                                                                   tracks,
                                                                   frame_number + nb_step - 1,
                                                                   frame)
    print(np.mean(deep_similarities))
    return tracks_history