# Object Tracking Project with Kalman Filter Integration

## Overview
This project enhances an existing object tracking system, which utilizes Intersection over Union (IoU) and the Hungarian algorithm, by integrating Kalman Filters. This integration significantly improves the tracking accuracy, especially in predicting the future positions of objects.

## Features
- **Kalman Filter for Predictive Tracking**: Integrates Kalman Filters to predict the trajectory of each object, providing more accurate tracking, especially in situations where objects move unpredictably or get occluded.
- **Enhanced Trajectory Visualization**: The system not only tracks objects in real-time but also predicts and visualizes their future positions, offering a clearer understanding of object movement patterns.
- **Robustness in Complex Scenarios**: The use of Kalman Filters makes the tracking system more robust to variations in object motion and temporary occlusions.
- **Looking in forward frames**: The system can look forward in time to predict the future positions of objects, which is useful for planning and decision-making in autonomous driving.



## Results
-----
### Looking forward without resnet model
#### Parameters for tracking algorithm
CONF_THRESH: 40.0  
sigma: 0.35  
nb_step: 10  

-> fps: 365.74  
-> nombre d'id: 122  

------
### Looking forward with resnet model
#### Parameters for tracking algorithm
CONF_THRESH: 40.0  
sigma: 0.5  
nb_step: 6  

-> fps: 6.24  
-> nombre d'id: 106  


-----
### Resnet Model (TP5)
sigma = 0.5  
CONF_THRESH = 40.0  

-> fps: 5.38  
-> nombre d'id: 188  

-----
### Without resnet model (tp4)
sigma = 0.35  
CONF_THRESH = 40.0  

-> fps: 2453.73  
-> nombre d'id: 240  



## Config

If you want to test the code you may change the config inside `inputs.yaml`  

To launch the code: `python3 final.py inputs.yaml`  

## Dependencies

Inside the `requirements.txt` file you will find all the dependencies needed to run the code.

## Results

You can find the track text file and the video inside the `data` folder after running the code.


## Authors
- [**Hugo DEPLAGNE**]