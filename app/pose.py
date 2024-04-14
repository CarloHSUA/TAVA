import cv2
import mediapipe as mp
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib
import pandas as pd
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

feature_names = ['x_nose', 'y_nose', 'z_nose', 'x_left_eye_inner', 'y_left_eye_inner',
       'z_left_eye_inner', 'x_left_eye', 'y_left_eye', 'z_left_eye',
       'x_left_eye_outer', 'y_left_eye_outer', 'z_left_eye_outer',
       'x_right_eye_inner', 'y_right_eye_inner', 'z_right_eye_inner',
       'x_right_eye', 'y_right_eye', 'z_right_eye', 'x_right_eye_outer',
       'y_right_eye_outer', 'z_right_eye_outer', 'x_left_ear', 'y_left_ear',
       'z_left_ear', 'x_right_ear', 'y_right_ear', 'z_right_ear',
       'x_mouth_left', 'y_mouth_left', 'z_mouth_left', 'x_mouth_right',
       'y_mouth_right', 'z_mouth_right', 'x_left_shoulder', 'y_left_shoulder',
       'z_left_shoulder', 'x_right_shoulder', 'y_right_shoulder',
       'z_right_shoulder', 'x_left_elbow', 'y_left_elbow', 'z_left_elbow',
       'x_right_elbow', 'y_right_elbow', 'z_right_elbow', 'x_left_wrist',
       'y_left_wrist', 'z_left_wrist', 'x_right_wrist', 'y_right_wrist',
       'z_right_wrist', 'x_left_pinky_1', 'y_left_pinky_1', 'z_left_pinky_1',
       'x_right_pinky_1', 'y_right_pinky_1', 'z_right_pinky_1',
       'x_left_index_1', 'y_left_index_1', 'z_left_index_1', 'x_right_index_1',
       'y_right_index_1', 'z_right_index_1', 'x_left_thumb_2',
       'y_left_thumb_2', 'z_left_thumb_2', 'x_right_thumb_2',
       'y_right_thumb_2', 'z_right_thumb_2', 'x_left_hip', 'y_left_hip',
       'z_left_hip', 'x_right_hip', 'y_right_hip', 'z_right_hip',
       'x_left_knee', 'y_left_knee', 'z_left_knee', 'x_right_knee',
       'y_right_knee', 'z_right_knee', 'x_left_ankle', 'y_left_ankle',
       'z_left_ankle', 'x_right_ankle', 'y_right_ankle', 'z_right_ankle',
       'x_left_heel', 'y_left_heel', 'z_left_heel', 'x_right_heel',
       'y_right_heel', 'z_right_heel', 'x_left_foot_index',
       'y_left_foot_index', 'z_left_foot_index', 'x_right_foot_index',
       'y_right_foot_index', 'z_right_foot_index']

modelo_cargado = joblib.load('log_reg.pkl')
# For webcam input:
cap = cv2.VideoCapture(0)
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    inputs = list()
    try:
        for i in range(len(results.pose_landmarks.landmark)):
            inputs.append(results.pose_landmarks.landmark[i].x)
            inputs.append(results.pose_landmarks.landmark[i].y)
            inputs.append(results.pose_landmarks.landmark[i].z)
    except:
        pass
    
    output = modelo_cargado.predict(pd.DataFrame([inputs], columns=feature_names))
    print(output)
    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == ord('q'):
      break
cap.release()
