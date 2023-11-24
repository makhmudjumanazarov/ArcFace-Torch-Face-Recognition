import time
import cv2
import numpy as np
import onnxruntime
# from imread_from_url import imread_from_url
from YOLOv8 import *
import os
import torch
# from backbones import get_model

from utils import xywh2xyxy, draw_detections, multiclass_nms

def face_detection(img_path):
    img = cv2.imread(img_path)
    boxes, scores, class_ids = yolov8_detector.detect_objects(img)
    return boxes, scores, class_ids, img

def preprocess_face(face_img):
    if face_img is None or face_img.size == 0:
        # Return None to indicate that the face is not valid
        return None
    
    face_img = cv2.resize(face_img, (112, 112))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = np.transpose(face_img, (2, 0, 1))
    face_img = torch.from_numpy(face_img).unsqueeze(0).float()
    face_img.div_(255).sub_(0.5).div_(0.5)
    return face_img

def database_loader(database_folder, net):
    database = {}

    for person_name in os.listdir(database_folder):
        person_path = os.path.join(database_folder, person_name)
        
        if os.path.isdir(person_path):
            person_images = []
            for image_file in os.listdir(person_path):
                image_path = os.path.join(person_path, image_file)
                img = cv2.imread(image_path)
                # Perform face detection
                faces, _ = face_detection(image_path)

                if faces.any():
                    # Use only the first detected face
                    bbox = faces[0]
                    x_min, y_min, x_max, y_max, confidence = bbox
                    
                    # You can then use these values as needed, for example:
                    face_img = img[int(y_min):int(y_max), int(x_min):int(x_max)]
                    face_tensor = preprocess_face(face_img)

                    if face_tensor is not None:
                        # Perform inference to get the feature vector
                        with torch.no_grad():
                            feature_vector = net(face_tensor).numpy()

                        person_images.append(feature_vector)

            # Average the feature vectors for each person
            if person_images:
                average_feature_vector = np.mean(person_images, axis=0)
                database[person_name] = average_feature_vector

    return database


model_path = "/home/airi/Makhmud/Google-Cloud-CPU-ONNX-Deploy-via-ngrok-with-streamlit-face-detecion/best.onnx"

# Initialize YOLOv8 object detector
yolov8_detector = YOLOv8(model_path, conf_thres=0.3, iou_thres=0.5)

# weight = '/home/airi/Makhmud/insightface/recognition/arcface_torch/work_dirs/ms1mv3_r50/model.pt'
# name = 'r50'

database_folder = '/home/airi/Makhmud/insightface/recognition/arcface_torch/data/dataset'

database = {}
list_soni = 0
numpy_soni = 0
for person_name in os.listdir(database_folder):
    person_path = os.path.join(database_folder, person_name)
    
    if os.path.isdir(person_path):
        person_images = []
        for image_file in os.listdir(person_path):
            image_path = os.path.join(person_path, image_file)
            img = cv2.imread(image_path)

            # Perform face detection
            faces, scores, class_ids, img = face_detection(image_path)

            if not isinstance(faces, list) and faces.shape != (2, 4):
                if faces.any():
                    # Use only the first detected face
                    bbox = faces[0]
                    x_min, y_min, x_max, y_max = bbox
                    # You can then use these values as needed, for example:
                    face_img = img[int(y_min):int(y_max), int(x_min):int(x_max)]
                    face_tensor = preprocess_face(face_img)

                

               

                    

