import os 
import cv2
import numpy as np


######################################################################################################
# Loadind data

database_folder = '/home/airi/Makhmud/insightface/recognition/arcface_torch/ishxona'

database = {}

for person_name in os.listdir(database_folder):
    person_path = os.path.join(database_folder, person_name)
    
    if os.path.isdir(person_path):
        person_images = []
        for image_file in os.listdir(person_path):
            image_path = os.path.join(person_path, image_file)
        
            if image_path:
                person_images.append(np.load(image_path))

        # Average the feature vectors for each person
        if person_images:
            average_feature_vector = np.mean(person_images, axis=0)
            database[person_name] = average_feature_vector


######################################################################################################################
import os
import os.path as osp
import cv2
import numpy as np
import torch
from backbones import get_model
# import dlib
import argparse
import onnxruntime
from scrfd import SCRFD
from arcface_onnx import ArcFaceONNX
import re

onnxruntime.set_default_logger_severity(3)

assets_dir = osp.expanduser('~/.insightface/models/buffalo_l')

detector = SCRFD(os.path.join(assets_dir, 'det_10g.onnx'))
detector.prepare(0)

weight = '/home/airi/Makhmud/insightface/recognition/arcface_torch/work_dirs/ms1mv3_r50/model.pt'
name = 'r50'

def face_detection(img_path):
    img = cv2.imread(img_path)
    bboxes1, kpss1 = detector.autodetect(img, max_num=10)
    return bboxes1, img

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

database_folder = '/home/airi/Makhmud/insightface/recognition/arcface_torch/dataset'

def load_recognition(weight, name):

    # Load pre-trained model
    net = get_model(name, fp16=False)
    net.load_state_dict(torch.load(weight))
    net.eval()

    return net

net = load_recognition(weight, name)

# test = database_loader(database_folder, recognition_model)

database = {}

for person_name in os.listdir(database_folder):
    person_path = os.path.join(database_folder, person_name)
    
    if os.path.isdir(person_path):
        person_images = []
        
        # Create a folder for each person
        output_folder = os.path.join('ishxona', person_name)
        os.makedirs(output_folder, exist_ok=True)

        for image_file in os.listdir(person_path):
            image_path = os.path.join(person_path, image_file)

            # Use regex to extract the name from the file name
            match = re.search(r'([^/]+)-(\d+)\.\w+', image_file)
            if match:
                extracted_name = match.group(1) + '-' + match.group(2)
            
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
                        feature_vector = net(face_tensor)
                        
                        # Save the feature vector to a .npy file in the person's folder
                        np.save(os.path.join(output_folder, f'{extracted_name}.npy'), feature_vector)

        # # Average the feature vectors for each person
        # if person_images:
        #     # average_feature_vector = np.mean(person_images, axis=0)
        #     database[person_name] = person_images
# feat2 = np.load('/home/airi/Makhmud/insightface/recognition/arcface_torch/ishxona/Foziljon.npy')
