import os
import os.path as osp
import cv2
import numpy as np
import onnxruntime
from scrfd import SCRFD
from arcface_onnx import ArcFaceONNX

onnxruntime.set_default_logger_severity(3)

assets_dir = osp.expanduser('~/.insightface/models/buffalo_l')

detector = SCRFD(os.path.join(assets_dir, 'det_10g.onnx'))
detector.prepare(0)
model_path = os.path.join(assets_dir, 'w600k_r50.onnx')
rec = ArcFaceONNX(model_path)
rec.prepare(0)

database_folder = '/home/airi/Makhmud/Face_Recognition/insightface/recognition/arcface_torch/data/dataset'

def database_loader(database_folder):
    database = {}

    for person_name in os.listdir(database_folder):
        person_path = os.path.join(database_folder, person_name)
        
        if os.path.isdir(person_path):
            for image_file in os.listdir(person_path):
                image_path = os.path.join(person_path, image_file)
                img = cv2.imread(image_path)
                # Perform face detection
                bboxes, kpss = detector.autodetect(img , max_num=20)

                if bboxes.shape[0]==0:
                    return -1.0, "Face not found in Image"

                kps = kpss[0]
                feat = rec.get(img, kps)
                database[person_name] =  feat
    
    return database

database = database_loader(database_folder)

def inference(image_path):
    # img = cv2.imread(image_path)
    img = image_path
    faces, kpss = detector.autodetect(img, max_num=20)

    # Preprocess and recognize each detected face
    for face, kps in zip(faces, kpss):

        x_min, y_min, x_max, y_max, confidence = int(face[0]), int(face[1]), int(face[2]), int(face[3]),int(face[4])
        feat = rec.get(img, kps)

        # Calculate similarity with database
        for identity, db_feat in database.items():
            similarity = rec.compute_sim(feat, db_feat)
            
            # Set a similarity threshold (you may need to experiment with this value)
            if similarity > 0.3:
                print(f"Found person {identity} with similarity {similarity}")
                # Draw bounding box around the detected face (replace this with your drawing code)
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
                cv2.putText(img, f"{identity}, {int(similarity*100)}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:    
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
                # cv2.putText(img, f"'unknown'", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img
    # cv2.imwrite("output.jpg", img)
 