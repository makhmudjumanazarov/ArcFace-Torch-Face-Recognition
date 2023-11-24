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

onnxruntime.set_default_logger_severity(3)

assets_dir = osp.expanduser('~/.insightface/models/buffalo_l')

detector = SCRFD(os.path.join(assets_dir, 'det_10g.onnx'))
detector.prepare(0)

weight = '/home/airi/Makhmud/Face_Recognition/ArcFace_Torch/arcface_torch/work_dirs/ms1mv3_r50/model.pt'
name = 'r50'
database_folder = '/home/airi/Makhmud/Face_Recognition/ArcFace_Torch/arcface_torch/data/dataset'

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

def load_recognition(weight, name):

    # Load pre-trained model
    net = get_model(name, fp16=False)
    net.load_state_dict(torch.load(weight))
    net.eval()

    return net

recognition_model = load_recognition(weight, name)

 # Load the database
database = database_loader(database_folder, recognition_model)  

@torch.no_grad()
def inference(img_path):

    # Load the database
    # database = database_loader(database_folder, recognition_model)  

    # Perform face detection
    # faces, img = face_detection(img_path)

    # img = cv2.imread(img_path)
    img = img_path
    faces, kpss1 = detector.autodetect(img_path, max_num=10)
    
    # Preprocess and recognize each detected face
    for face in faces:

        x_min, y_min, x_max, y_max, confidence = int(face[0]), int(face[1]), int(face[2]), int(face[3]),int(face[4])
        
        # You can then use these values as needed, for example:
        face_img = img[int(y_min):int(y_max), int(x_min):int(x_max)]
        face_tensor = preprocess_face(face_img)

        if face_tensor is None:
            return img
        # Inference
        feat = recognition_model(face_tensor).numpy()

        # Calculate similarity with database
        for identity, db_feat in database.items():
            # Calculate cosine similarity
            similarity = np.dot(feat, db_feat.T) / (np.linalg.norm(feat) * np.linalg.norm(db_feat))

            # Set a similarity threshold (you may need to experiment with this value)
            if similarity > 0.3:
                print(f"Found person {identity} with similarity {similarity}")
                # Draw bounding box around the detected face (replace this with your drawing code)
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
                cv2.putText(img, f"{identity}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return img

    # Display or save the image with bounding boxes
    # cv2.imwrite("Result_image.jpg", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# # Example usage
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='PyTorch ArcFace Inference')
#     parser.add_argument('--network', type=str, default='r50', help='backbone network')
#     parser.add_argument('--weight', type=str, default='')
#     parser.add_argument('--img', type=str, default=None)
#     parser.add_argument('--database', type=str, default='path/to/database/folder')
#     args = parser.parse_args()
#     inference(args.weight, args.network, args.img, args.database)
