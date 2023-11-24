import os
import os.path as osp
import cv2
import numpy as np
import onnxruntime
from scrfd import SCRFD
from arcface_onnx import ArcFaceONNX
from numpy.linalg import norm

onnxruntime.set_default_logger_severity(3)

assets_dir = osp.expanduser('~/.insightface/models/buffalo_l')

detector = SCRFD(os.path.join(assets_dir, 'det_10g.onnx'))
detector.prepare(0)
model_path = os.path.join(assets_dir, '/home/airi/Makhmud/insightface/recognition/arcface_torch/work_dirs/ms1mv3_r50/onnx.onnx')
rec = ArcFaceONNX(model_path)
rec.prepare(0)
 
def func(image):
    image1 = cv2.imread(image)
    bboxes1, kpss1 = detector.autodetect(image1, max_num=1)
    if bboxes1.shape[0]==0:
        return -1.0, "Face not found in Image-1"
    kps1 = kpss1[0]
    feat1 = rec.get(image1, kps1)

    return feat1

def compute_sim(feat1, feat2):
    
    feat1 = feat1.ravel()
    feat2 = feat2.ravel()
    sim = np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))
    return sim

image = '/home/airi/Makhmud/insightface/recognition/arcface_torch/data/Jasur/Jasur-10.jpg'
feat1 = np.load('/home/airi/Makhmud/insightface/web-demos/src_recognition/ishxona/Abduraxim.npy')

feat2 = func(image)
result = {}

for feat in os.listdir('ishxona/'):
    path = 'ishxona' + '/' + feat
    feature = np.load(path)
    sim = compute_sim(feature, feat2)
    result[feat] = sim

print(result)