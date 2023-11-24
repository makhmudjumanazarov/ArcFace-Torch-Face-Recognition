import os
import os.path as osp
import cv2
import numpy as np
import onnxruntime
from scrfd import SCRFD
from arcface_onnx import ArcFaceONNX
from time import time 

onnxruntime.set_default_logger_severity(3)

assets_dir = osp.expanduser('~/.insightface/models/buffalo_l')

detector = SCRFD(os.path.join(assets_dir, 'det_10g.onnx'))
detector.prepare(0)

model_path = os.path.join(assets_dir, '/home/airi/Makhmud/insightface/recognition/arcface_torch/work_dirs/ms1mv3_r50/onnx.onnx')
rec = ArcFaceONNX(model_path)
rec.prepare(0)

img1 = '/home/airi/Makhmud/insightface/web-demos/src_recognition/face_test_images/brad.jpg'
image1 = cv2.imread(img1)
bboxes1, kpss1 = detector.autodetect(image1, max_num=10)

prev_time = time()
feat1 = rec.get(image1, kpss1)
print(len(feat1))
current_time = time()
print(current_time - prev_time)

# # Assuming feat1 is a NumPy array
# np.save('ishxona/leo_young.npy', feat1)    
