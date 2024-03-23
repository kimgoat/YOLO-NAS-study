
import cv2
import torch.cuda
from super_gradients.training import models
from super_gradients.common.object_names import Models


#Note that currently YOLOX and PRYOLOE are spported

model = models.get(Models.YOLOX_N, pretrained_weights='coco')

model = model.to("cuda" if torch.cuda.is_available()else 'cpu')

models.convert_to_onnx(model = model, input_shape = (3,640, 640), out_path = "yolo_nas_s.onnx")

# model.predict_webcam()
