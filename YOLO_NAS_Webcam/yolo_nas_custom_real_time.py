import cv2
import torch.cuda
from super_gradients.training import models
from super_gradients.common.object_names import Models


# 1. apply custom modeling to webcam
# model = models.get('yolo_nas_s', num_classes=7, checkpoint_path='weights/checkpoint-best.pth')
# model = model.to("cuda" if torch.cuda.is_available() else 'cpu')
#
# model.predict_webcam()


# 2. apply custom modeling to image
img = cv2.imread("images/img01.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

model = models.get('yolo_nas_s', num_classes=7, checkpoint_path='weights/checkpoint-best.pth')
model = model.to("cuda" if torch.cuda.is_available() else 'cpu')

outputs = model.predict(img)
outputs.show()