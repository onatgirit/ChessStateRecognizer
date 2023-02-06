import torch
from PIL import Image
import torchvision.transforms.functional as F
from ChessboardConfiguration import ChessboardConfiguration as cfg
import cv2
from nms import nms
from iou import intersection_over_union
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
import numpy as np
import torchvision

model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/yolov5/content/yolov5/runs/train/exp/weights/best.pt')

x_path = "Warped.png"
model.eval()
output = model(x_path)
output.save()  # Output of the object detection model is saved under runs/detect/exp
o = output.pred[0].type(torch.FloatTensor).tolist()
classes_list = output.names
classes_map = dict(zip(list(range(6)), classes_list))
preds = nms(o, 0.9, 0.3)
preds = np.array(preds)
coordinates = torch.tensor(preds[:, :4], dtype=torch.int)
confidence_scores = preds[:, 4]
class_preds = preds[:, 5]
class_preds = [classes_map[class_preds[i]] for i in range(len(class_preds))]

img = read_image(x_path)
img = draw_bounding_boxes(img, coordinates, width=5, labels=class_preds, fill=False)
img = torchvision.transforms.ToPILImage()(img)
img.save("output.png")

output_csv = ""
for coordinate, class_pred in zip(coordinates, class_preds):
    x_avg = (coordinate[0] + coordinate[2]) / 2
    y_avg = (coordinate[1] + coordinate[3]) / 2
    piece_coordinate = (x_avg // 80) + 1, (y_avg // 80) + 1
    output_csv += f"{int(piece_coordinate[0])},{int(piece_coordinate[1])},{class_pred}\n"
output_file = open("Result.csv", "w")
output_file.write(output_csv)
output_file.close()
