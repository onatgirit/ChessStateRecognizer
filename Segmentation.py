from DeepLabV3 import DeepLabV3
from ContourExtractor import ContourExtractor
import cv2

model = DeepLabV3()

model.load_save()

img_path = "Test.png"
output = model(img_path)
img = cv2.imread(img_path)
orig_img = img.copy()
img = cv2.resize(img, (240, 135))
approx = ContourExtractor.get_approx_quad(output)
output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
cv2.drawContours(output, [approx], 0, (0, 255, 0), 2)
warped = ContourExtractor.get_warped_board(orig_img, approx)
cv2.imwrite("Warped.png", warped)
