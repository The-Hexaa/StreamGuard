# from ultralytics import YOLO
# from PIL import Image
# import supervision as sv
# import cv2
# import os
# # Load the YOLOv8 model
# model = YOLO('best.pt')  # You can choose the specific variant (n, s, m, l, x) based on your need


# #identifies one and multiple faces
# def identify_face_one_multiple():

#     directory_path = 'content/tagged'
#     for filename in os.listdir(directory_path):
#         file_path = os.path.join(directory_path, filename)
#         tag_image = cv2.imread(file_path)

#         results = model(tag_image)
#         detections = sv.Detections.from_ultralytics(results[0])

#         box_annotator = sv.BoundingBoxAnnotator(thickness=2)
#         label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=1)

#         tag_image = box_annotator.annotate(scene=tag_image, detections=detections)
#         tag_image = label_annotator.annotate(scene=tag_image, detections=detections)

#         # Assuming 'frame' is a NumPy array or similar
#         face_img_resized = cv2.resize(tag_image, (500, 500))

#         cv2.imwrite(f"content/output/{filename}.jpg", face_img_resized)


from ultralytics import YOLO
from PIL import Image
import supervision as sv
from keras_facenet import FaceNet
import cv2
import numpy as np
from matplotlib import pyplot as plt


