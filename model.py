from ultralytics import YOLO
from PIL import Image
import supervision as sv
from keras_facenet import FaceNet
import cv2
import numpy as np
import os
# Load the YOLOv8 model
model = YOLO('best.pt')  # You can choose the specific variant (n, s, m, l, x) based on your need

#video frame could be multiple faces
def getYoloFace_multiple_face(img):
    results = model(img)

    detections = sv.Detections.from_ultralytics(results[0])

    all_face_xyxy = detections.xyxy

    crop_faces = []
    for index, one_face_xyxy in enumerate(all_face_xyxy):
        x1, y1, x2, y2 = map(int, one_face_xyxy)

        face = img[y1:y2, x1:x2]
        face_img_resized = cv2.resize(face, (500, 500))
        crop_faces.append(face_img_resized)

    return crop_faces

def encodeFace(faces):
    
    print("Encoding face with FaceNet...")
    print("facenet model load before")
    embedder = FaceNet()
    print("facenet model load") 
    
    embeddings = []
    for face in faces:
        # FaceNet expects images of size (160, 160)
        face_img_resized = cv2.resize(face, (160, 160))
        
        # Expand dimensions since FaceNet expects a batch of images
        face_img_resized = np.expand_dims(face_img_resized, axis=0)
        
        # Encode the face using FaceNet
        embeddings.append(embedder.embeddings(face_img_resized)[0])
        
    print("Face encoded.")
    return embeddings

def getSimilarity(video_frame):

    directory_path = 'content/tagged'
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        tag_image = cv2.imread(file_path)  #have to be single pic

        crop_faces_from_frame_arr = getYoloFace_multiple_face(video_frame)  # arr could be multiple
        tag_face2_img = getYoloFace_multiple_face(tag_image)  # arr but one face only

        if(len(tag_face2_img) == 0):
            print("Cant detected face in face2_img")
            return
    
        elif(len(crop_faces_from_frame_arr) == 0):
            print("Cant detected face in faccrop_faces_from_frame_arre1_img")
            return

        face1_embeddings = encodeFace(crop_faces_from_frame_arr) # arr could be multiple
        face2_embeddings = encodeFace(tag_face2_img) # arr but one face only

        for index, face_from_video_frame in enumerate(face1_embeddings):
            similarity = np.linalg.norm(face_from_video_frame - face2_embeddings)
            print(f"''{filename}'', Similarity score with {index+1} face from video frame: {similarity}")
    
    return similarity


# this function loop thru all frame 
def loop_thru_all_frame():
    generator = sv.get_video_frames_generator('content/raw/umer-umair.mp4')
    iterator = iter(generator)

    box_annotator = sv.BoundingBoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=1)
    
    index = 0
    
    while True:
        try:
            frame = next(iterator)
        except StopIteration:
            break  # Exit the loop if there are no more frames
        
        getSimilarity(frame)  




def thief_getYoloFace_embed_and_save_it():
    loop_thru_all_frame()

