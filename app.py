from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi import Request
from fastapi import FastAPI
import os
import string
import random
from model import thief_getYoloFace_embed_and_save_it
from fastapi import FastAPI, Response
import cv2
from ultralytics import YOLO
import supervision as sv
from keras_facenet import FaceNet
import numpy as np
from file_monitor import on_modified, send_mail_to_admin
import asyncio

app = FastAPI()
model = YOLO('best.pt') 
embedder = FaceNet()


# Mount the static directory to serve static files like videos
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


def generate_random_string(length=5):
    characters = string.ascii_letters + string.digits  # Includes uppercase, lowercase letters, and digits
    return ''.join(random.choice(characters) for _ in range(length))
@app.get("/video")
async def serve_video(filename: str):
    """
    Serve a video file.

    To call this endpoint using curl:
    curl -X 'GET' \
      'http://127.0.0.1:8000/video?filename=output.mp4' \
      -H 'accept: application/json'
    """
    file_path = os.path.join('content/record', filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        return {"error": "File not found"}

@app.get("/", response_class=HTMLResponse)
async def get_html(request: Request):
    # html_content = """
    # <!DOCTYPE html>
    # <html>
    # <head>
    #     <title>Live Stream</title>
    # </head>
    # <body>
    #     <h1>Live Video Stream</h1>
    #     <img id="videoFeed" src="/video_feed" width="800" height="600" />
    #     <button id="screenshotBtn">Tag a person</button>

    #     <script>
    #     document.getElementById("screenshotBtn").addEventListener("click", function() {
    #         fetch('/capture_frame', { method: 'POST' })
    #             .then(response => {
    #                 if (response.ok) {
    #                     return response.blob();
    #                 } else {
    #                     throw new Error('Failed to capture frame');
    #                 }
    #             })
    #             .then(blob => {
    #                 const link = document.createElement('a');
    #                 link.href = URL.createObjectURL(blob);
    #                 link.download = 'screenshot.jpg';
    #                 link.click();
    #             })
    #             .catch(error => {
    #                 alert('Error: ' + error.message);
    #             });
    #     });
    #     </script>
    # </body>
    # </html>
    # """
    # return HTMLResponse(content=html_content)
    return templates.TemplateResponse("index.html", {"request": request})

def generate_random_string(length=5):
    characters = string.ascii_letters + string.digits  # Includes uppercase, lowercase letters, and digits
    return ''.join(random.choice(characters) for _ in range(length))


@app.post("/upload")
async def upload_image(image: UploadFile = File(...)):
    # try:
        # Read the image file
        contents = await image.read()
        
        file_path = 'content/tagged/' + generate_random_string() + '.png'
        with open(file_path, "wb") as f:
            f.write(contents)

        # Optionally, you can process the image here

        # thief_getYoloFace_embed_and_save_it()

    #     return JSONResponse(content={"message": "Image uploaded successfully!"}, status_code=200)
    # except Exception as e:
    #     return JSONResponse(content={"message": f"Failed to upload image: {str(e)}"}, status_code=500)


@app.get("/video_feed")
async def video_feed():
    def generate_frames():
        url = 'http://192.168.2.67:4747/video'
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            print("Error: Could not open video stream.")
            return JSONResponse(
            status_code=500,
            content={"detail": "Error: Could not open video stream."}
        )

        fourcc = cv2.VideoWriter_fourcc(*'vp80')
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        output_file_path = 'content/record/output.webm'
        print(f"Frame width: {frame_width}, Frame height: {frame_height}")
        out = cv2.VideoWriter(output_file_path, fourcc, 30.0, (frame_width, frame_height))

        while True:
            success, frame = cap.read()

            if not out.isOpened():
                print(f"Error: Could not open video writer with path {output_file_path}.")
                return


            if not success:
                print("Error: Could not read frame.")
                on_modified()
                cap.release()  # Release the capture when done
                out.release()
                break

            if frame is not None:
                out.write(frame)
                print("Frame written to video.")
            
            results = model(frame)

            similarity_results = getSimilarity(frame)

            detections = sv.Detections.from_ultralytics(results[0])

            for i in range(len(detections)):
                if i < len(similarity_results):
                    similarity, filename = similarity_results[i]
                    detections.data['class_name'][i]  = f"{similarity:.2f} - {filename}"

            box_annotator = sv.BoundingBoxAnnotator(thickness=2)
            label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=1)

            frame = box_annotator.annotate(scene=frame, detections=detections)
            frame = label_annotator.annotate(scene=frame, detections=detections)


            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        cap.release()  # Release the capture when done
        out.release()

    return StreamingResponse(generate_frames(), media_type='multipart/x-mixed-replace; boundary=frame')



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
    similarity_results = []
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

        min_similarity = 5.0
        for index, face_from_video_frame in enumerate(face1_embeddings):
            similarity = np.linalg.norm(face_from_video_frame - face2_embeddings)
            
            if(similarity < min_similarity):
                min_similarity = similarity

            similarity_results.append((similarity, filename))
            # print(f"''{filename}'', Similarity score with {index+1} face from video frame: {similarity}")
    
    if min_similarity != 5.0:
        asyncio.run(send_mail_to_admin(filename, min_similarity))
    return similarity_results



@app.get("/record", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("record.html", {"request": request})
