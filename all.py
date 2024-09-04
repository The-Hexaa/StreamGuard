# from fastapi.staticfiles import StaticFiles
# from fastapi.responses import HTMLResponse, StreamingResponse
# from fastapi.templating import Jinja2Templates
# from fastapi import FastAPI, UploadFile, File
# from fastapi.responses import JSONResponse
# from fastapi import Request
# from fastapi import FastAPI
# import os
# import string
# import random
# from model import thief_getYoloFace_embed_and_save_it
# from fastapi import FastAPI, Response
# import cv2
# from ultralytics import YOLO
# import supervision as sv


# app = FastAPI()

# # Mount the static directory to serve static files like videos
# app.mount("/static", StaticFiles(directory="static"), name="static")

# templates = Jinja2Templates(directory="templates")


# def generate_random_string(length=5):
#     characters = string.ascii_letters + string.digits  # Includes uppercase, lowercase letters, and digits
#     return ''.join(random.choice(characters) for _ in range(length))


# # @app.get("/", response_class=HTMLResponse)
# # async def read_root(request: Request):
# #     identify_face_one_multiple()
# #     return templates.TemplateResponse("index.html", {"request": request})

# # @app.post("/upload")
# # async def upload_image(image: UploadFile = File(...)):
# #     # try:
# #         # Read the image file
# #         contents = await image.read()
        
# #         file_path = 'content/tagged/' + generate_random_string() + '.png'
# #         with open(file_path, "wb") as f:
# #             f.write(contents)

# #         # Optionally, you can process the image here

# #         # thief_getYoloFace_embed_and_save_it()

# #     #     return JSONResponse(content={"message": "Image uploaded successfully!"}, status_code=200)
# #     # except Exception as e:
# #     #     return JSONResponse(content={"message": f"Failed to upload image: {str(e)}"}, status_code=500)


# from fastapi import FastAPI, Request
# from fastapi.responses import HTMLResponse, Response, StreamingResponse
# import cv2
# import io

# app = FastAPI()


# @app.get("/", response_class=HTMLResponse)
# async def get_html(request: Request):
#     html_content = """
#     <!DOCTYPE html>
#     <html>
#     <head>
#         <title>Live Stream</title>
#     </head>
#     <body>
#         <h1>Live Video Stream</h1>
#         <img id="videoFeed" src="/video_feed" width="800" height="600" />
#         <button id="screenshotBtn">Tag a person</button>

#         <script>
#         document.getElementById("screenshotBtn").addEventListener("click", function() {
#             fetch('/capture_frame', { method: 'POST' })
#                 .then(response => {
#                     if (response.ok) {
#                         return response.blob();
#                     } else {
#                         throw new Error('Failed to capture frame');
#                     }
#                 })
#                 .then(blob => {
#                     const link = document.createElement('a');
#                     link.href = URL.createObjectURL(blob);
#                     link.download = 'screenshot.jpg';
#                     link.click();
#                 })
#                 .catch(error => {
#                     alert('Error: ' + error.message);
#                 });
#         });
#         </script>
#     </body>
#     </html>
#     """
#     return HTMLResponse(content=html_content)

# @app.post("/model_working")
# def model_working():
#    identify_face_one_multiple()

# @app.get("/video_feed")
# async def video_feed():
#     def generate_frames():
#         # model = YOLO('best.pt') 
#         url = 'http://192.168.2.67:4747/video'
#         cap = cv2.VideoCapture(url)
#         if not cap.isOpened():
#             print("Error: Could not open video stream.")
#             return
        
#         while True:
#             success, frame = cap.read()
#             if not success:
#                 print("Error: Could not read frame.")
#                 break
            
#             results = model(frame)

#             detections = sv.Detections.from_ultralytics(results[0])

#             box_annotator = sv.BoundingBoxAnnotator(thickness=2)
#             label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=1)

#             frame = box_annotator.annotate(scene=frame, detections=detections)
#             frame = label_annotator.annotate(scene=frame, detections=detections)


#             _, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
#         cap.release()  # Release the capture when done

#     return StreamingResponse(generate_frames(), media_type='multipart/x-mixed-replace; boundary=frame')



# # @app.get("/video_feed")
# # async def video_feed():
# #     def generate_frames():

# #         model = YOLO('best.pt') 

# #         url = 'http://192.168.2.67:4747/video'
# #         cap = cv2.VideoCapture(url)

# #         if not cap.isOpened():
# #             print("Error: Could not open camera.")
# #         else:
# #             while True:
# #                 ret, frame = cap.read()

# #             if not ret:
# #                 print("Error: Could not read frame.")
# #                 return

# #                 # Display the resulting frame
# #                 results = model(frame)

# #                 detections = sv.Detections.from_ultralytics(results[0])

# #                 box_annotator = sv.BoundingBoxAnnotator(thickness=2)
# #                 label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=1)

# #                 frame = box_annotator.annotate(scene=frame, detections=detections)
# #                 frame = label_annotator.annotate(scene=frame, detections=detections)

# #                 _, buffer = cv2.imencode('.jpg', frame)
# #                 frame = buffer.tobytes()
# #                 yield (b'--frame\r\n'
# #                     b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
# #             cap.release()  # Release the capture when done

# #         return StreamingResponse(generate_frames(), media_type='multipart/x-mixed-replace; boundary=frame')


