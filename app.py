from fastapi.staticfiles import StaticFiles
from fastapi import UploadFile, File
import os
import string
import random
from fastapi import FastAPI, Response
import cv2
from sqlalchemy import create_engine, Column, Integer, String, select
from sqlalchemy.orm import sessionmaker, declarative_base
from passlib.context import CryptContext
from itsdangerous import URLSafeTimedSerializer, BadSignature
from starlette.middleware.sessions import SessionMiddleware
from fastapi import FastAPI, Request, Form, Depends, status, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import JSONResponse, FileResponse
import sys
import requests
import numpy as np
from file_monitor import on_modified
# Database setup
DATABASE_URL = "sqlite:///./test.db"
Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Security setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = "your-secret-key"
serializer = URLSafeTimedSerializer(SECRET_KEY)

# Models
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)

Base.metadata.create_all(bind=engine)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

app = FastAPI()

app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

# Mount the static directory to serve static files like videos
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


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

test_url = "http://localhost:7978/get_frame_to_working_return_Frame/"
@app.get("/video_feed")
async def video_feed(request: Request, db: SessionLocal = Depends(get_db)):
    def generate_frames():

        url = 'http://192.168.2.67:4747/video'
        cap = cv2.VideoCapture(url)
        
        if not cap.isOpened():
            print("Error: Could not open video stream.")
            return

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

            out.write(frame)
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # frame = get_frame_to_working_return_Frame(frame)
            try:
                response = requests.post(test_url, files={"file": ("frame.jpg", frame, "image/jpeg")})
                print("response", response)
            except requests.exceptions.RequestException as e:
                print(f"An error occurred: {e}")
            
            nparr = np.frombuffer(response.content, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)


            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()


            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        cap.release()
        out.release()

    user = get_user_by_email(db, request.session.get("user"))

    if not user:
        return RedirectResponse(url="/login", status_code=303)
    return StreamingResponse(generate_frames(), media_type='multipart/x-mixed-replace; boundary=frame')



@app.get("/record", response_class=HTMLResponse)
async def read_root(request: Request, db: SessionLocal = Depends(get_db)):
    
    user = get_user_by_email(db, request.session.get("user"))
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse("record.html", {"request": request})


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_user(db, email: str):
    user = db.get(email)
    return user

def authenticate_user(fake_db, email: str, password: str):
    user = get_user(fake_db, email)
    if not user:
        return False
    if not verify_password(password, user['hashed_password']):
        return False
    return user


@app.get("/", response_class=HTMLResponse)
async def get_html(request: Request, db: SessionLocal = Depends(get_db)): 
    user = get_user_by_email(db, request.session.get("user"))
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/login", response_class=HTMLResponse)
async def login_get(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/login", response_class=HTMLResponse)
async def login(email: str = Form(...), password: str = Form(...), db: SessionLocal = Depends(get_db), request: Request = None):
    user = get_user_by_email(db, email)
    if not user or not verify_password(password, user.hashed_password):
                return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid credentials"})
    
    request.session["user"] = user.email
    return RedirectResponse(url="/", status_code=303)

@app.get("/logout")
async def logout(request: Request):
    request.session.pop("user", None)
    return RedirectResponse(url="/login", status_code=303)


# Utility functions
def get_user_by_email(db, email: str):
    return db.execute(select(User).filter(User.email == email)).scalar_one_or_none()

def get_user_by_username(db, username: str):
    return db.execute(select(User).filter(User.username == username)).scalar_one_or_none()

@app.get("/register", response_class=HTMLResponse)
async def login_get(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})


@app.post("/register", response_class=HTMLResponse)
async def register(request: Request, username: str = Form(...), email: str = Form(...), password: str = Form(...), db: SessionLocal = Depends(get_db)):
    existing_email = get_user_by_email(db, email)
    user = get_user_by_username(db, username)
    if user or existing_email:
        return JSONResponse(content={"error": "Username or email already registered"}, status_code=400)
    
    hashed_password = get_password_hash(password)
    new_user = User(username=username, email=email, hashed_password=hashed_password)
    print(username, email, hashed_password)
    db.add(new_user)
    db.commit()
    return JSONResponse(content={"message": "Registration successful"}, status_code=200)

    




