from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from jose import jwt, JWTError
from passlib.context import CryptContext
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io
import os
from sqlalchemy import Text, Float, DateTime
from datetime import datetime
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from pydantic import BaseModel
from fastapi import Security
from datetime import datetime, timedelta
from fastapi import UploadFile, File 
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import io
from PIL import Image
# from db_models import SessionLocal, User, Detection
from fastapi import Depends
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session
from db_models import User # or wherever your User model is defined
from db_models import SessionLocal # your DB session creator
from db_models import User, Detection, SessionLocal
import json
import shutil
from typing import List
from db_models import Upload
from database1 import users_collection, detections_collection, uploads_collection


# Updated
object_info = {
"10337336": {
"description": "Fine-threaded screw for precision assemblies.",
"use": "Used in electronics or fine mechanics.",
"price": "$0.10",
"location": "Shelf A1"
},
"10344614": {
"description": "Quarter-inch fine thread screw.",
"use": "Used in light metal construction.",
"price": "$0.12",
"location": "Shelf A2"
},
"10345315": {
"description": "Heavy-duty metric screw.",
"use": "Structural or industrial uses.",
"price": "$0.50",
"location": "Shelf B1"
},
"10345517": {
"description": "Metric 4mm screw with 8mm length.",
"use": "Used in panels and brackets.",
"price": "$0.08",
"location": "Shelf C3"
},
"0348974": {
"description": "5mm screw, commonly used in mounts.",
"use": "Used in robotics and enclosures.",
"price": "$0.10",
"location": "Shelf C4"
},
"10439665": {
"description": "5mm screw, commonly used in mounts.",
"use": "Used in robotics and enclosures.",
"price": "$0.10",
"location": "Shelf C4"
},
"10452055": {
"description": "5mm screw, commonly used in mounts.",
"use": "Used in robotics and enclosures.",
"price": "$0.10",
"location": "Shelf C4"
},
"0454022": {
"description": "5mm screw, commonly used in mounts.",
"use": "Used in robotics and enclosures.",
"price": "$0.10",
"location": "Shelf C4"
},
"10454776": {
"description": "5mm screw, commonly used in mounts.",
"use": "Used in robotics and enclosures.",
"price": "$0.10",
"location": "Shelf C4"
},
"10454837": {
"description": "5mm screw, commonly used in mounts.",
"use": "Used in robotics and enclosures.",
"price": "$0.10",
"location": "Shelf C4"
},
"10558519": {
"description": "5mm screw, commonly used in mounts.",
"use": "Used in robotics and enclosures.",
"price": "$0.10",
"location": "Shelf C4"
},
"0565488": {
"description": "5mm screw, commonly used in mounts.",
"use": "Used in robotics and enclosures.",
"price": "$0.10",
"location": "Shelf C4"
},
"video2": {
"description": "5mm screw, commonly used in mounts.",
"use": "Used in robotics and enclosures.",
"price": "$0.10",
"location": "Shelf C4"
},
"video_frame": {
"description": "5mm screw, commonly used in mounts.",
"use": "Used in robotics and enclosures.",
"price": "$0.10",
"location": "Shelf C4"
},
}

# close update


# JWT Configuration
SECRET_KEY = "your_super_secret_key"  # You can change this to a random string
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60  # Token will expire after 60 minutes

# update
# close update

def create_access_token(data: dict, expires_delta: timedelta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)):
 to_encode = data.copy()
 expire = datetime.utcnow() + expires_delta
 to_encode.update({"exp": expire})
 encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
 return encoded_jwt 

class UserCreate(BaseModel):
 username: str
 password: str

# --- JWT CONFIG ---
SECRET_KEY = "mysecretkey"
ALGORITHM = "HS256"

# --- Auth + DB Setup ---
Base = declarative_base()
DATABASE_URL = "sqlite:///./new_database.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)

# class Detection(Base):
#  tablename = "detections"
# id = Column(Integer, primary_key=True, index=True)
# class_name = Column(String)
# confidence = Column(Float)
# bbox = Column(Text)
# filename = Column(String)
# timestamp = Column(DateTime, default=datetime.utcnow)

# class Detection(Base):
#     __tablename__ = "detections"

#     id = Column(Integer, primary_key=True, index=True)
#     class_name = Column(String, nullable=False)
#     confidence = Column(Float, nullable=False)
#     bbox = Column(Text, nullable=False)
#     filename = Column(String, nullable=False)
#     timestamp = Column(DateTime, default=datetime.utcnow)
from db_models import Detection


Base.metadata.create_all(bind=engine)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# updated
def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
 credentials_exception = HTTPException(
status_code=status.HTTP_401_UNAUTHORIZED,
detail="Could not validate credentials",
headers={"WWW-Authenticate": "Bearer"},
)
 try:
  payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
  username: str = payload.get("sub")
  if username is None:
   raise credentials_exception
 except JWTError:
  raise credentials_exception
 user = db.query(User).filter(User.username == username).first()
 if user is None:
    raise credentials_exception
 return user
# close update

def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain, hashed):
    return pwd_context.verify(plain, hashed)

def create_access_token(data: dict):
    return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)

# --- FastAPI Init ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[""],
    allow_credentials=True,
    allow_methods=[""],
    allow_headers=["*"],
)



app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Load YOLO Model ---
# model = YOLO("runs/detect/train/weights/best.pt")
model = YOLO("best (5).pt")
print("Loaded YOLOv8 model classes:", model.names)

# --- Auth Schemas ---
class UserIn(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

# --- Routes ---
@app.get("/", response_class=HTMLResponse)
async def serve_home():
    return FileResponse(os.path.join("static", "index.html"))

@app.get("/")
def read_root():
    return {"message": "FastAPI + MongoDB Atlas Connected!"}

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# @app.post("/register")
# def register(user: UserIn, db: Session = Depends(get_db)):
#     if db.query(User).filter(User.username == user.username).first():
#         raise HTTPException(status_code=400, detail="Username already exists")
#     db_user = User(username=user.username, hashed_password=get_password_hash(user.password))
#     db.add(db_user)
#     db.commit()
#     return {"message": "User registered successfully"}

@app.post("/register")
def register_user(user: UserCreate, db: Session = Depends(get_db)):
 existing = db.query(User).filter(User.username == user.username).first()
 if existing:
  raise HTTPException(status_code=400, detail="User already exists")
 hashed = pwd_context.hash(user.password)
 db_user = User(username=user.username, hashed_password=hashed)
 db.add(db_user)
 db.commit()
 db.refresh(db_user)
 return {"message": "User registered successfully"}


# @app.post("/token", response_model=Token)
# def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
#     user = db.query(User).filter(User.username == form_data.username).first()
#     if not user or not verify_password(form_data.password, user.hashed_password):
#         raise HTTPException(status_code=401, detail="Invalid credentials")
#     token = create_access_token({"sub": user.username})
#     return {"access_token": token, "token_type": "bearer"}

# @app.post("/token")
# def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
#  user = db.query(User).filter(User.username == form_data.username).first()
#  if not user or not pwd_context.verify(form_data.password, user.hashed_password):
#   raise HTTPException(status_code=401, detail="Invalid credentials")
#  return {"access_token": user.username, "token_type": "bearer"}

# @app.post("/token")
# def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
#  user = db.query(User).filter(User.username == form_data.username).first()
#  if not user or not pwd_context.verify(form_data.password, user.hashed_password):
#   raise HTTPException(status_code=401, detail="Invalid credentials") 

#   token_data = {"sub": user.username}
#   token = create_access_token(token_data)

#  return {"access_token": token, "token_type": "bearer"}
@app.post("/token")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not pwd_context.verify(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token_data = {"sub": user.username}
    token = create_access_token(token_data)  # ✅ Now token is defined

    return {"access_token": token, "token_type": "bearer"}

@app.get("/protected")
def protected(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        return {"message": f"Hello, {username}. You are authorized."}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     contents = await file.read()
#     image = Image.open(io.BytesIO(contents)).convert("RGB")
#     results = model(image)
#     predictions = []
#     for r in results:
#         for box in r.boxes:
#             cls_id = int(box.cls[0])
#             score = float(box.conf[0])
#             bbox = box.xyxy[0].tolist()
#             predictions.append({
#                 "class": model.names[cls_id],
#                 "score": score,
#                 "bbox": bbox
#             })
#     if not predictions:
#         return JSONResponse({"message": "No objects detected."})
#     return {"results": predictions}


# @app.post("/predict")
# async def predict(file: UploadFile = File(...),
# db: Session = Depends(get_db),
# token: str = Depends(oauth2_scheme)
# ):
#   contents = await file.read()
#   file: UploadFile = File(...),
#   db: Session = Depends(get_db),
#   token: str = Depends(oauth2_scheme)
#   contents = await file.read()
#   image = Image.open(io.BytesIO(contents)).convert("RGB")
#   filename = file.filename
#   results = model(image)
#   predictions = []

#   for r in results:
#     for box in r.boxes:
#         cls_id = int(box.cls[0])
#         score = float(box.conf[0])
#         bbox = box.xyxy[0].tolist()

#         predictions.append({
#             "class": model.names[cls_id],
#             "score": score,
#             "bbox": bbox
#         })

#         # Store each detection in the database
#         db_detection = Detection(
#             class_name=model.names[cls_id],
#             confidence=score,
#             bbox=str(bbox),
#             filename=filename
#         )
#         db.add(db_detection)

#     db.commit()

#     if not predictions:
#      return JSONResponse({"message": "No objects detected."})
#   return {"results": predictions}

# @app.post("/predict")
# async def predict(
# file: UploadFile = File(...),
# # db: Session = Depends(get_db),
# # token: str = Depends(oauth2_scheme)
# ):
#  contents = await file.read()
#  image = Image.open(io.BytesIO(contents)).convert("RGB")
#  filename = file.filename
#  results = model(image)
#  predictions = []
#  for r in results:
#   boxes = r.boxes
#   for box in boxes:
#    cls_id = int(box.cls[0])
#    confidence = float(box.conf[0])
#    x1, y1, x2, y2 = box.xyxy[0].tolist()
#    predictions.append({
#    "class": model.names[cls_id],
#    "score": confidence,
#    "bbox": [x1, y1, x2, y2]
#   })

#   return {"results": predictions}


# /updated

# @app.post("/predict")
# async def predict(
#    file: UploadFile = File(...)):
#  contents = await file.read()
#  image = Image.open(io.BytesIO(contents)).convert("RGB")
#  filename = file.filename
#  results = model(image)
#  predictions = []
#  for r in results:
#     boxes = r.boxes
#     for box in boxes:
#         cls_id = int(box.cls[0])
#         confidence = float(box.conf[0])
#         x1, y1, x2, y2 = box.xyxy[0].tolist()
#         class_name = model.names[cls_id]

# #         # Get object metadata if available
#         meta = object_info.get(class_name, {})
    
#         predictions.append({
#             "class": class_name,
#             "score": confidence,
#             "bbox": [x1, y1, x2, y2],
#             "description": meta.get("description", "N/A"),
#             "use": meta.get("use", "N/A"),
#             "price": meta.get("price", "N/A"),
#             "location": meta.get("location", "N/A")
#         })

#  return {"results": predictions}

# //update

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    filename = file.filename
    results = model(image)

    predictions = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_name = model.names[cls_id]

            detection = Detection(
                username=current_user.username,
                class_name=class_name,
                confidence=confidence,
                bbox=json.dumps([x1, y1, x2, y2]),
                filename=filename,
                timestamp = datetime.now(),  # Uses local time
                user_id=current_user.id
            )
            db.add(detection)
            
            results = model(image)

            predictions.append({
                "class": class_name,
                "score": confidence,
                "bbox": [x1, y1, x2, y2],
                "description": object_info.get(class_name, {}).get("description", "N/A"),
                "use": object_info.get(class_name, {}).get("use", "N/A"),
                "price": object_info.get(class_name, {}).get("price", "N/A"),
                "location": object_info.get(class_name, {}).get("location", "N/A")
            })



    db.commit()
    return {"results": predictions}




        # close 

        # updated
# @app.post("/predict")
# async def predict(
#   file: UploadFile = File(...),
#   db: Session = Depends(get_db),
#   current_user: User = Depends(get_current_user)
# ):
#  try:
#   contents = await file.read()
#   image = Image.open(io.BytesIO(contents)).convert("RGB")
#   filename = file.filename
#   results = model(image)

#   predictions = []
#   for r in results:
#         boxes = r.boxes
#         for box in boxes:
#             cls_id = int(box.cls[0])
#             confidence = float(box.conf[0])
#             x1, y1, x2, y2 = box.xyxy[0].tolist()

#             class_name = model.names[cls_id]
#             bbox_str = f"{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f}"

#             # Save to database
#             db_detection = Detection(
#                 class_name=class_name,
#                 confidence=confidence,
#                 bbox=bbox_str,
#                 filename=filename,
#                 timestamp=datetime.utcnow(),
#                 user_id=current_user.id  # if your Detection model supports it
#             )
#             db.add(db_detection)

#             predictions.append({
#                 "class": class_name,
#                 "score": confidence,
#                 "bbox": [x1, y1, x2, y2]
#             })

#         db.commit()
#         return {"results": predictions}
  
#  except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)  # get user from JWT
):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    results = model(image)  # assuming model is already loaded globally
    detection_data = results[0].boxes.data.tolist()  # adjust if your model returns differently

    predictions = []

    # for box in detection_data:
    #     x1, y1, x2, y2, conf, cls = box[:6]
    #     class_name = model.names[int(cls)]

    #     # Save to database
    #     detection_entry = Detection(
    #         user_id=current_user.id,
    #         class_name=class_name,
    #         confidence=float(conf),
    #         bbox=json.dumps([x1, y1, x2, y2])
    #     )
    #     db.add(detection_entry)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_name = model.names[cls_id]

            detection = Detection(
                username=current_user.username,
                class_name=class_name,
                confidence=conf,
                bbox=json.dumps([x1, y1, x2, y2]),
                filename=filename,
                timestamp = datetime.now(),  # Uses local time
                user_id=current_user.id
                # username=current_user.username  # ✅ ensure user islinked
            )
            db.add(detection)

        # Prepare for frontend display
        predictions.append({
            "class": class_name,
            "score": float(conf),
            "bbox": [x1, y1, x2, y2],
            "description": "N/A",
            "use": "N/A",
            "price": "N/A",
            "location": "N/A"
        })
    
    db.commit()

    return {"results": predictions}


# @app.get("/history")
# def get_history(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
#     detections = db.query(Detection).filter(Detection.user_id == current_user.id).all()
#     return [{"class": d.class_name, "score": d.confidence, "bbox": json.loads(d.bbox), "time": d.timestamp} for d in detections]

# /add-image Endpoint

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
RESULTS_DIR = "static/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

@app.post("/add-images")
async def add_images(
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    try:
        uploaded_files = []
        for file in files:
            if not file.content_type.startswith("image/"):
                return JSONResponse(status_code=400, content={"status": "error", "message": f"{file.filename} is not a valid image."})

            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as f:
                f.write(await file.read())

            # Save metadata in DB
            new_upload = Upload(filename=file.filename, uploaded_by=current_user.username, timestamp=datetime.now())
            db.add(new_upload)
            uploaded_files.append(file.filename)

        db.commit()
        return {"status": "success", "message": f"{len(uploaded_files)} images uploaded successfully.", "files": uploaded_files}

    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/add_user")
def add_user(name: str, age: int):
    user = {"name": name, "age": age}
    result = collection.insert_one(user)
    return {"inserted_id": str(result.inserted_id)}

# @app.post("/predict-uploaded")
# async def predict_uploaded(filename: str):
#     try:
#         file_path = os.path.join(UPLOAD_DIR, filename)
#         if not os.path.exists(file_path):
#             return {"status": "error", "message": "File not found."}

#         results = model.predict(source=file_path, conf=0.25)
#         detections = []
#         for r in results[0].boxes:
#             cls_id = int(r.cls)
#             detections.append({
#                 "class": model.names[cls_id],
#                 "confidence": float(r.conf),
#                 "bbox": r.xyxy.tolist()[0]
#             })

#         if not detections:
#             return {"status": "error", "message": "No objects detected."}

#         return {"status": "success", "detections": detections}

#     except Exception as e:
#         return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
# @app.post("/predict-uploaded")
# async def predict_uploaded(filename: str = Form(...)):
#     image_path = os.path.join(UPLOAD_DIR, filename)

#     if not os.path.exists(image_path):
#         return JSONResponse(content={"error": "File not found"}, status_code=404)

#     results = model(image_path)
#     predictions = results[0].boxes

#     if len(predictions) == 0:
#         return {
#             "status": "failed",
#             "message": "No objects detected",
#             "image_url": f"/{image_path}"  # Still return original image
#         }

#     objects = []
#     for box in predictions:
#         cls_id = int(box.cls)
#         objects.append({
#             "class": model.names[cls_id],
#             "confidence": round(float(box.conf) * 100, 2)
#         })

#     # Save annotated image
#     annotated_path = os.path.join(RESULTS_DIR, filename)
#     results[0].save(filename=annotated_path)

#     return {
#         "status": "success",
#         "predictions": objects,
#         "image_url": f"/{annotated_path}"
#     }

@app.post("/predict-uploaded")
async def predict_uploaded(data: dict, current_user: User = Depends(get_current_user)):
    filename = data.get("filename")
    if not filename:
        return {"status": "failed", "message": "Filename missing"}

    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        return {"status": "failed", "message": "File not found"}

    results = model.predict(file_path, save=True, project=RESULTS_DIR, name="detect_uploaded", exist_ok=True)

    detections = []
    for r in results[0].boxes.data.tolist():
        x1, y1, x2, y2, score, cls_id = r
        if score < 0.5:
            continue
        cls_name = model.names[int(cls_id)]
        detections.append({
            "class": cls_name,
            "confidence": float(score),
            "description": f"Details for {cls_name}",
            "price": "₹500",
            "location": "Section A"
        })

    if not detections:
        return {"status": "failed", "message": "No objects detected"}

    return {
        "status": "success",
        "image_url": f"/{RESULTS_DIR}/detect_uploaded/{os.path.basename(file_path)}",
        "detections": detections
    }

@app.get("/users")
def get_users():
    users = list(collection.find({}, {"_id": 0}))
    return users

# //close


@app.get("/history")
def get_history(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    detections = db.query(Detection).filter(Detection.user_id == current_user.id).all()
    return [{
        "class": d.class_name,
        "score": d.confidence,
        "bbox": json.loads(d.bbox),
        "time": d.timestamp
    } for d in detections]
def get_detection_history(db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    detections = db.query(Detection).filter(Detection.user_id == current_user.id).all()
    return detections
def get_full_detection_history(db: Session = Depends(get_db)):
    results = db.query(User.username, Detection.class_name, Detection.timestamp)\
                .join(User, User.id == Detection.user_id).all()
    return results
def userwise_detections(db: Session = Depends(get_db)):
    return db.query(User.username, Detection.class_name, Detection.timestamp)\
             .join(User, User.id == Detection.user_id).all()



        # close update

#         predictions.append({
#             "class": class_name,
#             "score": confidence,
#             "bbox": [x1, y1, x2, y2],
#             "description": meta.get("description", "N/A"),
#             "use": meta.get("use", "N/A"),
#             "price": meta.get("price", "N/A"),
#             "location": meta.get("location", "N/A")
#         })

#  return {"results": predictions}

# close update

# python
# Copy
# Edit
#  for r in results:
#     for box in r.boxes:
#         cls_id = int(box.cls[0])
#         score = float(box.conf[0])
#         bbox = box.xyxy[0].tolist()

#         predictions.append({
#             "class": model.names[cls_id],
#             "score": score,
#             "bbox": bbox
#         })

#         # Save detection to DB
#         db_detection = Detection(
#             class_name=model.names[cls_id],
#             confidence=score,
#             bbox=str(bbox),
#             filename=filename
#         )
#       db.add(db_detection)

#    db.commit()

#  if not predictions:
#     return JSONResponse({"message": "No objects detected."})

#  return {"results": predictions}