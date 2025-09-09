# from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form
# from fastapi.security import OAuth2PasswordRequestForm
# from fastapi.middleware.cors import CORSMiddleware
# from backend.database import users_collection, detections_collection
# from backend.schemas import UserRegister, UserOut, Token
# from bson import ObjectId
# import json, uuid, shutil, os
# from datetime import datetime
# from backend.database import users_collection, detections_collection
# from backend.auth import get_password_hash, verify_password, create_access_token, get_current_user
# from backend.models import user_helper, detection_helper


# app = FastAPI(title="FastAPI + MongoDB Atlas", version="1.0.0")

# # CORS for local dev
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Change to your frontend origin in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# UPLOAD_DIR = "uploads"
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# @app.get("/health")
# async def health():
#     return {"status": "ok"}

# # Register
# @app.post("/auth/register", response_model=UserOut)
# async def register(payload: UserRegister):
#     existing = await users_collection.find_one({"email": payload.email})
#     if existing:
#         raise HTTPException(status_code=400, detail="Email already registered")

#     user_doc = {
#         "email": payload.email,
#         "full_name": payload.full_name,
#         "hashed_password": get_password_hash(payload.password),
#         "created_at": datetime.utcnow()
#     }
#     result = await users_collection.insert_one(user_doc)
#     new_user = await users_collection.find_one({"_id": result.inserted_id})
#     return user_helper(new_user)

# # Login
# @app.post("/auth/login", response_model=Token)
# async def login(form_data: OAuth2PasswordRequestForm = Depends()):
#     user = await users_collection.find_one({"email": form_data.username})
#     if not user or not verify_password(form_data.password, user["hashed_password"]):
#         raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

#     token = create_access_token({"sub": str(user["_id"]), "email": user["email"]})
#     return {"access_token": token, "token_type": "bearer"}

# # Me
# @app.get("/auth/me")
# async def me(current_user=Depends(get_current_user)):
#     return user_helper(current_user)

# # Predict (Protected)
# @app.post("/predict")
# async def predict(file: UploadFile = File(...), current_user=Depends(get_current_user)):
#     ext = os.path.splitext(file.filename or "")[1]
#     safe_name = f"{uuid.uuid4().hex}{ext}"
#     save_path = os.path.join(UPLOAD_DIR, safe_name)

#     with open(save_path, "wb") as f:
#         shutil.copyfileobj(file.file, f)

#     # Dummy prediction (replace with model logic)
#     dummy_result = {"predictions": [{"label": "example", "confidence": 0.99}]}

#     detection_doc = {
#         "user_id": current_user["_id"],
#         "image_path": save_path,
#         "result_json": json.dumps(dummy_result),
#         "created_at": datetime.utcnow()
#     }
#     result = await detections_collection.insert_one(detection_doc)
#     new_det = await detections_collection.find_one({"_id": result.inserted_id})
#     return detection_helper(new_det)

from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from bson import ObjectId
from datetime import datetime
import os, uuid, json, shutil
from ultralytics import YOLO
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from datetime import datetime
from pymongo import MongoClient
from pydantic import BaseModel, EmailStr
from fastapi.responses import JSONResponse
from PIL import Image
import io
from fastapi import APIRouter
from datetime import datetime
# update
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.exception_handlers import RequestValidationError as ValidationErrorHandler
# vlose
from fastapi import Form
# from database import detections_collection  # MongoDB collection for detections
# from auth import get_current_user  # For authentication
# from object_info import object_info  # Import your object info dictionary
from datetime import datetime, timedelta
from backend.object_info import object_info
import jwt
from backend.database import (
    users_collection,
    detections_collection,
    ensure_indexes,
)
from backend.schemas import UserRegister, UserOut, Token, DetectionOut, DetectionList
from backend.models import user_helper, detection_helper
from backend.auth import (
    get_password_hash,
    verify_password,
    create_access_token,
    get_current_user,
)
from fastapi.security import OAuth2PasswordRequestForm

SECRET_KEY = "yoursecretkey"
ALGORITHM = "HS256"

app = FastAPI(title="FastAPI + MongoDB Atlas", version="1.1.0")

# --- FastAPI Init ---
app = FastAPI()
class User(BaseModel):
    username: str
    password: str
# Dummy database

# /update
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    # Check if it's an email validation issue
    for error in exc.errors():
        if "email" in str(error["loc"]) and error["type"] == "value_error.email":
            return JSONResponse(
                status_code=400,
                content={"detail": "❌ Invalid email format. Please enter a valid email (example@gmail.com)."}
            )
    # fallback for other validation errors
    return JSONResponse(
        status_code=422,
        content={"detail": "Invalid input. Please check your data and try again."}
    )

# close

router = APIRouter()
users_db = {}
# @app.post("/register")
# async def register(user: User):
#     return {"message": f"User {user.username} registered successfully!"}
# @app.post("/register")
# async def register(user: User):
#     if user.username in users_db:
#         raise HTTPException(status_code=400, detail="Username already exists")
#     users_db[user.username] = user.password
#     return {"message": f"User {user.username} registered successfully!"}

# @app.post("/login")
# async def login(user: User):
#     if user.username not in users_db:
#         raise HTTPException(status_code=400, detail="User not found")
#     if users_db[user.username] != user.password:
#         raise HTTPException(status_code=400, detail="Incorrect password")

 # Generate JWT Token
#     token_data = {
#         "sub": user.username,
#         "exp": datetime.utcnow() + timedelta(hours=1)
#     }
#     token = jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)
#     return {"access_token": token, "token_type": "bearer"}

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
# Mount the frontend folder

# update 
# client = MongoClient("your_mongo_atlas_connection_string")
# db = client["inventory_db"]
# detections_collection = db["detections"]
# closeupdate

app.mount("/static", StaticFiles(directory="static"), name="static")
# model = YOLO("runs/detect/train/weights/best.pt")
model = YOLO("best (5).pt")
print("Loaded YOLOv8 model classes:", model.names)


UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.on_event("startup")
async def _startup():
    await ensure_indexes()

@app.get("/health")
async def health():
    return {"status": "ok"}

# ---------- auth ----------
# Serve index.html as root
@app.get("/")
async def serve_home():
    return FileResponse("static/index.html")

@app.post("/auth/register", response_model=UserOut)
async def register(payload: UserRegister):
    existing = await users_collection.find_one({"email": payload.email})
    if existing:
        raise HTTPException(status_code=400, detail="⚠️ Email already registered")

    user_doc = {
        "email": payload.email,
        "full_name": payload.full_name,
        "hashed_password": get_password_hash(payload.password),
        "created_at": datetime.utcnow(),
    }
    ins = await users_collection.insert_one(user_doc)
    new_user = await users_collection.find_one({"_id": ins.inserted_id})
    return user_helper(new_user)
    # return {"message": "✅ User registered successfully"}

@app.post("/auth/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await users_collection.find_one({"email": form_data.username})
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid credentials")
    token = create_access_token({"sub": str(user["_id"]), "email": user["email"]})
    return {"access_token": token, "token_type": "bearer"}

@app.get("/auth/me", response_model=UserOut)
async def me(current_user=Depends(get_current_user)):
    return user_helper(current_user)

# ---------- detections ----------
# Upload an image and (for now) return a dummy detection; store in Atlas
# @app.post("/predict", response_model=DetectionOut)
# async def predict(
#     file: UploadFile = File(...),
#     current_user=Depends(get_current_user),
# ):
#     # save uploaded file
#     ext = os.path.splitext(file.filename or "")[1]
#     fname = f"{uuid.uuid4().hex}{ext}"
#     fpath = os.path.join(UPLOAD_DIR, fname)
#     with open(fpath, "wb") as f:
#         shutil.copyfileobj(file.file, f)

    # TODO: replace with your real model inference
#     result = {"predictions": [{"label": "example", "confidence": 0.99}]}

#     doc = {
#         "user_id": current_user["_id"],
#         "image_path": fpath,
#         "result_json": result,
#         "created_at": datetime.utcnow(),
#     }
#     ins = await detections_collection.insert_one(doc)
#     created = await detections_collection.find_one({"_id": ins.inserted_id})
#     return detection_helper(created)

# @app.post("/predict")
# async def predict(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
#     contents = await file.read()
#     image = Image.open(io.BytesIO(contents)).convert("RGB")
#     filename = file.filename
#     results = model(image)

#     predictions = []

#     for r in results:
#         for box in r.boxes:
#             cls_id = int(box.cls[0])
#             confidence = float(box.conf[0])
#             x1, y1, x2, y2 = box.xyxy[0].tolist()
#             class_name = model.names[cls_id]

            # Save detection in MongoDB
            # detection_data = {
            #     "username": current_user,
            #     "class_name": class_name,
            #     "confidence": confidence,
            #     "bbox": [x1, y1, x2, y2],
            #     "filename": filename,
            #     "timestamp": datetime.utcnow()
            # }
            # detections_collection.insert_one(detection_data)

            # Add object info from dictionary
    #         info = object_info.get(class_name, {})
    #         predictions.append({
    #             "class": class_name,
    #             "score": confidence,
    #             "bbox": [x1, y1, x2, y2],
    #             "description": info.get("description", "N/A"),
    #             "use": info.get("use", "N/A"),
    #             "price": info.get("price", "N/A"),
    #             "location": info.get("location", "N/A")
    #         })

    # return JSONResponse(content={"results": predictions})

# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     contents = await file.read()

    # Save temporarily
    # temp_path = f"temp_{file.filename}"
    # with open(temp_path, "wb") as f:
    #     f.write(contents)

    # Run YOLOv8
    # results = model(temp_path)

    # detections = []
    # for box in results[0].boxes:
    #     cls_id = int(box.cls[0])
    #     label = model.names[cls_id]
    #     conf = float(box.conf[0])
    #     xyxy = [float(x) for x in box.xyxy[0].tolist()]

    #     info = object_info.get(label, {})
    #     detections.append({
    #         "class": label,
    #         "score": conf,
    #         "bbox": xyxy,
    #         "description": info.get("description", "N/A"),
    #         "use": info.get("use", "N/A"),
    #         "price": info.get("price", "N/A"),
    #         "location": info.get("location", "N/A"),
    #     })

    # os.remove(temp_path)

    # detections_collection.insert_one({
    #     "username": current_user.get("username") or current_user.get("sub"),
    #     "timestamp": datetime.utcnow(),
    #     "image": file.filename,
    #     "results": detections
    # })

    # return {"results": detections}

    # update
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)   # ✅ require logged-in user
):
    contents = await file.read()

    # Save temporarily
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(contents)

    # Run YOLOv8 inference
    results = model(temp_path)

    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf = float(box.conf[0])
        xyxy = [float(x) for x in box.xyxy[0].tolist()]

        info = object_info.get(label, {})
        detections.append({
            "class": label,
            "score": conf,
            "bbox": xyxy,
            "description": info.get("description", "N/A"),
            "use": info.get("use", "N/A"),
            "price": info.get("price", "N/A"),
            "location": info.get("location", "N/A"),
        })

    os.remove(temp_path)

    # ✅ Insert detection history in its own collection
    detection_doc = {
        "username": current_user.get("username") or current_user.get("email"),
        "filename": file.filename,
        "detections": detections,
        "timestamp": datetime.utcnow()
    }
    await detections_collection.insert_one(detection_doc)

    return {"results": detections, "message": "Detection saved successfully"}

    # close


# List current user’s detection history (paginated)
@app.get("/detections", response_model=DetectionList)
async def list_detections(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    current_user=Depends(get_current_user),
):
    cursor = detections_collection.find(
        {"user_id": current_user["_id"]}
    ).sort("created_at", -1).skip(skip).limit(limit)

    items = [detection_helper(d) async for d in cursor]
    total = await detections_collection.count_documents({"user_id": current_user["_id"]})
    return {"items": items, "total": total}

# Get one detection by id (only if it belongs to the current user)
@app.get("/detections/{det_id}", response_model=DetectionOut)
async def get_detection(det_id: str, current_user=Depends(get_current_user)):
    try:
        _id = ObjectId(det_id)
    except Exception:
        raise HTTPException(400, "Invalid detection id")

    d = await detections_collection.find_one({"_id": _id, "user_id": current_user["_id"]})
    if not d:
        raise HTTPException(404, "Not found")
    return detection_helper(d)

# (Optional) delete a detection
@app.delete("/detections/{det_id}")
async def delete_detection(det_id: str, current_user=Depends(get_current_user)):
    try:
        _id = ObjectId(det_id)
    except Exception:
        raise HTTPException(400, "Invalid detection id")

    res = await detections_collection.delete_one({"_id": _id, "user_id": current_user["_id"]})
    if res.deleted_count == 0:
        raise HTTPException(404, "Not found")
    return {"deleted": True}


# update
from fastapi import Form

@app.post("/add_to_dataset")
async def add_to_dataset(
    file: UploadFile = File(...),
    class_name: str = Form(...),
    current_user: dict = Depends(get_current_user)
):
    # Define dataset directory
    dataset_dir = os.path.join("dataset_images", class_name)
    os.makedirs(dataset_dir, exist_ok=True)

    # Save image
    file_path = os.path.join(dataset_dir, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    return {"message": f"Image added to dataset under class '{class_name}'"}

# closeupdate