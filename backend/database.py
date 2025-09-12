import os
import motor.motor_asyncio
# update
from motor.motor_asyncio import AsyncIOMotorClient
import certifi
# close

MONGO_URL = os.getenv(
    "MONGO_URL",
    # put your own SRV here or load from .env
    # "mongodb+srv://Payal:Payal%40123@cluster0.ptv1b7b.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    "mongodb+srv://Payal:Payal%40123@cluster0.ptv1b7b.mongodb.net/fastapi_db?retryWrites=true&w=majority&tls=true"
)

# client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URL)
# db = client["fastapi_db"]            # <- the DB you’re using (see your Atlas screenshot)
# users_collection = db["users"]
# detections_collection = db["detections"]  # name it whatever you prefer (e.g., detections)

# update
client = AsyncIOMotorClient(
    MONGO_URL,
    tls=True,
    tlsCAFile=certifi.where()
)
# close

# Optional but recommended: create indexes once at startup
async def ensure_indexes():
    await users_collection.create_index("email", unique=True)
    await detections_collection.create_index([("user_id", 1), ("created_at", -1)])

# update
db = client["fastapi_db"]
users_collection = db["users"]
detections_collection = db["detections"]
# close




