# import os
# from dotenv import load_dotenv
# from pymongo import MongoClient

# # Load environment variables from .env
# load_dotenv()

# MONGO_URI = os.getenv("MONGO_URI")
# DB_NAME = os.getenv("DB_NAME")

# # Create MongoDB client
# client = MongoClient(MONGO_URI)

# # Access the database
# db = client[DB_NAME]

# print("✅ MongoDB connected successfully!")


# from pymongo.mongo_client import MongoClient
# from pymongo.server_api import ServerApi

# uri = "mongodb+srv://Payal:Payal%40123@cluster0.ptv1b7b.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# # Create a new client and connect to the server
# client = MongoClient(uri, server_api=ServerApi('1'))

# # Send a ping to confirm a successful connection
# try:
#     client.admin.command('ping')
#     print("Pinged your deployment. You successfully connected to MongoDB!")
# except Exception as e:
#     print(e)

# start
# from pymongo.mongo_client import MongoClient
# from pymongo.server_api import ServerApi
# from dotenv import load_dotenv
# import os

# # Load .env
# load_dotenv()
# MONGO_URI = os.getenv("MONGO_URI")

# client = MongoClient(MONGO_URI, server_api=ServerApi('1'))

# db = client["Inventory"]  # choose your database name
# collection = db["users"]  # choose your collection name
# detections_collection = db["detections"]
# uploads_collection = db["uploads"]
# end

import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env file

MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    raise Exception("MONGO_URI not found in .env file")

client = MongoClient(MONGO_URI)
db = client["inventory_db"]  # Name of your database

# Collections
users_collection = db["users"]
detections_collection = db["detections"]
uploads_collection = db["uploads"]

try:
    client.admin.command('ping')
    print("✅ Connected to MongoDB Atlas successfully!")
except Exception as e:
    print("❌ Connection failed:", e)
