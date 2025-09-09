from pymongo import MongoClient
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

# Connect to MongoDB
client = MongoClient(MONGO_URI)
print("✅ Connected to MongoDB!")

# Choose your database and collection
db = client["payalDB"]        # you can change this name
collection = db["students"]   # you can change this name

# Insert a test document
student = {"name": "Payal", "age": 23, "course": "Python"}
result = collection.insert_one(student)
print("Inserted document ID:", result.inserted_id)

# Fetch the document
fetched = collection.find_one({"name": "Payal"})
print("Fetched document:", fetched)
