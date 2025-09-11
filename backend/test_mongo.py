from pymongo import MongoClient

# Replace <username>, <password>, <clustername> with your MongoDB Atlas credentials
uri = "mongodb+srv://Payal:Payal%40123@cluster0.ptv1b7b.mongodb.net/?retryWrites=true&w=majority"

# Connect to MongoDB
client = MongoClient(uri, serverSelectionTimeoutMS=5000)

# List databases to test connection
print(client.list_database_names())
