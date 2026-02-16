import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

class MongoDBClient:
    def __init__(self):
        self.uri = os.getenv("MONGODB_URI")
        self.db_name = os.getenv("DATABASE_NAME", "aqi")
        self.client = MongoClient(self.uri)
        self.db = self.client[self.db_name]
        self.collection = self.db["aqi"]

    def insert_record(self, record):
        """Inserts a single record into the aqi collection."""
        return self.collection.insert_one(record)

    def insert_many(self, records):
        """Inserts multiple records into the aqi collection."""
        if not records:
            return None
        return self.collection.insert_many(records)

    def fetch_all(self):
        """Fetches all records from the aqi collection."""
        return list(self.collection.find({}, {"_id": 0}))

    def close(self):
        """Closes the MongoDB connection."""
        self.client.close()
