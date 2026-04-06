import sys
print("Interpreter:", sys.executable)
try:
    from pymongo import MongoClient
    client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=2000)
    db = client['deepguard_ai']
    coll = db['detection_history']
    print(f"Server info: {client.server_info()}")
    print(f"Total documents: {coll.count_documents({})}")
    for doc in coll.find().limit(5):
        print(doc)
except Exception as e:
    print(f"Error: {e}")
