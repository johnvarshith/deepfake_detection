import sys
with open('test_mongo_output.txt', 'w') as f:
    f.write("Interpreter: " + sys.executable + "\n")
    try:
        from pymongo import MongoClient
        client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=2000)
        db = client['deepguard_ai']
        coll = db['detection_history']
        f.write(f"Server info: {client.server_info()}\n")
        f.write(f"Total documents: {coll.count_documents({})}\n")
        for doc in coll.find().limit(5):
            f.write(str(doc) + "\n")
    except Exception as e:
        f.write(f"Error: {e}\n")
