from flask import Flask
import time
import db_deepface
import face_deepface
import cv2

app = Flask(__name__)

db_connection = db_deepface.connect().get("connection")
cam = cv2.VideoCapture(0)

# Face verification
@app.route('/face/verify/<uid>')
def verify(uid):

    function_start = time.perf_counter()
    source_embeddings = {}
    result = db_deepface.fetchEmbeddings(uid, db_connection)
    print("Embeddings fetched")

    if(result["status"]):
        source_embeddings = result["embeddings"]
        result = face_deepface.verify(source_embeddings, cam)

    function_finish = time.perf_counter()
    result["total_execution_time"] = round(function_finish-function_start, 2)

    return result

# Face registration
@app.route('/face/register/<uid>')
def register(uid):

    function_start = time.perf_counter()
    result = face_deepface.fetchEmbeddings(cam)

    if result["status"]:
        result = db_deepface.writeEmbeddings(uid, result["embeddings"], db_connection)

    function_finish = time.perf_counter()
    result["total_execution_time"] = round(function_finish-function_start, 2)

    return result

# Face re-registration
@app.route('/face/update/<uid>')
def update(uid):

    function_start = time.perf_counter()
    result = face_deepface.fetchEmbeddings(cam)

    if result["status"]:
        result = db_deepface.updateEmbeddings(uid, result["embeddings"], db_connection)

    function_finish = time.perf_counter()
    result["total_execution_time"] = round(function_finish-function_start, 2)

    return result

if __name__ == "__main__":
    app.run()