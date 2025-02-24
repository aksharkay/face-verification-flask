from flask import Flask
import time
import db_edgeface
import face_edgeface
import cv2
import torch
import models
from backbones import get_model

app = Flask(__name__)

db_connection = db_edgeface.connect().get("connection")
cam = cv2.VideoCapture(0)

edgeface_verification_models = models.edgeface_verification_models
edgeface_base_model = None
edgeface_s_gamma_05_model = None
edgeface_xs_q_model = None
edgeface_models = {}

def loadModels():

    global edgeface_base_model
    arch = edgeface_verification_models[0]["model"]
    edgeface_base_model = get_model()
    checkpoint_path = f'checkpoints/{arch}.pt'
    edgeface_base_model.load_state_dict(torch.load(checkpoint_path, map_location = 'cpu'))
    edgeface_base_model.eval()

    global edgeface_s_gamma_05_model
    arch = edgeface_verification_models[1]["model"]
    edgeface_s_gamma_05_model = get_model(arch)
    checkpoint_path = f'checkpoints/{arch}.pt'
    edgeface_s_gamma_05_model.load_state_dict(torch.load(checkpoint_path, map_location = 'cpu'))
    edgeface_s_gamma_05_model.eval()

    global edgeface_xs_q_model
    arch = edgeface_verification_models[2]["model"]
    edgeface_xs_q_model = get_model(arch)
    checkpoint_path = f'checkpoints/{arch}.pt'
    edgeface_xs_q_model.load_state_dict(torch.load(checkpoint_path, map_location = 'cpu'))
    edgeface_xs_q_model.eval()

    global edgeface_models
    edgeface_models = {
        edgeface_verification_models[0]["model"]: edgeface_base_model,
        edgeface_verification_models[1]["model"]: edgeface_s_gamma_05_model,
        edgeface_verification_models[2]["model"]: edgeface_xs_q_model
    }

# Face verification
@app.route('/face/verify/<uid>')
def verify(uid):

    function_start = time.perf_counter()
    source_embeddings = {}
    result = db_edgeface.fetchEmbeddings(uid, db_connection)
    print("Embeddings fetched")

    if(result["status"]):
        source_embeddings = result["embeddings"]
        result = face_edgeface.verify(source_embeddings, cam, edgeface_models)

    function_finish = time.perf_counter()
    result["total_execution_time"] = round(function_finish-function_start, 2)

    return result

# Face registration
@app.route('/face/register/<uid>')
def register(uid):

    function_start = time.perf_counter()
    result = face_edgeface.fetchEmbeddings(cam)

    if result["status"]:
        result = db_edgeface.writeEmbeddings(uid, result["embeddings"], db_connection)

    function_finish = time.perf_counter()
    result["total_execution_time"] = round(function_finish-function_start, 2)

    return result

# Face re-registration
@app.route('/face/update/<uid>')
def update(uid):

    function_start = time.perf_counter()
    result = face_edgeface.fetchEmbeddings(cam)

    if result["status"]:
        result = db_edgeface.updateEmbeddings(uid, result["embeddings"], db_connection)

    function_finish = time.perf_counter()
    result["total_execution_time"] = round(function_finish-function_start, 2)

    return result

if __name__ == "__main__":

    loadModels()
    app.run()