from flask import Flask
import time
import db
import deepface_ops
import edgeface_ops
import cv2
import models
import torch
from backbones import get_model

app = Flask(__name__)

db_connection = db.connect().get("connection") # connect to DB
cam = cv2.VideoCapture(0) # initialize camera

# load EdgeFace models
edgeface_verification_models = models.edgeface_verification_models
edgeface_base_model = None
edgeface_s_gamma_05_model = None
edgeface_xs_q_model = None
edgeface_models = {}

def loadModels():

    global edgeface_base_model
    arch = edgeface_verification_models[0]["model"]
    edgeface_base_model = get_model(arch)
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

# Verify face embeddings using either DeepFace or EdgeFace models
@app.route('/face/<model>/verify/<uid>')
def verify(model, uid):

    function_start = time.perf_counter()

    # default response
    result = {
        "match": False,
        "message": "Invalid model"
    }

    # return default response if invalid model is passed in URL
    if model != "deepface" and model != "edgeface":
        return result

    sub_models = models.deepface_verification_models if model == "deepface" else models.edgeface_verification_models # fetch face verification models associated with the selected library
    db_result = db.fetchEmbeddings(db_connection, uid, sub_models) # fetch face embeddings associated with this UID from DB

    # proceed only if DB fetch operation was successful
    if(db_result["status"]):
        source_embeddings = db_result["embeddings"]

        # compare source and target face embeddings
        if model == "deepface":
            result = deepface_ops.verify(source_embeddings, cam, sub_models)
        else:
            result = edgeface_ops.verify(source_embeddings, cam, sub_models, edgeface_models)
    else:
        result["message"] = db_result["message"]

    function_finish = time.perf_counter()
    result["total_execution_time"] = round(function_finish-function_start, 2)

    return result

# Registering new UID's face embeddings to DB
@app.route('/face/register/<uid>')
def register(uid):

    function_start = time.perf_counter()
    # default response
    result = {
        "status": False,
        "message": "An error occurred"
    }

    # fetch both face verification models
    deepface_sub_models = models.deepface_verification_models
    edgeface_sub_models = models.edgeface_verification_models

    deepface_result = deepface_ops.fetchEmbeddings(cam, deepface_sub_models)
    edgeface_result = edgeface_ops.fetchEmbeddings(cam, edgeface_sub_models, edgeface_models)

    if deepface_result["status"] and edgeface_result["status"]:
        result = db.writeEmbeddings(db_connection, uid, deepface_result["embeddings"], edgeface_result["embeddings"], deepface_sub_models, edgeface_sub_models)
    else:
        result["message"] = f"DeepFace: {deepface_result['message']}  EdgeFace: {edgeface_result['message']}"

    function_finish = time.perf_counter()
    result["total_execution_time"] = round(function_finish-function_start, 2)

    return result

# Re-registering existing UID's face embeddings to DB (WIP)
@app.route('/face/update/<uid>')
def update(uid):

    function_start = time.perf_counter()
    result = deepface_ops.fetchEmbeddings(cam)

    if result["status"]:
        result = db.updateEmbeddings(uid, result["embeddings"], db_connection)

    function_finish = time.perf_counter()
    result["total_execution_time"] = round(function_finish-function_start, 2)

    return result

if __name__ == "__main__":
    loadModels()
    app.run()