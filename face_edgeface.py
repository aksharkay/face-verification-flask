import models
import json
import cv2
import time
import torch
import numpy as np
from pathlib import Path
from backbones import get_model
from face_alignment import align
from torchvision import transforms

#Capture face image
def captureImage(cam):

    result = {
        "image": [],
        "message": "",
        "status": False
    }

    try:
        cam_result, image = cam.read()

        if not cam_result:
            result["message"] = "No image detected in current frame"

            return result
        
        image = cv2.flip(image, 1)
        result["image"] = image
        result["message"] = "Image captured successfully"
        result["status"] = True

    except Exception as e:
        result["message"] = str(e)

    return result

# generate embeddings for a face image
def generateEmbedding(img, model):

    result = {
        "status": False,
        "message": "Error occurred in embedding generation",
        "embedding": None
    }

    try:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        aligned = align.get_aligned_face(img)
        transformed_input = transform(aligned).unsqueeze(0)
        embedding_obj = model(transformed_input)
        embedding = embedding_obj.detach().numpy()

        result["status"] = True
        result["message"] = "Embedding generated successfully"
        result["embedding"] = embedding

    except Exception as e:
        result["message"] = str(e)

    return result

# find distance between 2 face embeddings
def findMatch(image, source_embedding, verification_model, threshold):

    result = {
        "distance": -1,
        "match": False,
        "message": "Match not found"
    }

    try:
        embeddings_result = generateEmbedding(image, verification_model)

        if not embeddings_result["status"]:
            result["status"] = embeddings_result["status"]

            return result
        
        target_embedding = embeddings_result["embedding"]
        diff = np.subtract(source_embedding, target_embedding)
        dist = np.sum(np.square(diff), 1)
        match = bool(dist <= threshold)
        
        result["match"] = match
        result["message"] = "Match found" if match else "Match not found"
        result["distance"] = dist[0]

    except Exception as e:
        result["message"] = str(e)

    return result

# Register new face embeddings in DB
def fetchEmbeddings(cam):

    edgeface_verification_models = models.edgeface_verification_models
    embeddings_dict = {}
    result = {
        "status": False,
        "embeddings": {},
        "message": "An error occurred"
    }

    try:
        cam_result = captureImage(cam)

        if not cam_result["status"]:
            result["message"] = cam_result["message"]

            return result

        image =  cam_result["image"]

        for model in edgeface_verification_models:
            embeddings_result = generateEmbedding(image, model)

            if not embeddings_result["status"]:
                result["message"] = embeddings_result["message"]

                return result
            
            embeddings = embeddings_result["embeddings"]
            embeddings_json = json.dumps(embeddings)
            embeddings_dict[model["model"]] = embeddings_json
        
        result["embeddings"] = embeddings_dict
        result["status"] = True
        result["message"] = "Embeddings generated successfully"

    except Exception as e:
        result["message"] = str(e)

    return result

# Verify face embeddings
def verify(source_embeddings, cam, edgeface_models):

    iter_count = 0
    iteration_result = {
        "match": False,
        "message": "Match Not Found"
    }

    edgeface_verification_models = models.edgeface_verification_models

    try:
        while iter_count < 5:
            
            match_count = 0
            start_time = time.perf_counter()
            cam_result = captureImage(cam)
            end_time = time.perf_counter()
            print(f"Camera Operation Time: {end_time - start_time}")

            if not cam_result["status"]:
                iteration_result["message"] = cam_result["message"]
                print(f"Iteration {iter_count}: {iteration_result}")
                iter_count += 1
                continue

            image = cam_result["image"]
            
            try:
                start_time = time.perf_counter()
                result1 = findMatch(image, source_embeddings[edgeface_verification_models[0]["model"]], edgeface_models[edgeface_verification_models[0]["model"]], edgeface_verification_models[0]["threshold"])
                result2 = findMatch(image, source_embeddings[edgeface_verification_models[1]["model"]], edgeface_models[edgeface_verification_models[1]["model"]], edgeface_verification_models[1]["threshold"])
                result3 = findMatch(image, source_embeddings[edgeface_verification_models[2]["model"]], edgeface_models[edgeface_verification_models[2]["model"]], edgeface_verification_models[2]["threshold"])
                end_time = time.perf_counter()
                print(f"Verification Operation Time: {end_time - start_time}")

                iteration_result[edgeface_verification_models[0]["model"]] = result1
                iteration_result[edgeface_verification_models[1]["model"]] = result2
                iteration_result[edgeface_verification_models[2]["model"]] = result3

                if result1["match"]:
                    match_count += 1

                if result2["match"]:
                    match_count += 1

                if result3["match"]:
                    match_count += 1

                if match_count > 1:
                    iteration_result["match"] = True
                    iteration_result["message"] = "Face verified with " + str(match_count) + " model(s)"
                    break
                
            except Exception as e:
                iteration_result["message"] = str(e)

            print(f"Iteration {iter_count}: {iteration_result}")
            iter_count += 1

    except Exception as e:
        iteration_result["message"] = str(e)

    return iteration_result