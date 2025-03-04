from deepface import DeepFace
from deepface.modules import verification
import models
import json
import cv2
import time

# Capture face image
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

# Generate embeddings for a face image
def generateEmbeddings(image, model):

    result = {
        "status": False,
        "message": "An error occurred",
        "embeddings": []
    }

    try: 
        embeddings_obj = DeepFace.represent(
            img_path = image,
            model_name = model["model"],
            detector_backend = model["detector"],
            normalization = model["normalization"],
            anti_spoofing = True
        )

        if len(embeddings_obj) > 1:
            result["message"] = "Multiple faces detected in image"
        elif len(embeddings_obj) == 1:
            result["embeddings"] = embeddings_obj[0]["embedding"]
            result["status"] = True

    except Exception as e:
        result["message"] = str(e)

    return result

# Compare face from camera feed with source embeddings in DB
def findMatch(image, source_embedding, verification_model):

    try:
        target_embedding_obj = DeepFace.represent(
            img_path = image,
            model_name = verification_model["model"],
            detector_backend = verification_model["detector"],
            normalization = verification_model["normalization"],
            anti_spoofing = True
        )

        result = {
            "distance": -1,
            "match": False,
            "message": "Match not found"
        }

        for embedding_obj in target_embedding_obj:

            target_embedding = embedding_obj["embedding"]
            
            distance = float(verification.find_distance(
                target_embedding,
                source_embedding,
                verification_model["distance_metric"]
            ))
            match = bool(distance <= verification_model["threshold"])
            
            if match:
                result = {
                    "distance": distance,
                    "match": match,
                    "message": "Match found" if match else "Match not found"
                }

                break
            else:
                result = {
                    "distance": distance,
                    "match": False,
                    "message": "Match not found"
                }

    except Exception as e:
        result = {
            "distance": -1,
            "match": False,
            "message": str(e)
        }
        
    return result

# Generate embeddings for a face image using multiple models
def fetchEmbeddings(cam, verification_models):

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

        for model in verification_models:
            embeddings_result = generateEmbeddings(image, model)

            if not embeddings_result["status"]:
                result["message"] = embeddings_result["message"]

                return result
            
            embeddings = embeddings_result["embeddings"]
            embeddings_json = json.dumps(embeddings)
            embeddings_dict[model["model_column"]] = embeddings_json
        
        result["embeddings"] = embeddings_dict
        result["status"] = True
        result["message"] = "Embeddings generated successfully"

    except Exception as e:
        result["message"] = str(e)

    return result

# Verify face embeddings
def verify(source_embeddings, cam, verification_models):

    iter_count = 0
    iteration_result = {
        "match": False,
        "message": "Match Not Found"
    }

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
                result1 = findMatch(image, source_embeddings[verification_models[0]["model_column"]], verification_models[0])
                result2 = findMatch(image, source_embeddings[verification_models[1]["model_column"]], verification_models[1])
                result3 = findMatch(image, source_embeddings[verification_models[2]["model_column"]], verification_models[2])
                end_time = time.perf_counter()
                print(f"Verification Operation Time: {end_time - start_time}")

                iteration_result[verification_models[0]["model"]] = result1
                iteration_result[verification_models[1]["model"]] = result2
                iteration_result[verification_models[2]["model"]] = result3

                if result1["match"]:
                    match_count += 1

                if result2["match"]:
                    match_count += 1

                if result3["match"]:
                    match_count += 1

                if match_count >= 2:
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