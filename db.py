from dotenv import load_dotenv
import mysql.connector
import json
import os

# Establish connection to MySQL DB
def connect():
    
    result = {
        "connection": "",
        "status": False,
        "message": "Some error occurred"
    }

    try:
        load_dotenv()

        host = os.getenv("HOST")
        user = os.getenv("USER")
        password = os.getenv("PASSWORD")
        database = os.getenv("DATABASE")
        
        db_connection = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )

        result["connection"] = db_connection
        result["status"] = True
        result["message"] = "Connection created successfully"
    
    except Exception as e:
        result["message"] = str(e)

    return result

# Fetch embeddings for a UID from DB
def fetchEmbeddings(db, uid, sub_models):

    db_result = {
        "status": False,
        "message": "No embeddings retrieved from DB",
        "embeddings": {}
    }

    try:
        columns = [sub_model["model_column"] for sub_model in sub_models if "model_column" in sub_model]
        columns_str = ", ".join(columns)
        cursor = db.cursor()
        query = f"SELECT {columns_str} FROM embeddings WHERE uid = %s"
        cursor.execute(query, (uid,))
        result = cursor.fetchone()
        embeddings = {}

        if result:
            embeddings[columns[0]] = json.loads(result[0])
            embeddings[columns[1]] = json.loads(result[1])
            embeddings[columns[2]] = json.loads(result[2])
            db_result["status"] = True
            db_result["message"] = "Embeddings retrieved from DB successfully"
            db_result["embeddings"] = embeddings
    
    except Exception as e:
        db_result["message"] = str(e)

    return db_result

# Create entry in DB for a UID and the corresponding face embeddings
def writeEmbeddings(db, uid, deepface_embeddings_dict, edgeface_embeddings_dict, deepface_sub_models, edgeface_sub_models):

    result = {
        "status": False,
        "message": "An error occurred"
    }

    deepface_columns = [sub_model["model_column"] for sub_model in deepface_sub_models if "model_column" in sub_model]
    edgeface_columns = [sub_model["model_column"] for sub_model in edgeface_sub_models if "model_column" in sub_model]
    columns_str = ", ".join(deepface_columns)
    columns_str += ", "
    columns_str += ", ".join(edgeface_columns)

    try:
        cursor = db.cursor()
        query = f"INSERT INTO embeddings (uid, {columns_str}) VALUES (%s, %s, %s, %s, %s, %s, %s)"
        # query = f"INSERT INTO embeddings (uid, {columns_str}) VALUES (%s, %s, %s, %s)"
        cursor.execute(query, (
            uid,
            deepface_embeddings_dict[deepface_sub_models[0]["model_column"]],
            deepface_embeddings_dict[deepface_sub_models[1]["model_column"]],
            deepface_embeddings_dict[deepface_sub_models[2]["model_column"]],
            edgeface_embeddings_dict[edgeface_sub_models[0]["model_column"]],
            edgeface_embeddings_dict[edgeface_sub_models[1]["model_column"]],
            edgeface_embeddings_dict[edgeface_sub_models[2]["model_column"]]
        ))
        # cursor.execute(query, (uid, deepface_embeddings_dict[deepface_sub_models[0]["model_column"]], deepface_embeddings_dict[deepface_sub_models[1]["model_column"]], deepface_embeddings_dict[deepface_sub_models[2]["model_column"]]))
        db.commit()

        result["status"] = True
        result["message"] = "User face embeddings inserted into DB"

    except Exception as e:
        result["message"] = str(e)

    return result

# Update existing entry in DB for a UID with new face embeddings
def updateEmbeddings(uid, embeddings_dict, db):

    db_result = {
        "status": False,
        "message": "An error occurred"
    }

    try:
        cursor = db.cursor()
        query = "SELECT uid FROM embeddings WHERE uid = %s"
        cursor.execute(query, (uid,))
        result = cursor.fetchone()

        if not result:
            db_result["message"] = "UID does not exist in DB"

            return db_result
        
        query = "UPDATE embeddings SET facenet = %s, facenet512 = %s, arcface = %s WHERE uid = %s"
        cursor.execute(query, (embeddings_dict["Facenet"], embeddings_dict["Facenet512"], embeddings_dict["ArcFace"], uid))
        db.commit()

        db_result["status"] = True
        db_result["message"] = "User face embeddings inserted into DB"

    except Exception as e:
        db_result["message"] = str(e)

    return db_result