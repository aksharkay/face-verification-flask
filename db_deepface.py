from dotenv import load_dotenv
import mysql.connector
import json
import os

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

def fetchEmbeddings(uid, db):

    db_result = {
        "status": False,
        "message": "No embeddings retrieved from DB",
        "embeddings": {}
    }

    try:
        cursor = db.cursor()
        query = "SELECT facenet, facenet512, arcface FROM embeddings WHERE uid = %s"
        cursor.execute(query, (uid,))
        result = cursor.fetchone()
        embeddings = {}

        if result:
            embeddings["Facenet"] = json.loads(result[0])
            embeddings["Facenet512"] = json.loads(result[1])
            embeddings["ArcFace"] = json.loads(result[2])
            db_result["status"] = True
            db_result["message"] = "Embeddings retrieved from DB successfully"
            db_result["embeddings"] = embeddings
    
    except Exception as e:
        db_result["message"] = str(e)

    return db_result

def writeEmbeddings(uid, embeddings_dict, db):

    result = {
        "status": False,
        "message": "An error occurred"
    }

    try:
        cursor = db.cursor()
        query = "INSERT INTO embeddings (uid, facenet, facenet512, arcface) VALUES (%s, %s, %s, %s)"
        cursor.execute(query, (uid, embeddings_dict["Facenet"], embeddings_dict["Facenet512"], embeddings_dict["ArcFace"]))
        db.commit()

        result["status"] = True
        result["message"] = "User face embeddings inserted into DB"

    except Exception as e:
        result["message"] = str(e)

    return result

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