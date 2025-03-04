verification_thresholds = {
    "Facenet": {
        "cosine": 0.40,
        "euclidean": 10,
        "euclidean_l2": 0.80
    },
    "Facenet512": {
        "cosine": 0.30,
        "euclidean": 23.56,
        "euclidean_l2": 1.04
    },
    "ArcFace": {
        "cosine": 0.68,
        "euclidean": 4.15,
        "euclidean_l2": 1.13
    },
    "edgeface_base": {
        "euclidean_l2": 1.0
    },
    "edgeface_s_gamma_05": {
        "euclidean_l2": 1.0
    },
    "edgeface_xs_q": {
        "euclidean_l2": 1.0
    }
}

deepface_verification_models = [
    {
        "model": "Facenet",
        "model_column": "facenet",
        "normalization": "Facenet2018",
        "detector": "opencv",
        "distance_metric": "euclidean_l2",
        "threshold": verification_thresholds["Facenet"]["euclidean_l2"]
    },
    {
        "model": "Facenet512",
        "model_column": "facenet512",
        "normalization": "Facenet2018",
        "detector": "opencv",
        "distance_metric": "euclidean_l2",
        "threshold": verification_thresholds["Facenet512"]["euclidean_l2"]
    },
    {
        "model": "ArcFace",
        "model_column": "arcface",
        "normalization": "ArcFace",
        "detector": "opencv",
        "distance_metric": "euclidean_l2",
        "threshold": verification_thresholds["ArcFace"]["euclidean_l2"]
    }
]

edgeface_verification_models = [
    {
        "model": "edgeface_base",
        "model_column": "edgeface_base",
        "threshold": verification_thresholds["edgeface_base"]["euclidean_l2"]
    },
    {
        "model": "edgeface_s_gamma_05",
        "model_column": "edgeface_s_gamma_05",
        "threshold": verification_thresholds["edgeface_s_gamma_05"]["euclidean_l2"]
    },
    {
        "model": "edgeface_xs_q",
        "model_column": "edgeface_xs_q",
        "threshold": verification_thresholds["edgeface_xs_q"]["euclidean_l2"]
    }
]

detection_models = [
    "opencv",
    "retinaface",
    "mtcnn",
    "ssd",
    "dlib"
]