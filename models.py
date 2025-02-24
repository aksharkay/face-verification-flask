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
        "euclidean_l2": None
    },
    "edgeface_s_gamma_05": {
        "euclidean_l2": None
    },
    "edgeface_xs_q": {
        "euclidean_l2": None
    }
}

deepface_verification_models = [
    {
        "model": "Facenet",
        "normalization": "Facenet2018",
        "detector": "opencv",
        "distance_metric": "euclidean_l2",
        "threshold": verification_thresholds["Facenet"]["euclidean_l2"]
    },
    {
        "model": "Facenet512",
        "normalization": "Facenet2018",
        "detector": "opencv",
        "distance_metric": "euclidean_l2",
        "threshold": verification_thresholds["Facenet512"]["euclidean_l2"]
    },
    {
        "model": "ArcFace",
        "normalization": "ArcFace",
        "detector": "opencv",
        "distance_metric": "euclidean_l2",
        "threshold": verification_thresholds["ArcFace"]["euclidean_l2"]
    }
]

edgeface_verification_models = [
    {
        "model": "edgeface_base",
        "threshold": verification_thresholds["edgeface_base"]["euclidean_l2"]
    },
    {
        "model": "edgeface_s_gamma_05",
        "threshold": verification_thresholds["edgeface_s_gamma_05"]["euclidean_l2"]
    },
    {
        "model": "edgeface_xs_q",
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