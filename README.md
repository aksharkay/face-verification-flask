# Face verification on Flask

This repo contains a Flask-based implementation for face verification, using 2 open-source face verification models:

1. DeepFace (https://github.com/serengil/deepface): Main file 'main_deepface.py'
2. EdgeFace (https://github.com/otroshi/edgeface): Main file 'main_edgeface.py'

## Model Evaluation on the Indian Faces Dataset

The 'EdgeFace IFD Results.xlsx' file contains the face verification results, using EdgeFace models, for all possible face pairs in the public 'Indian Faces Dataset'. The dataset can be found in the 'IndianFacesDatabase.zip' file. The analysis of the results to calculate the optimal verification threshold for each EdgeFace model is still in progress.
The 'idf_eval.py' file is used to generate the results and store them in the 'EdgeFace IFD Results.xlsx' file.
