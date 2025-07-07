from fastapi import APIRouter, UploadFile, Request, Query, File
from api.utils.faiss_index import search_face
from api.utils.face_utils import embed_face, detect_faces
from collections import Counter
import numpy as np
import cv2


router = APIRouter()

@router.post("/recognize_face")
async def recognize_face(request: Request, file: UploadFile = File(...),
                         threshold: float = Query(0.5, ge=0.0, le=1.0, description="Confidence threshold between 0 and 1")):

    session = request.app.state.session
    input_name = request.app.state.input_name
    detection_model = request.app.state.detection_model
    index = request.app.state.index
    labels = request.app.state.labels

    # Read uploaded image bytes
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)   # in BGR format

    # Detect the faces in an image
    faces = detect_faces(image, detection_model)  

    # For better embedding consistency, convert to grayscale since the FAISS index was built on grayscale images
    faces = [cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) for face in faces]

    # search for identity matches 
    results = []
    for face in faces:

        # create embedding vector for each detected face
        embedding = embed_face(session, input_name, face)

        # retrieve matches using FAISS search
        scores, indices = search_face(index, embedding.reshape(1, -1))

        # Collect matches with labels and scores
        matches = [(labels[i], float(scores[0][j])) for j, i in enumerate(indices[0])]

        # Tally votes
        vote_counter = Counter([name for name, _ in matches])

        # Select most frequent identity
        majority_identity, vote_count = vote_counter.most_common(1)[0]

        # Find the best score among the majority group
        majority_scores = [score for name, score in matches if name == majority_identity]
        top_score = max(majority_scores) if majority_scores else 0.0

        # Format the final response
        results.append({
            "identity": majority_identity,
            "verified": top_score>threshold,
            "score": top_score,
            "candidates": [{"name": name, "score": score} for name, score in matches]
        })

    return {"results": results}
