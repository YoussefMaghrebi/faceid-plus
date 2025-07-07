from fastapi import APIRouter, UploadFile, Request, File
import cv2
import numpy as np
from api.utils.faiss_index import search_face
from api.utils.face_utils import embed_face, detect_faces
router = APIRouter()

@router.post("/recognize_face")
async def recognize_face(request: Request, file: UploadFile = File(...)):

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
        embedding = embed_face(session, input_name, face)
        scores, indices = search_face(index, embedding.reshape(1, -1))
        matches = [(labels[i], float(scores[0][j])) for j, i in enumerate(indices[0])]
        results.append({"matches": matches})

    return {"results": results}
