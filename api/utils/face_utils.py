from api.utils.preprocessing import preprocess_arcface
import cv2

def embed_face(session, input_name, image):
    """
    Generate a face embedding vector from a cropped and aligned face image using an ONNX model.

    Args:
        session (onnxruntime.InferenceSession): The ONNX runtime session for the embedding model.
        input_name (str): The name of the model's input node.
        image (np.ndarray): A cropped and aligned face image as a BGR numpy array. 
                            Can be grayscale, in which case it will be converted to BGR.

    Returns:
        np.ndarray: A 1D numpy array representing the face embedding vector (e.g., 512-dimensional).
    """
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    input_data = preprocess_arcface(image)
    outputs = session.run(None, {input_name: input_data})
    embedding = outputs[0][0]
    return embedding

def detect_faces(image, detection_model):
    """
    Detect faces in an image using the provided InsightFace detection model.
    
    Args:
        image (np.ndarray): Input image (BGR).
        detection_model: Initialized FaceAnalysis model.

    Returns:
        List[np.ndarray]: List of cropped face images (BGR).
    """
    faces = detection_model.get(image)

    if not faces:
        return []

    cropped_faces = []
    for face in faces:
        x1, y1, x2, y2 = [int(coord) for coord in face.bbox]
        cropped = image[y1:y2, x1:x2]
        cropped_faces.append(cropped)

    return cropped_faces
