import onnxruntime as ort
import insightface

def load_embedding_model(model_path: str):
    """
    Load the ONNX embedding model.

    Args:
        model_path (str): Path to the ONNX model file.

    Returns:
        session (onnxruntime.InferenceSession): ONNX runtime session.
        input_name (str): Name of the input tensor for the model.
    """
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    return session, input_name

def load_detection_model(name="buffalo_l", ctx_id=0):
    """
    Load and prepare a face detection and embedding model from InsightFace.

    This function initializes a specified model pack (e.g., 'buffalo_l'), 
    which includes:
        - Face detection
        - Landmark detection
        - Face embedding extraction

    Args:
        name (str): Model pack name from InsightFace (e.g., 'buffalo_l', 'antelopev2').
        ctx_id (int): Context ID for computation device. 
                      Use 0 for GPU, -1 for CPU. Default is 0 (GPU).

    Returns:
        insightface.app.FaceAnalysis: A prepared InsightFace model instance ready for inference.
    """
    model = insightface.app.FaceAnalysis(name=name)
    model.prepare(ctx_id=ctx_id)
    return model
