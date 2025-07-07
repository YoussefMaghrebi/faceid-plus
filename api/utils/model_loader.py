import onnxruntime as ort

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