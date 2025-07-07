import cv2
import numpy as np

def preprocess_arcface(image):
    """
    Preprocess an input face image for the ArcFace ONNX model.

    Steps:
      - Resize to 112x112 pixels
      - Convert BGR to RGB
      - Normalize pixel values (mean=127.5, std=128)
      - Change data layout from (H, W, C) to (C, H, W)
      - Add batch dimension and convert to float32

    Args:
        image (numpy.ndarray): Input image in BGR format (H, W, C).

    Returns:
        numpy.ndarray: Preprocessed image of shape (1, 3, 112, 112) ready for model input.
    """

    # Resize to 112x112 (input size of ResNet 100)
    img = cv2.resize(image, (112, 112))

    # Convert BGR (OpenCV default) to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize to match ArcFace training (mean=127.5, std=128)
    img = (img - 127.5) / 128.0

    # Change from (H,W,C) to (C,H,W)
    img = img.transpose(2, 0, 1)

    # Add batch dimension to obtain (1,C,H,W)
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img
