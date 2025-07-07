import requests

image_path = "data/sample_faces/face_1.jpg"
url = "http://127.0.0.1:8000/recognize_face"

with open(image_path, "rb") as f:
    response = requests.post(url, files={"file": f})

print('response: ', response.json())