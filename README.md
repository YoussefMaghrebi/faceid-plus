# 🧠 FaceID+: Face Recognition with Liveness Detection and Vector Search

A minimal yet complete facial recognition system built with **ArcFace**, **FAISS**, **FastAPI**, and **liveness detection**. Designed to demonstrate core facial biometrics functionalities expected in real-world deployments (authentication, anti-spoofing, vector search), this project showcases how to build and deploy a functional system using production-relevant tools.

---

## 🚀 Features

- ✅ Face detection, alignment, and normalization
- 🔐 Liveness detection to prevent spoofing
- 🧑‍💼 Identity recognition via ArcFace embeddings
- 📦 ONNX Runtime GPU acceleration, enabling faster face recognition inference
- 📚 Vector similarity search using FAISS
- 🔐 FAISS Index encyption using AES algorithm to increase security 
- ⚡ FastAPI backend with simple HTTP interface

---

## 🛠️ Tech Stack

| Category              | Technology                      |
|----------------------|----------------------------------|
| Deep Learning        | PyTorch, ONNX                    |
| Face Recognition     | InsightFace (ArcFace backend)    |
| Liveness Detection   | Pretrained CNN or binary classifier |
| Preprocessing        | OpenCV                           |
| Vector Search        | FAISS                            |
| Encryption           | cryptography library by PyCA     |
| API Framework        | FastAPI                          |
| Language             | Python 3.12.1                    |

---

## 🧪 Example Use Cases

- Authenticate user from a face image and detect spoof attempts.
- Register new users with facial embeddings.
- Verify a query image against an indexed database.

---

## 📁 Folder Breakdown

The **Project Structure** is detailed as follows:

```
faceid-plus/
│
├── api/                          # FastAPI app
│   ├── main.py                   # Main API entrypoint
│   ├── endpoints/                # Organized endpoints
│   │   ├── recognize.py          # Face recognition endpoint
│   │   └── liveness.py           # Liveness detection endpoint
|   |                                    
│   └── utils/                    # Utility modules
│       ├── face_utils.py         # Face detection and embedding utils
│       ├── faiss_index.py        # FAISS index utils
│       ├── model_loader.py       # Model loading utils
│       ├── preprocessing.py      # Image preprocessing utils
│       └── security.py           # Security and encryption utils
│
├── models/                       # Pretrained and optimized models
│   ├── arcface.onnx              # arcface_r100_v1 embedding model
│   └── liveness_model.pth
│
├── data/                         # Sample data and FAISS index
|   ├── employee_faces/           # Image database folder
|   |   ├── 0/                    # containing 10 images of id: 0
|   |   ├── 1/                    # containing 10 images of id: 1
|   |   ...
|   |   └── 157/                  # containing 10 images of id: 157
|   | 
│   ├── faiss_index.faiss         # FAISS index of embedded vectors
│   ├── faiss_labels.pkl          # Identity labels mapping for each vector
|   |    
│   └── sample_faces/             # sample face images for testing
│       ├── face_1.jpg
│       └── face_2.jpg
│
├── scripts/                      # Local preparation and test scripts
│   ├── build_index.py            # Script to build the FAISS index 
│   ├── trace_imports.py          # Script to check the imported libraries by insightface 
│   └── test_pipeline.py
│
├── notebooks/                    # Local preparation and test notebooks
│   ├── face_detection.ipynb      # Notebook showing example face detection
│   └── image_db_creation.ipynb   # Notebook detailing the steps to create the face image dataset "employee_faces/"
|
├── project_report.md             # Technical documentation
├── README.md                     # Project overview and usage
├── requirements.txt              # Dependencies
└── .gitignore
```

---

## 📦 Installation

```bash
# Clone repo
git clone https://github.com/YoussefMaghrebi/faceid-plus.git && cd faceid-plus

# Create virtual environment
python -m venv venv && source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```
---

## 🔍 Building FAISS Index

To build the vector index used for face identification, run the following script `build_index.py` found in `scripts/`, from the root directory of the project:

```bash
python build_index.py
```

This script performs the following:

* Loads cropped and aligned face images from the `data/employee_faces/` directory.
* Embeds each image using a pre-trained ArcFace model.
* Normalizes and adds the embeddings to a FAISS index (using cosine similarity).
* Saves the index (`face_index.faiss`) and the corresponding identity labels (`face_labels.pkl`) for later search.

> **ℹ️ Note:** The employee face images should be organized in subfolders under `data/employee_faces/`, where each subfolder is named after the identity and contains multiple face images of that person. A notebook detailing this process is found in `notebooks/image_db_creation.ipynb`.

### 📌 PS:
We used the ArcFace ResNet100 ONNX model hosted on **Hugging Face** by **FoivosPar/Arc2Face**, found in the link (https://huggingface.co/FoivosPar/Arc2Face/blob/da2f1e9aa3954dad093213acfc9ae75a68da6ffd/arcface.onnx), since the original `"arcface_r100_v1"` is no longer hosted by InsightFace.
This embedding model is compatible with the ArcFace model integrated into the InsightFace `"buffalo_l"` pack and was selected for standalone embedding of already-aligned and cropped faces that don't need to undergo the detection phase.

---

## Known Issues

* Attempting to install `faiss-gpu` via `pip` is not supported, and downloading it with `conda` may lead to difficult runtime conflicts due to OpenMP (`libiomp5md.dll`) initialization errors when used alongside other GPU libraries like `onnxruntime-gpu`.
* For more details on this issue and the current workaround, please see the **Dependency Choice & the FAISS-GPU Dilemma** section in the project report.

---

## 🔒 Notes on Security

Embeddings are stored locally using FAISS with added AES encryption.

The API can be extended with HTTPS, JWT, and role-based access control.

---

## 📄 Documentation

All technical decisions and implementation details are in PROJECT_REPORT.md

---

## 👨‍💻 Author

Youssef Maghrebi