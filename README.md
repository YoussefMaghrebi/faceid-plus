# 🧠 FaceID+: Face Recognition with Liveness Detection and Vector Search

A minimal yet complete facial recognition system built with **ArcFace**, **FAISS**, **FastAPI**, and **liveness detection**. Designed to demonstrate core facial biometrics functionalities expected in real-world deployments (authentication, anti-spoofing, vector search), this project showcases how to build and deploy a functional system using production-relevant tools.

---

## 🚀 Features

- ✅ Face detection, alignment, and normalization
- 🔐 Liveness detection to prevent spoofing
- 🧑‍💼 Identity recognition via ArcFace embeddings
- 📚 Vector similarity search using FAISS
- ⚡ FastAPI backend with simple HTTP interface
- 📦 ONNX/TorchScript export for model optimization

---

## 🛠️ Tech Stack

| Category              | Technology                      |
|----------------------|----------------------------------|
| Deep Learning        | PyTorch, ONNX, TorchScript       |
| Face Recognition     | InsightFace (ArcFace backend)    |
| Preprocessing        | RetinaFace / MTCNN, OpenCV       |
| Liveness Detection   | Pretrained CNN or binary classifier |
| API Framework        | FastAPI                          |
| Vector Search        | FAISS                            |
| Language             | Python 3.12.1                    |

---

## 🧪 Example Use Cases

- Authenticate user from a face image and detect spoof attempts.
- Register new users with facial embeddings.
- Verify a query image against an indexed database.

---

## 📁 Folder Breakdown

The **Project Structure** is detailed as follows:

faceid-plus/
│
├── api/                          # FastAPI app
│   ├── main.py                   # Main API endpoints
│   ├── endpoints/                # Organized endpoints (recognition, liveness)
│   │   ├── recognize.py
│   │   └── liveness.py
│   └── utils/                    # Helper functions
│       ├── preprocessing.py
│       ├── faiss_index.py
│       ├── model_loader.py
│       └── security.py
│
├── models/                       # Pretrained and optimized models
│   ├── arcface.onnx
│   └── liveness_model.pth
│
├── data/                         # Sample/test data and FAISS DB
│   ├── faiss_index.index
│   └── sample_faces/
│       ├── person1.jpg
│       └── person2.jpg
│
├── scripts/                      # Local test scripts and index builder
│   ├── build_index.py
│   └── test_pipeline.py
│
├── project_report.md             # Technical and development documentation
├── README.md                     # Project overview and usage instructions
├── requirements.txt              # Project dependencies
└── .gitignore

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

## 🔒 Notes on Security

Embeddings are stored locally using FAISS with optional encryption.

The API can be extended with HTTPS, JWT, and role-based access control.

---

## 📄 Documentation

All technical decisions and implementation details are in PROJECT_REPORT.md

---

## 👨‍💻 Author

Youssef Maghrebi