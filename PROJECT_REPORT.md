## 🧭 Objective

To build a simplified version of a production-ready face recognition system incorporating:
- Facial identity verification
- Spoofing detection (liveness check)
- Scalable and fast vector similarity search
- Encrypted embeddings for enhanced security
- A RESTful API interface 

---

## 🔧 Tools & Libraries

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

## 🔄 Pipeline Overview

1. **Input Image**
        ↓
2. **Face Detection & Alignment**
        ↓
3. **Liveness Detection**
        ↓
4. **ArcFace Embedding**
        ↓
5. **FAISS Index Search**
        ↓
6. **Identity Match Output**

---

## Ground Truth Employee Face Database Simulation

To simulate a realistic employee face database, we leveraged two versions of the Labeled Faces in the Wild (LFW) dataset:

1. **Sklearn LFW People Dataset:**
   This dataset consists of tightly cropped, preprocessed, and mostly grayscale face images. We used it to represent the company's employee face database — a controlled repository of known identities with clean, aligned face images, ready for embedding extraction. These images act as the *ground truth* references for identity comparison during face search.

2. **Original LFW Dataset (lfw\_funneled):**
   Unlike the sklearn version, the original LFW dataset contains wider, colored images that include faces along with surrounding context such as background and partial body features. These images simulate real-world camera feeds typically encountered in access control scenarios like computer login or physical building entry. Because these images are unprocessed and contain additional visual information, they are ideal for testing the full application pipeline — starting from face detection, followed by alignment, embedding, and finally identity recognition.

### Rationale and Considerations

While the sklearn LFW dataset is grayscale, we acknowledge that color information is valuable, especially since the original images used for testing are colored. However, we chose to use the sklearn cropped faces as the database because it is illogical to run detection on the original images to generate cropped faces and then test recognition on those very same processed images. This approach could bias the results and does not reflect realistic use cases.

Instead, the original LFW images serve as *input samples* to the detection and recognition pipeline, ensuring an end-to-end evaluation under conditions closer to operational environments, where faces appear within a varied scene.

### Note

In practical applications, companies typically generate their ground truth cropped face images using AI-based face detection and alignment tools—similar to the methods used in this project. These curated reference images are then compared against new, unseen images captured from live camera feeds during authentication or access control.

---

## ⚙️ Dependency Choice & the FAISS-GPU Dilemma

During the dependencies decision phase, we initially aimed for maximum optimization — the plan was to integrate `faiss-gpu` alongside `insightface`, `onnxruntime-gpu`, and CUDA-enabled PyTorch for a fully accelerated pipeline. Then we found out... things don’t always play that nicely together.

FAISS-GPU isn’t available via `pip` (only through `conda` or manual compilation), so we considered setting up a `conda` environment to accommodate it. However, this led to some frustrating runtime conflicts — most notably the infamous **OpenMP error**:

> `OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.`

This happens when multiple libraries (like `onnxruntime`, FAISS, or PyTorch) try to load and initialize different versions of the **OpenMP runtime** (typically `libiomp5md.dll`), which causes hard crashes due to duplicate initialization. It’s a known headache when combining GPU-accelerated packages from different sources.

To avoid falling into this kind of dependency chaos and to keep the project simple and reproducible, we chose to use the `faiss-cpu` version installed via `pip`. It integrates cleanly with the rest of the pipeline and performs well for small to medium datasets.

That said, **FAISS-GPU remains the preferred option** for large-scale or high-speed systems. In the future, we plan to explore containerized setups or more controlled environments that allow stable use of `faiss-gpu` without runtime issues.
If you already have a working setup (or the patience to build one), we **highly encourage** you to switch to `faiss-gpu` — the pipeline is fully compatible and should run seamlessly with the faster backend.

---

## 🔍 Building the FAISS Face Index

To enable fast and accurate face identification, we implemented a face vector indexing pipeline using **FAISS** with cosine similarity. The process involved embedding all employee face images into high-dimensional vectors and indexing them for efficient similarity search.

We used a **pre-trained ArcFace ResNet100 model in ONNX format**, hosted on Hugging Face by [FoivosPar/Arc2Face](https://huggingface.co/FoivosPar/Arc2Face/blob/da2f1e9aa3954dad093213acfc9ae75a68da6ffd/arcface.onnx). This model was chosen because it is compatible with the embedding model used by the InsightFace `"buffalo_l"` pack, ensuring coherence between detection-based and standalone embedding flows.

The face images used for indexing are assumed to be already **cropped and aligned**, and are stored under `data/employee_faces/`, with each subfolder named after the corresponding identity and containing multiple face images of that person. We created this image database using the `LFW People` dataset from `scikit-learn` (explained in a separate section).

The indexing process includes:

* Loading all the identity images.
* Preprocessing and embedding each image with the ArcFace ONNX model.
* Normalizing the resulting embeddings.
* Adding the embeddings to a **FAISS IndexFlatIP** index for cosine similarity search.
* Saving the resulting index and label mappings to disk (`face_index.faiss`, `face_labels.pkl`).

This setup allows the system to quickly retrieve top-matching identities when a new face embedding is queried.

> **Note:** In future iterations of this project, we plan to experiment with alternative embedding models (e.g., other ArcFace variants, CosFace, or even vision transformers) and evaluate their performance in terms of speed and recognition accuracy.

---

## 🧑‍💼 Face Detection and Alignment

For face detection and alignment, we utilized the InsightFace library, specifically the **buffalo\_l** model. This model provides a comprehensive pipeline that combines both high-accuracy face detection and precise facial landmark alignment, which is essential for reliable embedding extraction.

The **buffalo\_l** model is an all-in-one solution that simplifies preprocessing by automatically detecting faces in images and aligning them to a canonical pose. This alignment step normalizes facial orientation and scale, improving the consistency and robustness of downstream face embedding generation.

By leveraging **buffalo\_l**, we avoided the need to train separate detection or alignment models and significantly reduced preprocessing complexity. The model supports GPU acceleration (when available), allowing faster inference, which is beneficial for scaling the face recognition pipeline.

This approach ensures that each face is properly localized and standardized before generating embeddings, which ultimately improves the accuracy and reliability of the face search system.

---

## 📊 Challenges

- faiss-gpu integration with the current setup
- Finding an accurate but lightweight liveness detection model
- Maintaining embedding consistency across preprocessing steps
- Optimizing search performance with FAISS index update/rebuilds

---

## ✅ Future Work

- Deploy via Docker + HTTPS endpoint
- Train a lightweight custom liveness model from scratch
- Use secure user registration via face ID and public key encryption
- Face Enrollment Endpoint: Allow admins to upload new faces to dynamically update the FAISS index and labels.
- Webcam-based Client: A simple Python or JS interface that captures frames from a webcam and POSTs them to the /recognize_face API.
- Metrics Dashboard: A GET /metrics endpoint that shows: Number of queries, Recognition success rate, Most recognized users
- Rebuild Index on Demand: Add a /rebuild_index endpoint or CLI command that reruns build_index.py from inside the app.

---