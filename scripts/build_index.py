from api.utils.faiss_index import build_faiss_index

index = build_faiss_index(
    dataset_dir="data/employee_faces",
    model_path="models/arcface.onnx",
    index_output_path="data/face_index.faiss",
    labels_output_path="data/face_labels.pkl"
)