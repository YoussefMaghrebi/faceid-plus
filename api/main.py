from fastapi import FastAPI
from api.endpoints import recognize  
from contextlib import asynccontextmanager
from api.utils.faiss_index import load_index, load_labels
from api.utils.model_loader import load_embedding_model, load_detection_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code
    print("Starting up...")
    
    # Load models & index once on startup
    embed_model_path = "models/arcface.onnx"
    detect_model_id = 'buffalo_l'
    index_path = "data/face_index.faiss"
    labels_path = "data/face_labels.pkl"

    session, input_name = load_embedding_model(embed_model_path)
    detection_model = load_detection_model(name=detect_model_id)
    index = load_index(index_path)
    labels = load_labels(labels_path)

    app.state.session = session
    app.state.input_name = input_name
    app.state.detection_model = detection_model
    app.state.index = index
    app.state.labels = labels

    yield  # this allows requests to be processed

    # Shutdown code
    print("Shutting down...")

app = FastAPI(lifespan= lifespan)
app.include_router(recognize.router)