from api.utils.preprocessing import preprocess_arcface
from api.utils.model_loader import load_embedding_model
from tqdm import tqdm
import numpy as np
import faiss
import pickle
import cv2
import os


def build_faiss_index(
    dataset_dir: str = "data/employee_faces",
    model_path: str = "models/arcface.onnx",       
    index_output_path: str = "face_index.faiss",
    labels_output_path: str = "face_labels.pkl"
):
    """
    Build a FAISS index and save it along with label metadata from a dataset of images.

    Args:
        dataset_dir (str): Path to the dataset folder containing subfolders of identities.
        model_path (str): Path to the ONNX ArcFace embedding model.
        index_output_path (str): Path to save the generated FAISS index file.
        labels_output_path (str): Path to save the labels pickle file.

    Returns:
        index (faiss.Index): The generated FAISS index object.
    """

    # load the embedding model 
    session, input_name = load_embedding_model(model_path)

    # generate embeddings 
    embeddings = []                           # list to hold all vector embeddings
    labels = []                               # list of people names corresponding to the vector embeddings 

    # loop over identity folders
    for folder_name in tqdm(os.listdir(dataset_dir), desc="Processing identities"):
        full_path = os.path.join(dataset_dir, folder_name)
        if os.path.isdir(full_path):

            # extract employee's name
            id = folder_name
            for image in os.listdir(full_path):

                # read the image
                image_path = os.path.join(full_path, image)
                image = cv2.imread(image_path)

                if image is None:
                    print(f"[Warning] Could not read image: {image_path}")
                    continue

                # Convert grayscale to BGR
                if len(image.shape) == 2:         # grayscale image
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

                # preprocess the image for the embedding model
                input_data = preprocess_arcface(image)

                # generate the embedding vector, shape: (512,)
                outputs = session.run(None, {input_name: input_data})
                embedding = outputs[0][0]  

                # add the vector and the corresponding label to the list
                embeddings.append(embedding)
                labels.append(id)

    # create embedding matrix, shape (N, 512) 
    embeddings = np.vstack(embeddings)  

    # normalize the embeddings to be able to use cosine similarity 
    faiss.normalize_L2(embeddings)

    # Create FAISS index using IndexFlatIP on normalized vectors is equivalent to cosine similarity
    vector_dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(vector_dimension)  
    index.add(embeddings)

    # Save index and labels
    faiss.write_index(index, index_output_path)
    with open(labels_output_path, "wb") as f:
        pickle.dump(labels, f)

    return index

def load_index(index_path: str):
    """
    Loads a FAISS index from given path.

    Args:
        index_path (str): Path to load the FAISS index.

    Returns:
        index (faiss.Index): The loaded FAISS index object.
    """
    return faiss.read_index(index_path)

def load_labels(labels_path: str):
    """
    Loads the face labels list from given path.

    Args:
        labels_path (str): Path to load the labels list.

    Returns:
        list: The loaded face labels list.
    """
    with open(labels_path, "rb") as f:
        return pickle.load(f)

def search_face(index, query_embedding, top_k=5):
    """
    Perform a similarity search on a FAISS index using cosine similarity.

    Args:
        index (faiss.Index): A FAISS index containing normalized vectors.
        query_embedding (np.ndarray): A 2D numpy array of shape (N, D) representing 
                                      one or more query vectors to search for.
        top_k (int, optional): Number of top results to return for each query vector. Defaults to 5.

    Returns:
        tuple: A tuple (scores, indices) where:
            - scores (np.ndarray): Array of similarity scores of shape (N, top_k).
            - indices (np.ndarray): Array of indices of the nearest neighbors in the index, shape (N, top_k).
    """
    faiss.normalize_L2(query_embedding)
    scores, indices = index.search(query_embedding, top_k)
    return scores, indices