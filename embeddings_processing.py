import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from data import MedTable, Entity, EntityDefinition
import os
import torch



# Config
#model_name_full = "all-MiniLM-L6-v2"  # swap this to compare models
model_name_full = "neuml/pubmedbert-base-embeddings"
model_name = model_name_full.split('/')[-1]
index_path = f"faiss_{model_name}.index"
cuis_path = f"cuis_{model_name}.npy"
# Hardware
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
print(f"CUDA available: {torch.cuda.is_available()}")


# Check for existing index and CUI mapping
print(f"Checking for existing FAISS index at {index_path}...")
print(f"Checking for existing CUI mapping at {cuis_path}...")
if os.path.exists(index_path) and os.path.exists(cuis_path):
    print(f"Loading existing index and CUI data from {index_path}...")
    index = faiss.read_index(index_path)
    cuis = np.load(cuis_path, allow_pickle=True)
    model = SentenceTransformer(model_name_full)
else:
    print("No existing index found. Starting full embedding process...")
    
    # Load corpus
    med_table = MedTable.load()
    cuis_list = []
    embedding_documents = []
    
    for cui, ent in med_table.entities.items():
        print(f"Readying CUI [{cui}] definitions for embedding", end="\r", flush=True)
        cuis_list.append(cui)
        # Combine all available definitions for this CUI
        defs = " ".join(en_def.definition for en_def in ent.definitions)
        embedding_documents.append(defs)

    cuis = np.array(cuis_list)

    # Embed
    model = SentenceTransformer(model_name_full)
    embeddings = model.encode(
        embedding_documents,
        show_progress_bar=True,
        batch_size=256,
        normalize_embeddings=True
    )

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Save for future use
    faiss.write_index(index, index_path)
    np.save(cuis_path, cuis)
    print("\nProcessing complete. Files saved for future runs.")
# # --- Load corpus ---
# med_table = MedTable.load()
# cuis = []
# embedding_documents = []
# for cui, ent in med_table.entities.items():
#     print(f"Readying CUI [{cui}] definitions for embedding", end="\r", flush=True)
#     cuis.append(cui)
#     defs = ent.definitions
#     embedding_documents.append(" ".join(en_def.definition for en_def in defs))

# cuis = np.array(cuis)

# # --- Embed ---

# model_name_full = "neuml/pubmedbert-base-embeddings"
# model_name = model_name_full.split('/')[-1]
# index_path = f"faiss_{model_name}.index"
# cuis_path = f"cuis_{model_name}.npy"
# print(f"Checking for existing FAISS index at {index_path}...")
# cuis_path = f"cuis_{model_name}.npy"
# print(f"Checking for existing CUI mapping at {cuis_path}...")
# index = faiss.read_index(index_path) 
# cuis = np.load(cuis_path, allow_pickle=True)
# model = SentenceTransformer(model_name_full)
# # batch_size controls memory usage; 240k docs will take a few minutes on CPU
# embeddings = model.encode(
#     embedding_documents,
#     show_progress_bar=True,
#     batch_size=256,
#     normalize_embeddings=True,  # needed for cosine similarity via inner product
# )

# # --- Build FAISS index ---
# dim = embeddings.shape[1]
# # With normalized vectors, inner product == cosine similarity
# index = faiss.IndexFlatIP(dim)
# index.add(embeddings) # type: ignore

def dense_retrieve(query: str, k: int = 10):
    query_vec = model.encode([query], normalize_embeddings=True)
    scores, indices = index.search(query_vec, k) # type: ignore
    return [(cuis[i], float(s)) for i, s in zip(indices[0], scores[0])]


# --- Serialize for reuse ---
try:
    index = faiss.index_gpu_to_cpu(index) # type: ignore
except:
    pass
model_name = model_name_full.split("/")[-1] # Ensure when writing model publisher doesn't get interpreted as directory
faiss.write_index(index, f"faiss_{model_name}.index")
np.save(f"cuis_{model_name}.npy", cuis)

# --- Sanity check ---
if __name__ == "__main__":
    if 'embedding_documents' not in locals():
        print("Fetching documents from MedTable for display...")
        temp_med_table = MedTable.load()
        _, embedding_documents = temp_med_table.encoding_data()
    while True:
        query = input("Enter a query (e.g. 'heart attack') and press Enter: ")
        print(f"Dense retrieval ({model_name_full}):")
        if not query.strip():
            break
        for cui, score in dense_retrieve(query, k=5):
            idx = np.where(cuis == cui)[0][0]
            print(f"  {cui} ({score:.4f}): {embedding_documents[idx][:120]}")
