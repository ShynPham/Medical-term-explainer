import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import numpy as np
from data import MedTable
# --- Load corpus ---
med_table = MedTable.load()
cuis, documents = med_table.encoding_data()

# --- TF-IDF ---
tfidf = TfidfVectorizer(stop_words="english", max_features=50000)
tfidf_matrix = tfidf.fit_transform(documents) # sparse matrix

def tfidf_retrieve(query: str, k: int = 10):
    query_vec = tfidf.transform([query])
    scores = (tfidf_matrix @ query_vec.T).toarray().ravel() # type: ignore
    top_k = np.argpartition(scores, -k)[-k:]
    top_k = top_k[np.argsort(scores[top_k])[::-1]]
    return [(cuis[i], scores[i]) for i in top_k]

# --- BM25 ---
tokenized = [doc.lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized)

def bm25_retrieve(query: str, k: int = 10):
    scores = bm25.get_scores(query.lower().split())
    top_k = np.argpartition(scores, -k)[-k:]
    top_k = top_k[np.argsort(scores[top_k])[::-1]]
    return [(cuis[i], scores[i]) for i in top_k]

# --- Test ---
if __name__ == "__main__":
    while True:
        query = input("Enter a query (e.g. 'heart attack') and press Enter: ")
        print("TF-IDF Results:")
        for cui, score in tfidf_retrieve(query, k=5):
            print(f"  {cui} ({score:.4f}): {documents[np.where(cuis == cui)[0][0]][:100]}")

        print("\nBM25 results:")
        for cui, score in bm25_retrieve(query, k=5):
            print(f"  {cui} ({score:.4f}): {documents[np.where(cuis == cui)[0][0]][:100]}")