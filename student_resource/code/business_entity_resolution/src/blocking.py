import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def get_tfidf_candidates(df1, df2, text_col='combined_text', top_k=30):
    print(f"Generating TF-IDF candidates for {len(df1)} vs {len(df2)} records...")
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=10000)
    all_texts = pd.concat([df1[text_col], df2[text_col]]).fillna("")
    vectorizer.fit(all_texts)
    
    vecs1 = vectorizer.transform(df1[text_col].fillna("")).astype(np.float32)
    vecs2 = vectorizer.transform(df2[text_col].fillna("")).astype(np.float32)
    
    # FAISS with sparse matrices is tricky, but we can convert to dense if small, 
    # or just use sklearn nearest neighbors if memory is an issue.
    # We will use exact cosine similarity via matrix multiplication for simplicity if size permits.
    # Since dataset is large, we should process in batches.
    candidates = []
    
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=top_k, metric='cosine', n_jobs=-1)
    nn.fit(vecs2)
    
    # query in batches
    batch_size = 10000
    for i in range(0, vecs1.shape[0], batch_size):
        end = min(i+batch_size, vecs1.shape[0])
        dist, inds = nn.kneighbors(vecs1[i:end])
        for j in range(end-i):
            s1_idx = i + j
            s2_indices = inds[j]
            candidates.extend([(s1_idx, s2_idx) for s2_idx in s2_indices])
    return candidates

def get_jepa_candidates(embs1, embs2, top_k=30):
    print("Generating JEPA candidates...")
    index = faiss.IndexFlatIP(embs2.shape[1])
    faiss.normalize_L2(embs2)
    index.add(embs2)
    
    faiss.normalize_L2(embs1)
    D, I = index.search(embs1, k=top_k)
    candidates = []
    for s1_idx in range(I.shape[0]):
        for j in range(top_k):
            candidates.append((s1_idx, I[s1_idx, j]))
    return candidates

def merge_candidates(cand_tfidf, cand_jepa):
    # Cand format: list of (s1_idx, s2_idx)
    s1_to_s2 = {}
    for (s1, s2) in cand_tfidf + cand_jepa:
        if s1 not in s1_to_s2:
            s1_to_s2[s1] = set()
        s1_to_s2[s1].add(s2)
    return s1_to_s2
