import numpy as np, pandas as pd, faiss
from sentence_transformers import SentenceTransformer
from zenodo_autopilot import _clean_text
index = faiss.read_index('data/vocabulary-models_local--all-MiniLM-L6-v2.index')
d = np.load('data/vocabulary-models_local--all-MiniLM-L6-v2-meta.npz', allow_pickle=True)
raw = d['id_to_meta'].item()
vocab = {int(k): dict(v) for k, v in raw.items()}
model = SentenceTransformer('models_local/all-MiniLM-L6-v2')
query = 'soil moisture'
vec = model.encode(query, normalize_embeddings=True)
vec = np.asarray(vec, dtype='float32').reshape(1, -1)
D_match, I_match = index.search(vec, k=10)
indices = I_match.flatten(); distances = D_match.flatten()
for idx, dist in zip(indices, distances):
    if idx not in vocab:
        print('MISSING', idx)
        continue
    item = dict(vocab[idx])
    print('idx', idx, 'dist', dist, 'uri', item.get('concept_uri') or item.get('uri'), 'source', item.get('source_id') or item.get('source'), 'label', item.get('label'))
