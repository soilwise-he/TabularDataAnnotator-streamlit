import numpy as np, faiss
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('models_local/all-MiniLM-L6-v2')
index = faiss.read_index('data/vocabulary-models_local--all-MiniLM-L6-v2.index')
vec = model.encode('soil moisture', normalize_embeddings=True).astype('float32').reshape(1,-1)
D, I = index.search(vec, 10)
print('D', D[0][:10])
print('I', I[0][:10])
