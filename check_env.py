import torch
import sentence_transformers
print(torch.__version__)
print(sentence_transformers.__version__)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('models_local/all-MiniLM-L6-v2')
print(model.get_sentence_embedding_dimension())
