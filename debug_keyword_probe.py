import numpy as np, pandas as pd, faiss
from sentence_transformers import SentenceTransformer
from zenodo_autopilot import _match_keyword_candidates_direct
index = faiss.read_index('data/vocabulary-models_local--all-MiniLM-L6-v2.index')
d = np.load('data/vocabulary-models_local--all-MiniLM-L6-v2-meta.npz', allow_pickle=True)
raw = d['id_to_meta'].item()
vocab = {int(k): dict(v) for k, v in raw.items()}
print('keys', list(vocab.keys())[:3])
model = SentenceTransformer('models_local/all-MiniLM-L6-v2')
df = pd.DataFrame({'name':['sample_date','soil moisture','nitrogen','ph','temperature'],'description':['date when sample was taken','water content in soil','amount of nitrogen in soil','soil reaction value','temperature of soil layer']})
hits = _match_keyword_candidates_direct(df, index, vocab, model, k_nearest=10)
print('len', len(hits))
print(hits)
