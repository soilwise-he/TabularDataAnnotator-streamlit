import numpy as np, pandas as pd, faiss
from sentence_transformers import SentenceTransformer
from zenodo_autopilot import _match_keyword_candidates_direct
index_lib = faiss.read_index('data/vocabulary-models_local--all-MiniLM-L6-v2.index')
metadata_lib = np.load('data/vocabulary-models_local--all-MiniLM-L6-v2-meta.npz', allow_pickle=True)
raw_metadata = metadata_lib['id_to_meta'].item()
vocab_dict = {}
for identifier, item in raw_metadata.items():
    metadata = dict(item)
    metadata.setdefault('uri', metadata.get('concept_uri'))
    metadata.setdefault('source', metadata.get('source_id'))
    metadata.setdefault('QC_label', metadata.get('label_type'))
    metadata.setdefault('definition', metadata.get('definition_en'))
    metadata['source'] = str(metadata.get('source') or '').removeprefix('local_vocabularies:')
    vocab_dict[int(identifier)] = metadata
model = SentenceTransformer('models_local/all-MiniLM-L6-v2')
df = pd.DataFrame({'name':['sample_date','soil moisture','nitrogen','ph','temperature'],'description':['date when sample was taken','water content in soil','amount of nitrogen in soil','soil reaction value','temperature of soil layer']})
print(_match_keyword_candidates_direct(df, index_lib, vocab_dict, model, k_nearest=10))
