import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import zenodo_autopilot as z
meta = {'test': pd.DataFrame({'name':['sample_date','soil moisture','nitrogen','ph','temperature'],'description':['date when sample was taken','water content in soil','amount of nitrogen in soil','soil reaction value','temperature of soil layer']})}
matcher_module = z._load_module('dataannot_keyword_matcher_autopilot', z.KEYWORD_MATCHER_FILE)
print('matcher module loaded:', matcher_module is not None)
index_lib = None; vocab_dict = None; model = None
try:
    index_lib, vocab_dict = matcher_module.load_vocab_indexes(modelname='models_local/all-MiniLM-L6-v2')
    print('load_vocab_indexes OK', type(index_lib), len(vocab_dict))
except Exception as e:
    print('load_vocab_indexes FAIL', type(e), e)
    index_lib, vocab_dict = None, None
if index_lib is not None and vocab_dict is not None:
    try:
        if not hasattr(matcher_module, 'model') or matcher_module.model is None:
            if hasattr(matcher_module, 'load_sentence_model'):
                matcher_module.model = matcher_module.load_sentence_model(modelname='models_local/all-MiniLM-L6-v2')
        model = getattr(matcher_module, 'model', None)
        print('model from module', model is not None)
    except Exception as e:
        print('model load fail', type(e), e)
        model = None
if index_lib is None or vocab_dict is None or model is None:
    print('using fallback path')
    index_lib = faiss.read_index('data/vocabulary-models_local--all-MiniLM-L6-v2.index')
    metadata_lib = np.load('data/vocabulary-models_local--all-MiniLM-L6-v2-meta.npz', allow_pickle=True)
    raw_metadata = metadata_lib['id_to_meta'].item(); vocab_dict = {}
    for identifier, item in raw_metadata.items():
        metadata = dict(item)
        metadata.setdefault('uri', metadata.get('concept_uri'))
        metadata.setdefault('source', metadata.get('source_id'))
        metadata.setdefault('QC_label', metadata.get('label_type'))
        metadata.setdefault('definition', metadata.get('definition_en'))
        metadata['source'] = str(metadata.get('source') or '').removeprefix('local_vocabularies:')
        vocab_dict[int(identifier)] = metadata
    model = SentenceTransformer('models_local/all-MiniLM-L6-v2')
    print('fallback index ok', type(index_lib), len(vocab_dict), type(model))
for table_name, df in meta.items():
    candidate_df = df[['name','description']].copy(); candidate_df = candidate_df.fillna('');
    candidate_df['name'] = candidate_df['name'].map(lambda value: z._clean_text(value));
    candidate_df['description'] = candidate_df['description'].map(lambda value: z._clean_text(value));
    candidate_df = candidate_df[candidate_df['name'].astype(bool)]
    print('candidate_df', candidate_df)
    try:
        if matcher_module is not None and hasattr(matcher_module, 'find_nearest_vocab'):
            raw_hits = matcher_module.find_nearest_vocab(candidate_df, index_lib, vocab_dict, k_nearest=10)
            print('module raw hits type', type(raw_hits), 'len', len(raw_hits))
            print('first', raw_hits[:1])
        else:
            raw_hits = []
            print('no find_nearest_vocab on module')
    except Exception as e:
        print('module find_nearest_vocab failed', type(e), e)
        raw_hits = []
    if not raw_hits:
        raw_hits = z._match_keyword_candidates_direct(candidate_df, index_lib, vocab_dict, model, source_filter=None, k_nearest=10)
        print('direct raw hits', raw_hits)
