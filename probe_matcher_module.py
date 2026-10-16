import zenodo_autopilot
from sentence_transformers import SentenceTransformer
m = zenodo_autopilot._load_module('debug_keyword_matcher', zenodo_autopilot.KEYWORD_MATCHER_FILE)
print('before_has_model', hasattr(m, 'model'), getattr(m, 'model', None))
model = SentenceTransformer('models_local/all-MiniLM-L6-v2')
m.model = model
m.__dict__['model'] = model
print('after_has_model', hasattr(m, 'model'), m.__dict__.get('model'))
print('embed_vocab_test', m.embed_vocab('clay content', 'soil clay content').shape)
