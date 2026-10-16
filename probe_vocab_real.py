import pandas as pd
import zenodo_autopilot
mod = zenodo_autopilot._load_module('dataannot_keyword_matcher_autopilot', zenodo_autopilot.KEYWORD_MATCHER_FILE)
index_lib, vocab_dict = mod.load_vocab_indexes(modelname='models_local/all-MiniLM-L6-v2')
print('index', index_lib.ntotal, 'vocab', len(vocab_dict))
model = mod.load_sentence_model(modelname='models_local/all-MiniLM-L6-v2')
mod.model = model
sample = pd.DataFrame({'name':['soil moisture','temperature','nitrogen','clay content','country'], 'description':['water content of soil','temperature of soil layer','amount of nitrogen in soil','percentage of clay in soil','country name']})
out = mod.find_nearest_vocab(sample, index_lib, vocab_dict, k_nearest=5)
print(type(out), len(out))
for item in out[:10]:
    print(item.get('query'), item.get('uri'), item.get('source'), item.get('Distance'))
