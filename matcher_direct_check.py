import pandas as pd
import zenodo_autopilot as z
meta = {'test': pd.DataFrame({'name':['sample_date','soil moisture','nitrogen','ph','temperature'],'description':['date when sample was taken','water content in soil','amount of nitrogen in soil','soil reaction value','temperature of soil layer']})}
matcher_module = z._load_module('dataannot_keyword_matcher_autopilot', z.KEYWORD_MATCHER_FILE)
index_lib, vocab_dict = matcher_module.load_vocab_indexes(modelname='models_local/all-MiniLM-L6-v2')
matcher_module.model = matcher_module.load_sentence_model(modelname='models_local/all-MiniLM-L6-v2')
for table_name, df in meta.items():
    candidate_df = df[['name','description']].copy().fillna('')
    candidate_df['name'] = candidate_df['name'].map(lambda value: z._clean_text(value))
    candidate_df['description'] = candidate_df['description'].map(lambda value: z._clean_text(value))
    candidate_df = candidate_df[candidate_df['name'].astype(bool)]
    hits = matcher_module.find_nearest_vocab(candidate_df, index_lib, vocab_dict, k_nearest=10)
    print(type(hits), len(hits))
    print(hits[:2])
