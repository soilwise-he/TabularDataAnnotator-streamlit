import pandas as pd
import zenodo_autopilot

real_load = zenodo_autopilot._load_module

def wrapped(name, path, session_state_seed=None):
    mod = real_load(name, path, session_state_seed)
    original = mod.find_nearest_vocab
    def traced(df_descriptions, _index_lib, meta_data_dict, k_nearest=4):
        print('TRACE model in module?', 'model' in mod.__dict__, mod.__dict__.get('model'))
        print('TRACE hasattr model', hasattr(mod, 'model'))
        return original(df_descriptions, _index_lib, meta_data_dict, k_nearest=k_nearest)
    mod.find_nearest_vocab = traced
    return mod

zenodo_autopilot._load_module = wrapped
metadata = {'example_table': pd.DataFrame({'name':['Clay content (%)','Annual precipitation (mm)','Country','Crop type'],'description':['Percentage of clay in the soil.', 'Total annual precipitation in millimetres.', 'Country name.', 'Type of crop grown.']})}
print(zenodo_autopilot._generate_keyword_matches_for_metadata(metadata, k_nearest=5))
