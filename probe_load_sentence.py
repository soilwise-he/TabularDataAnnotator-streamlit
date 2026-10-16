import pandas as pd
import zenodo_autopilot

real_load = zenodo_autopilot._load_module

def wrapped(name, path, session_state_seed=None):
    mod = real_load(name, path, session_state_seed)
    orig = mod.load_sentence_model
    def traced(modelname='all-MiniLM-L6-v2'):
        out = orig(modelname)
        mod.model = out
        mod.__dict__['model'] = out
        print('TRACE load_sentence_model called, model set?', 'model' in mod.__dict__, mod.__dict__.get('model'))
        return out
    mod.load_sentence_model = traced
    return mod

zenodo_autopilot._load_module = wrapped
metadata = {'example_table': pd.DataFrame({'name':['Clay content (%)','Annual precipitation (mm)','Country','Crop type'],'description':['Percentage of clay in the soil.', 'Total annual precipitation in millimetres.', 'Country name.', 'Type of crop grown.']})}
print(zenodo_autopilot._generate_keyword_matches_for_metadata(metadata, k_nearest=5))
