import pandas as pd
import zenodo_autopilot

orig = zenodo_autopilot._match_keyword_candidates_direct

def wrapped(candidate_df, index_lib, vocab_dict, model, source_filter=None, k_nearest=50):
    print('DIRECT CALLED', candidate_df.head().to_dict())
    out = orig(candidate_df, index_lib, vocab_dict, model, source_filter=source_filter, k_nearest=k_nearest)
    print('DIRECT OUT KEYS', list(out.keys()))
    return out

zenodo_autopilot._match_keyword_candidates_direct = wrapped
metadata = {'example_table': pd.DataFrame({'name':['Clay content (%)','Annual precipitation (mm)','Country','Crop type'],'description':['Percentage of clay in the soil.', 'Total annual precipitation in millimetres.', 'Country name.', 'Type of crop grown.']})}
print(zenodo_autopilot._generate_keyword_matches_for_metadata(metadata, k_nearest=5))
