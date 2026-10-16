import pandas as pd
from zenodo_autopilot import _generate_keyword_matches_for_metadata
meta = {'test': pd.DataFrame({'name':['sample_date','soil moisture','nitrogen','ph','temperature'],'description':['date when sample was taken','water content in soil','amount of nitrogen in soil','soil reaction value','temperature of soil layer']})}
out = _generate_keyword_matches_for_metadata(meta, modelname='models_local/all-MiniLM-L6-v2', k_nearest=10)
print(type(out))
print(list(out.keys()))
print({k: len(v) for k, v in out.items()})
for k, v in out.items():
    print(k, v[:2])
