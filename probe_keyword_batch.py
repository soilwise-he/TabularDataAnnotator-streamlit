import pandas as pd
import zenodo_autopilot

metadata = {
    'example_table': pd.DataFrame({
        'name': ['Clay content (%)','Annual precipitation (mm)','Country','Crop type'],
        'description': ['Percentage of clay in the soil.', 'Total annual precipitation in millimetres.', 'Country name.', 'Type of crop grown.']
    })
}
print(zenodo_autopilot._generate_keyword_matches_for_metadata(metadata, k_nearest=5))
