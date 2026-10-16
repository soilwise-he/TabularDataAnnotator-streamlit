import json
from pathlib import Path

path = Path(".zenodo_autopilot_exports/14726493/autopilot_summary.json")
data = json.loads(path.read_text(encoding="utf-8"))

table_name = next(iter(data["tables"]))
rows = data["downstream"]["descriptions"][table_name]

for row in rows[:10]:
    name = row["name"]
    desc = row["description"]

    print(name, "->", desc)

    assert isinstance(desc, str)
    assert desc.strip()
    assert desc.lower() != name.lower() or len(desc.strip()) > 3