import pandas as pd
from rdflib import Graph

TTL_FILE = "all.ttl"
SPARQL_FILE = "query.sparql"
CSV_FILE = "all.csv"

g = Graph()
g.parse(TTL_FILE, format="turtle")

with open(SPARQL_FILE, "r") as f:
    query = f.read()

results = g.query(query)

columns = [str(var) for var in results.vars]
rows = [
    [str(val) if val is not None else "" for val in row]
    for row in results
]

df = pd.DataFrame(rows, columns=columns)
df.to_csv(CSV_FILE, index=False)
print(df.head(5))
print(df.keys())
print(f"Geschreven: {len(df)} rijen naar {CSV_FILE}")
