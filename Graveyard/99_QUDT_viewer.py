import os
import re
import json
import requests
import streamlit as st
from pathlib import Path
from rdflib import Graph, URIRef, Namespace
from rdflib.namespace import RDF, RDFS, SKOS

st.set_page_config(page_title="QUDT Hierarchy Browser", layout="wide")

QUDT = Namespace("http://qudt.org/schema/qudt/")
QK   = Namespace("http://qudt.org/vocab/quantitykind/")
UNIT = Namespace("http://qudt.org/vocab/unit/")
PREFIX = Namespace("http://qudt.org/vocab/prefix/")
SOU = Namespace("http://qudt.org/vocab/sou/")
SOQK = Namespace("http://qudt.org/vocab/soqk/")

DEFAULT_URL = "https://qudt.org/qudt-all"  # user-provided; QUDT also documents http://qudt.org/qudt-all [2](https://github.com/qudt/qudt-public-repo)

CACHE_DIR = Path(".cache_qudt")
CACHE_DIR.mkdir(exist_ok=True)

# ----------------------------
# Helpers: download + cache
# ----------------------------
def safe_filename(url: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", url)

@st.cache_data(show_spinner=False)
def download_qudt(url: str) -> Path:
    """
    Download QUDT qudt-all as Turtle using content negotiation.
    Cached by Streamlit; also persisted on disk for repeat runs.
    """
    fn = CACHE_DIR / f"{safe_filename(url)}.ttl"
    if fn.exists() and fn.stat().st_size > 0:
        return fn

    headers = {
        "Accept": "text/turtle, application/x-turtle;q=0.9, application/rdf+xml;q=0.8, */*;q=0.1",
        "User-Agent": "streamlit-qudt-browser/1.0"
    }
    r = requests.get(url, headers=headers, timeout=120)
    r.raise_for_status()
    fn.write_bytes(r.content)
    return fn

@st.cache_resource(show_spinner=False)
def load_graph(ttl_path: Path) -> Graph:
    g = Graph()
    # Turtle parsing; can be heavy for qudt-all
    g.parse(ttl_path.as_posix(), format="turtle")
    return g

# ----------------------------
# Helpers: labels, details
# ----------------------------
LABEL_PROPS = [SKOS.prefLabel, RDFS.label, SKOS.altLabel]

def get_best_label(g: Graph, s: URIRef) -> str:
    # prefer skos:prefLabel then rdfs:label
    for p in [SKOS.prefLabel, RDFS.label]:
        for o in g.objects(s, p):
            return str(o)
    # fallback: last part of URI
    return str(s).split("/")[-1]

def get_literals(g: Graph, s: URIRef, p: URIRef, max_n=5):
    vals = [str(o) for o in g.objects(s, p)]
    return vals[:max_n]

def iri_startswith_any(s: str, prefixes: list[str]) -> bool:
    return any(s.startswith(px) for px in prefixes)

# ----------------------------
# Hierarchy builders
# ----------------------------
def build_hierarchy(g: Graph, concept_filter_prefixes: list[str]):
    """
    Build hierarchy using SKOS broader/narrower, fallback to rdfs:subClassOf.
    Returns:
      nodes: set(URIRef)
      children: dict[parent -> list[child]]
      parents: dict[child -> list[parent]]
      roots: list[URIRef]
      rel_used: 'skos' or 'rdfs'
    """
    nodes = set()
    broader_edges = []

    # 1) Try SKOS broader
    for child, parent in g.subject_objects(SKOS.broader):
        cs, ps = str(child), str(parent)
        if iri_startswith_any(cs, concept_filter_prefixes) and iri_startswith_any(ps, concept_filter_prefixes):
            nodes.add(child); nodes.add(parent)
            broader_edges.append((parent, child))

    rel_used = "skos" if broader_edges else "rdfs"

    # 2) Fallback: rdfs:subClassOf
    if not broader_edges:
        for child, parent in g.subject_objects(RDFS.subClassOf):
            cs, ps = str(child), str(parent)
            if iri_startswith_any(cs, concept_filter_prefixes) and iri_startswith_any(ps, concept_filter_prefixes):
                nodes.add(child); nodes.add(parent)
                broader_edges.append((parent, child))

    children = {n: [] for n in nodes}
    parents = {n: [] for n in nodes}
    for parent, child in broader_edges:
        children.setdefault(parent, []).append(child)
        parents.setdefault(child, []).append(parent)

    # sort children by label later in UI (label needs graph)
    roots = [n for n in nodes if len(parents.get(n, [])) == 0]
    return nodes, children, parents, roots, rel_used

def search_nodes(g: Graph, nodes: set, q: str, limit=50):
    q = (q or "").strip().lower()
    if not q:
        return []
    hits = []
    for s in nodes:
        iri = str(s).lower()
        if q in iri:
            hits.append(s); 
            if len(hits) >= limit: break
            continue
        # labels
        lbl = get_best_label(g, s).lower()
        if q in lbl:
            hits.append(s)
            if len(hits) >= limit: break
    return hits

# ----------------------------
# UI
# ----------------------------
st.title("QUDT hiërarchische browser (Streamlit)")

with st.sidebar:
    st.header("Data bron")
    url = st.text_input("QUDT qudt-all URL", value=DEFAULT_URL,
                        help="QUDT documenteert ook de graph-URI http://qudt.org/qudt-all. Een versie speciefieke release is ook mogelijk [2](https://github.com/qudt/qudt-public-repo)")

    st.header("Concepttype / scope")
    scope = st.selectbox(
        "Kies wat je wil browsen",
        [
            ("Quantity Kinds", [str(QK)]),
            ("Units", [str(UNIT)]),
            ("Prefixes", [str(PREFIX)]),
            ("Systems of Units (sou)", [str(SOU)]),
            ("Systems of Quantity Kinds (soqk)", [str(SOQK)]),
            ("Alles onder qudt.org/vocab/", ["http://qudt.org/vocab/"]),
        ],
        format_func=lambda x: x[0]
    )

    max_children = st.slider("Max children per niveau (UI)", 10, 500, 100, 10)
    max_depth = st.slider("Max diepte (UI)", 1, 20, 8, 1)

# Load data
with st.spinner("Downloaden & laden van QUDT (kan groot zijn)…"):
    ttl_path = download_qudt(url)
    g = load_graph(ttl_path)

st.success(f"Geladen: {ttl_path.name} ({ttl_path.stat().st_size/1e6:.1f} MB)")

prefixes = scope[1]
nodes, children, parents, roots, rel_used = build_hierarchy(g, prefixes)

colA, colB = st.columns([1, 1], gap="large")

with colA:
    st.subheader("Navigatie")
    st.caption(f"Relatie gebruikt voor hiërarchie: **{rel_used}** (SKOS `broader` of fallback `subClassOf`).")

    # Search
    q = st.text_input("Zoek (label of URI)", value="")
    hits = search_nodes(g, nodes, q, limit=50) if q else []
    if q:
        st.write(f"Zoekresultaten: {len(hits)} (max 50)")
        if hits:
            hit_labels = [f"{get_best_label(g, h)} — {h}" for h in hits]
            sel_hit = st.selectbox("Kies resultaat", ["(geen)"] + hit_labels)
            if sel_hit != "(geen)":
                idx = hit_labels.index(sel_hit)
                st.session_state["current"] = hits[idx]

    # Root selection
    if "current" not in st.session_state:
        st.session_state["current"] = None

    # Sort roots by label
    roots_sorted = sorted(roots, key=lambda x: get_best_label(g, x).lower())
    root_labels = [f"{get_best_label(g, r)} — {r}" for r in roots_sorted]

    sel_root = st.selectbox("Start bij root", ["(kies)"] + root_labels)
    if sel_root != "(kies)":
        st.session_state["current"] = roots_sorted[root_labels.index(sel_root)]

    current = st.session_state["current"]

    # Breadcrumb/path navigation
    if current:
        st.markdown("### Pad (breadcrumb)")
        path = [current]
        # walk upwards by first parent (if multiple parents exist)
        seen = set()
        while True:
            p_list = parents.get(path[0], [])
            if not p_list:
                break
            p0 = p_list[0]
            if p0 in seen:
                break
            seen.add(p0)
            path.insert(0, p0)

        st.write(" → ".join([get_best_label(g, p) for p in path]))

        st.markdown("### Kinderen")
        kids = children.get(current, [])
        kids_sorted = sorted(kids, key=lambda x: get_best_label(g, x).lower())[:max_children]

        if not kids_sorted:
            st.info("Geen kinderen gevonden op basis van de gekozen hiërarchierelatie.")
        else:
            kid_labels = [f"{get_best_label(g, k)} — {k}" for k in kids_sorted]
            pick = st.selectbox("Ga naar kind", ["(kies)"] + kid_labels)
            if pick != "(kies)":
                st.session_state["current"] = kids_sorted[kid_labels.index(pick)]
                st.rerun()

        # Optional: show subtree with expanders (limited depth)
        with st.expander("Toon subtree (expanders)", expanded=False):
            def render_subtree(node, depth=0):
                if depth >= max_depth:
                    st.write("… (max diepte bereikt)")
                    return
                kids = children.get(node, [])
                kids = sorted(kids, key=lambda x: get_best_label(g, x).lower())[:max_children]
                for k in kids:
                    label = get_best_label(g, k)
                    with st.expander(("  " * depth) + label, expanded=False):
                        if st.button(f"Open {label}", key=f"open_{k}"):
                            st.session_state["current"] = k
                            st.rerun()
                        render_subtree(k, depth+1)

            render_subtree(current, depth=0)

with colB:
    st.subheader("Details")
    current = st.session_state.get("current")
    if not current:
        st.info("Kies een root of zoek een concept om details te zien.")
    else:
        st.code(str(current), language="text")

        # Labels
        st.markdown("**Labels**")
        labels = []
        for p in LABEL_PROPS:
            for o in g.objects(current, p):
                labels.append((p.n3(g.namespace_manager), str(o)))
        if labels:
            for p, v in labels[:20]:
                st.write(f"- {p}: {v}")
        else:
            st.write("- (geen labels gevonden)")

        # Definitions / comments
        st.markdown("**Definitie / commentaar**")
        defs = get_literals(g, current, SKOS.definition) + get_literals(g, current, RDFS.comment)
        if defs:
            for d in defs[:10]:
                st.write(f"- {d}")
        else:
            st.write("- (geen definitie/commentaar gevonden)")

        # QUDT-specific useful fields if present
        st.markdown("**QUDT velden (indien aanwezig)**")
        fields = [
            (QUDT.symbol, "qudt:symbol"),
            (QUDT.ucumCode, "qudt:ucumCode"),
            (QUDT.conversionMultiplier, "qudt:conversionMultiplier"),
            (QUDT.conversionOffset, "qudt:conversionOffset"),
            (QUDT.applicableUnit, "qudt:applicableUnit"),
            (QUDT.hasQuantityKind, "qudt:hasQuantityKind"),
            (QUDT.quantityKind, "qudt:quantityKind"),
        ]
        any_field = False
        for p, name in fields:
            vals = [str(o) for o in g.objects(current, p)]
            if vals:
                any_field = True
                st.write(f"- **{name}**:")
                for v in vals[:30]:
                    st.write(f"  - {v}")
        if not any_field:
            st.write("- (geen bekende QUDT velden gevonden voor dit concept)")

        # Parents / children quick view
        st.markdown("**Relaties**")
        ps = parents.get(current, [])
        cs = children.get(current, [])
        st.write(f"- Parents: {len(ps)}")
        for p in ps[:20]:
            st.write(f"  - {get_best_label(g, p)} — {p}")
        st.write(f"- Children: {len(cs)}")
        for c in sorted(cs, key=lambda x: get_best_label(g, x).lower())[:20]:
            st.write(f"  - {get_best_label(g, c)} — {c}")