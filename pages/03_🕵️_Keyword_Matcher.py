import streamlit as st
import pandas as pd
import io
import json
from datetime import datetime
from typing import List, Dict
import os
from PyPDF2 import PdfReader
import docx
import hashlib
from openai import OpenAI
import re
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from sklearn.preprocessing import normalize
from tqdm import tqdm
import requests
import ast
import csv
from ui.blocks import add_Soilwise_contact_sidebar,add_Soilwise_logo,add_clear_cache_button
from pathlib import Path
from typing import Optional, Sequence, Union
from collections import defaultdict
from util.metadata import apply_new_metadata_info


st.set_page_config(page_title="Tabular Soil Data Annotation", layout="wide")
add_Soilwise_logo()
add_clear_cache_button(key_prefix="Keyword_Matcher")

meta_key = f"metadata_df"

# -------------------- Helper data and functions --------------------

DATA_TYPE_OPTIONS = ["string", "numeric", "date"]


#BUG: General bug "Sourcefile" still in sourcelist of the vocab list !!!


def detect_column_type_from_series(s: pd.Series, sample_size: int = 200) -> str:
    # sample values (non-null, up to sample_size)
    series = s.dropna().astype(str).str.strip()
    if len(series) == 0:
        return "string"
    series = series.head(sample_size)

    numeric_count = 0
    date_count = 0
    total = 0

    for val in series:
        if val == "":
            continue
        total += 1
        # allow comma as decimal marker as well
        v = val.replace(',', '.')
        # numeric test (integers or floats)
        try:
            float(v)
            numeric_count += 1
            continue
        except Exception:
            pass
        # date test: try parse
        try:
            # heuristic: require separators or obvious ISO formats
            if any(sep in val for sep in ['/', '-', '.']) or val.isdigit():
                pd.to_datetime(val)
                date_count += 1
        except Exception:
            pass

    if total == 0:
        return "string"
    if numeric_count / total >= 0.8:
        return "numeric"
    if date_count / total >= 0.8:
        return "date"
    return "string"


def build_metadata_df_from_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = []
    for c in df.columns:
        dtype = detect_column_type_from_series(df[c])
        cols.append({
            "name": c,
            "datatype": dtype,
            "element": "",
            "unit": "",
            "method": "",
            "description": "",
            "element_uri": ""
        })
    return pd.DataFrame(cols)

def _read_bytes(source: Union[Path, str, "st.runtime.uploaded_file_manager.UploadedFile"]) -> bytes:
    if hasattr(source, "getvalue"):          # UploadedFile
        return source.getvalue()
    return Path(source).read_bytes()         # path-like

@st.cache_resource()
def read_csv_with_sniffer(uploaded_file) -> pd.DataFrame:
    raw = _read_bytes(uploaded_file)
    #raw = uploaded_file.getvalue()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        import chardet
        enc = chardet.detect(raw)["encoding"] or "cp1252"
        text = raw.decode(enc, errors="replace")

    dialect_uploaded = csv.Sniffer().sniff(text, delimiters=[",", ";", "\t", "|"])
    separator_uploaded = dialect_uploaded.delimiter
    df = pd.read_csv(io.StringIO(text), sep=separator_uploaded)
    return df

@st.cache_resource
def import_metadata_from_file(uploaded_file) -> pd.DataFrame:
    # Accept CSV or JSON (tableschema or csvw)
    name = uploaded_file.name.lower()

    #text = uploaded_file.getvalue().decode("utf-8")
    raw = uploaded_file.getvalue()
    
    try:
        if name.endswith('.csv'):
            df = read_csv_with_sniffer(uploaded_file)
            st.write(df.head())
            # Expect columns: name, element, unit, method, datatype, description
            if "name" not in df.columns:
                st.error("CSV metadata must contain a 'name' column matching column names in the data.")
                return None
            # Normalize
            result = df.rename(columns={c: c.lower() for c in df.columns})
            return result
        elif name.endswith('.json'):
            text = raw.decode("utf-8")
            j = json.loads(text)
            # TableSchema style
            if isinstance(j, dict) and j.get('fields'):
                rows = []
                for f in j['fields']:
                    rows.append({
                        'name': f.get('name'),
                        'datatype': f.get('type') or '',
                        'description': f.get('description') or '',
                        'unit': f.get('unit') or '',
                        'method': f.get('method') or '',
                        'element': f.get('title') or '',
                        'element_uri': f.get('element_uri') or ''
                    })
                return pd.DataFrame(rows)
            # CSVW style
            if isinstance(j, dict) and j.get('tableSchema') and j['tableSchema'].get('columns'):
                rows = []
                for f in j['tableSchema']['columns']:
                    rows.append({
                        'name': f.get('name'),
                        'datatype': f.get('datatype') or '',
                        'description': f.get('null') or '',
                        'unit': f.get('unit') or '',
                        'method': f.get('method') or '',
                        'element': (f.get('titles') or [''])[0],
                        'element_uri': f.get('element_uri') or ''
                    })
                return pd.DataFrame(rows)
            st.error('Unrecognized JSON metadata format (expecting TableSchema or CSVW).')
            return None
        else:
            st.error('Unsupported metadata file type. Upload a CSV or JSON.')
            return None
    except Exception as e:
        st.error(f'Failed to parse metadata file: {e}')
        return None


@st.cache_resource
def read_context_file(contextfile) -> str:
    context_text=""
    if contextfile.type == "application/pdf":
        reader = PdfReader(contextfile)
        for page in reader.pages:
            context_text += page.extract_text() + "\n"
    elif contextfile.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(contextfile)
        for para in doc.paragraphs:
            context_text += para.text + "\n"
    return context_text

def download_bytes(content: bytes, filename: str, mime: str = 'application/octet-stream'):
    st.download_button(label=f"Download {filename}", data=content, file_name=filename, mime=mime)

CACHE_FILE = "data/openai_cache.json"

# Init OpenAI client
def create_openai_client(api_key: str):
    """Create an OpenAI client from an API key (returns None if no key)."""
    if not api_key:
       return None
    return OpenAI(api_key=api_key)


# Init Apertus client
def create_apertus_client(api_key: str):

    if not api_key:
        st.warning("No Apertus API key found in st.secrets['apertus']['api_key']")
        return None

    sess = requests.Session()    
    sess.headers.update({
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "SoilWise_Streamlit_annotator/0.1.0"
    })
    return sess


# Cache Management for optimal LLM usage
def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

def make_hash(context_text: str, variable_names: list,provider:str):
    """Generate a reproducible hash for this exact query."""
    text = provider.strip() + "|" + context_text.strip() + "|" + ",".join(variable_names)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def get_cached_result(context_text, variable_names,provider):
    cache = load_cache()
    key = make_hash(context_text, variable_names,provider)
    return cache.get(key)

def store_result(context_text, variable_names, response_text, provider):
    cache = load_cache()
    key = make_hash(context_text, variable_names,provider)
    old_entry = cache.get(key)

    new_version = (old_entry["version"] + 1) if old_entry else 1
    cache[key] = {
        "context_excerpt": context_text[:500],
        "variables": variable_names,
        "response": response_text,
        "version": new_version,
        "timestamp": datetime.now().isoformat(),
        "provider": provider
    }
    save_cache(cache)

def parse_openai_json(raw_text: str):
    # Remove markdown code fences
    raw_text = re.sub(r"^```(json)?\s*", "", raw_text)
    raw_text = re.sub(r"\s*```$", "", raw_text)

    # Try to find first {...} block
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in the OpenAI response")
    
    json_str = match.group(0)

    # Parse
    return json.loads(json_str)

def get_response_OpenAI(prompt: str) -> str:
    client = st.session_state.get("openai_client")

    if client is None:
        raise RuntimeError("OpenAI client not configured. Add your OpenAI API key in the sidebar.")
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    return response.choices[0].message.content 

def get_response_Apertus(prompt: str) -> str:
    
    client = st.session_state.get("apertus_client")
    if client is None:
        raise RuntimeError("Apertus client not configured. Add your Apertus API key in the sidebar.")
    

    url = "https://api.publicai.co/v1/chat/completions"

    payload = {
        "model": "swiss-ai/apertus-8b-instruct",
        "messages": [{"role": "user","content": prompt}
        ]
    }

    resp = client.post(url, json=payload, timeout=600)
    resp.raise_for_status()
    j = resp.json()
    # try to mirror OpenAI chat response shape
    try:
        return j["choices"][0]["message"]["content"]
    except Exception:
        # fallback: return raw json string for debugging
        return json.dumps(j, indent=2, ensure_ascii=False)

# def get_response_Apertus(prompt: str) -> str:


#     messages_think = [
#         {"role": "user", "content": prompt}
#     ]

#     text = Apertus_tokenizer.apply_chat_template(
#         messages_think,
#         tokenize=False,
#         add_generation_prompt=True,
#     )

#     model_inputs = Apertus_tokenizer([text], return_tensors="pt", add_special_tokens=False).to(Apertus_model.device)

#     # Generate the output
#     generated_ids = Apertus_model.generate(**model_inputs)

#     # Get and decode the output
#     output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]
#     print(Apertus_tokenizer.decode(output_ids, skip_special_tokens=True))

#     return Apertus_tokenizer.decode(output_ids, skip_special_tokens=True)


def generate_descriptions_with_LLM(var_list: List[str], context: str, human_description: Dict) -> Dict[str, str]:
    if not human_description is None:
        # Use human descriptions as hints
        var_list = [f"{var} (hint: {human_description.get(var, '')})" for var in var_list]
    prompt = f"""
    You are given a list of variables from a CSV file and some contextual documentation. For some variables, a hint will be given.
    Context:\n{context[:30000]}
    
    Variables: {var_list}
    
    
    Return a JSON object where each key is a variable name and each value is a concise, factual definition (1–2 sentences) derived from the context. Use neutral, scientific language that describes *what the variable represents* or *how it is measured*, without inferring purpose, function, or evaluation. If the context does not provide enough information, make an educated guess based only on naming conventions and scientific norms, while remaining neutral.
    """

    # Check cache before calling OpenAI
    cached = get_cached_result(context, var_list, provider_selected)
    raw_output = None
    if cached:
        st.info(f"✅ Found cached result (version {cached['version']}, saved {cached['timestamp']})")
        raw_output = cached["response"]
    else:

        with st.spinner(f"🪶 The gnomes are working in colaboration with {provider_selected}..."):
            callfunc = LLM_selection_info.get(provider_selected, {}).get("callfunction")
            raw_output = callfunc(prompt)


        store_result(context, var_list, raw_output,provider_selected)
        st.success("✅ Saved new response to cache")
        
        # st.subheader("Raw OpenAI Output")
        # st.text_area("Raw Response", raw_output, height=300)
        
    if raw_output:
        st.session_state["raw_output"] = raw_output  # Save raw output        
        # --- Try to parse JSON ---
        try:

            variable_descriptions = parse_openai_json(raw_output)
            st.session_state["variable_descriptions"] = variable_descriptions


        except json.JSONDecodeError:
            st.error("Failed to parse JSON. Please check the raw response above.")

    return variable_descriptions

@st.cache_data
def load_sentence_model(modelname="all-MiniLM-L6-v2"):
    with st.spinner(f"🔄 Loading embedding model '{modelname}'... This may take a few seconds."):
        model = SentenceTransformer(modelname)
    return model

@st.cache_data
def load_vocab_indexes(modelname="all-MiniLM-L6-v2"):
    """Load FAISS indexes and associated metadata."""
    with st.spinner(f"🔄 Loading FAISS indexes and vocabulary for '{modelname}'..."):
        
        # Loading in FAISS indexes of preprocessed vocab embeddings
        index_lib = faiss.read_index(f"data/vocabCombined-{modelname.replace('/','--')}.index")

        # Loading library
        meta_data_lib = np.load(
            f"data/vocabCombined-{modelname.replace('/','--')}-meta.npz", allow_pickle=True
        )
        # Retrieving dictionairy with key index ID corresponding FAISS index
        meta_data_dict = meta_data_lib["id_to_meta"].item()
        # """
        #     structure of meta_data_dict:
        #     {
        #         0: {
        #             "uri": "...",
        #             "label": "...",
        #             "definition": "[...]",
        #             "QC_label": "prefLabel" or "altLabel"
        #             "source":"..."
        #         },
        #         1: { ... },
        #         ...
        # """

    return index_lib, meta_data_dict

def embed_vocab(word, definition=None, alpha=0.4):

    """
        Generates a semantic embedding for a given word, optionally influenced by its definition.

        Parameters:
        ----------
        word : str
            The target word to embed.
        definition : str or list, optional
            A textual definition or list of definitions to enrich the word embedding.
            If None or invalid, the word embedding is used alone.
        alpha : float, default=0.8
            Weighting factor between the word and definition embeddings.
            - alpha = 1.0: use only the word embedding
            - alpha = 0.0: use only the definition embedding
            - 0.0 < alpha < 1.0: blend both embeddings

        Returns:
        -------
        np.ndarray
            A normalized vector representing the blended embedding.
    """

    # encode separately
    emb_word = model.encode(word, normalize_embeddings=True)
    
    # definition embedding
    if definition and isinstance(definition, str) and definition.strip() and definition.lower() != 'nan':
        emb_def = model.encode(definition, normalize_embeddings=True)
    elif isinstance(definition, list) and len(definition) > 0:
        emb_def_list = model.encode(definition, normalize_embeddings=True)
        emb_def = emb_def_list.mean(axis=0)
    else:
        emb_def = emb_word
    
    # weighted combination
    vec = alpha * emb_word + (1 - alpha) * emb_def
    vec = normalize(vec.reshape(1, -1)).squeeze()
    return alpha * emb_word + (1 - alpha) * emb_def

@st.cache_resource
def find_nearest_vocab(df_descriptions, _index_lib, meta_data_dict, k_nearest=4)-> Dict:
    """
    Find nearest vocabulary matches for each variable description
    using both prefLabel and altLabel indexes.
    Deduplicate URIs with preference for prefLabel matches.
    Return all unique URIs sorted by similarity.
    """
    matches = []

    for _, row in tqdm(df_descriptions.iterrows(), total=len(df_descriptions), desc="🔍 Matching with vocabulary"):
        query = row["name"]
        definition = row["description"]
        q_emb = embed_vocab(query, definition).astype("float32").reshape(1, -1)

        # Search in combined index
        D_match, I_match = _index_lib.search(q_emb, k=k_nearest)

        # to get a 1D array
        indices = I_match.flatten()
        distances = D_match.flatten()

        # Combine all into a flat list of dicts
        combined = [
            {
                "query": query,
                "definition": definition,
                "id": f"{query}__{int(idx)}",
                **meta_data_dict[int(idx)],  # unpack nested keys
                "Distance": float(dist),
            }
            for idx, dist in zip(indices, distances)
            if idx in meta_data_dict
        ]

        df_comb = pd.DataFrame(combined)

        # Sort prefLabel first, then Distance ascending
        df_comb = df_comb.sort_values(
            by=["QC_label", "Distance"],
            ascending=[True, True],
            key=lambda col: col.map({"prefLabel": 0, "altLabel": 1}) if col.name == "QC_label" else col
        )

        # Deduplicate URIs — keep prefLabel version if present
        df_comb = df_comb.drop_duplicates(subset="uri", keep="first")

        df_comb = df_comb.sort_values(
            by="Distance",
            ascending=True,
        )

        matches.extend(df_comb.to_dict(orient="records"))

    # #  Final global sort on similarity
    # if matches:
    #     df_out = pd.DataFrame(matches).sort_values(by="Similarity", ascending=True)

    return matches


def add_visual_row_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'Group' emoji/marker column based on changes in the 'query' column.
    Works safely in st.data_editor (no styling applied).
    """
    query_change = df["query"].ne(df["query"].shift()).cumsum()
    df = df.copy()
    df["Group Marker"] = query_change.map(lambda x: "🔶" if x % 2 == 0 else "🔷") # other emoji options: ⬛⬜
    # Optionally move marker column to the front
    df.insert(0, " ", df.pop("Group Marker"))
    return df

def make_clickable_links(uris):
    # Convert list of URIs into HTML links separated by commas
    return ", ".join(f'<a href="{u}" target="_blank">{u}</a>' for u in uris)


def sources_selector_ncols(unique_sources, key_prefix="src", n=4):
    """
    Render sources as checkboxes in 3 columns with Select all / Deselect all buttons.
    Returns: list[str] of selected sources.
    """
    unique_sources = list(unique_sources)  # ensure indexable
    unique_sources_sorted = sorted(unique_sources)

    # --- Buttons row
    c1, c2, c3 = st.columns([1, 1, 3])
    with c1:
        select_all = st.button("Select all", width='stretch',icon="✅")
    with c2:
        deselect_all = st.button("Deselect all", width='stretch',icon="🟩")

    # Apply button actions by updating session_state for all checkbox keys
    if select_all:
        for s in unique_sources_sorted:
            st.session_state[f"{key_prefix}_{s}"] = True

    if deselect_all:
        for s in unique_sources_sorted:
            st.session_state[f"{key_prefix}_{s}"] = False

    st.write("Select sources:")

    # --- 3 column checkbox grid
    cols = st.columns(n)
    selected = []

    for i, s in enumerate(unique_sources_sorted):
        col = cols[i % n]
        with col:
            # Initialize default only once (if key not present)
            k = f"{key_prefix}_{s}"
            if k not in st.session_state:
                st.session_state[k] = True  # default checked

            checked = st.checkbox(s, key=k)
            if checked:
                selected.append(s)

    return selected

def is_allowed_source(meta_id, meta_data_dict, selected_sources):
    """
    Returns True if the metadata item belongs to a selected source.
    - If selected_sources is empty → allow nothing
    - If source is missing → exclude
    """
    if not selected_sources:
        return False

    meta = meta_data_dict.get(meta_id)
    if not meta:
        return False

    return meta.get("source") in selected_sources

def _default_for_value(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return None
    if isinstance(v, str):
        return ""
    if isinstance(v, list):
        return []
    if isinstance(v, dict):
        return {}
    return None

def make_blank_hit_smart(query, hits, force_keys=("query", "source")):
    sample = {}
    keys = set()

    for h in hits:
        if not isinstance(h, dict):
            continue
        keys.update(h.keys())
        for k, v in h.items():
            if k not in sample and v is not None:
                sample[k] = v

    keys.update(force_keys)

    blank = {k: _default_for_value(sample.get(k)) for k in keys}
    blank["query"] = query
    return blank

@st.cache_resource()
def get_UoM_Ansis_guess(dict_vocab:Dict, dicts_selection:Dict)-> Dict:
    results_dict = defaultdict(dict)
    for key, k_dict in dict_vocab.items():

        dict_selection = dicts_selection.get(key, {})

        results_dict[key]= defaultdict(dict)
        filtered_results_ansis = [
            hit for hit in k_dict
            if hit.get("source") in ["Ansis"]
        ]
        for hit in k_dict:
            results_dict[key][hit.get('query')]=defaultdict(dict)

        for hit in filtered_results_ansis:
            selected = dict_selection[hit.get('id')]
            hit['Selected'] = selected
            uri = hit.get("uri")
            hit['applicableUnit'] = applicable_unit_map.get(uri,'')
            hit['quantityKind'] = quantity_kind_map.get(uri,'')
            results_dict[key][hit.get('query')].setdefault("Potential_UoM", []).append((hit['applicableUnit'],hit['quantityKind']))
            if selected:
                results_dict[key][hit.get('query')]['selected_UoM'] = (hit['applicableUnit'],hit['quantityKind'])
            
        return results_dict
# PIN -------------------- UI --------------------
st.title("🕵️ STEP 3 : Keyword Matching")

st.markdown("""
            For .
            Later in the app, there is an option to augment the information, making use of an LLM. Also in that case, only the variable names and the context text (if you choose to upload some) will be used.
            """)

st.markdown(""" --- """)

if meta_key not in st.session_state or st.session_state.get(meta_key) is None:
    st.info("No metadata found. Please upload a data file with recognizable columns or a metadata file in step 1 to proceed with keyword matching.")
    
else:
    meta_dict= st.session_state[meta_key]
    st.markdown("### Element matcher")

    modelname = "models_local/all-MiniLM-L6-v2"
    model = load_sentence_model(modelname = modelname)

    
    index_vocabs, dict_vocabs = load_vocab_indexes(modelname = modelname)
    unique_sources = {v.get("source") for v in dict_vocabs.values() if v.get("source")}

    vocab_list = [v["label"] for v in dict_vocabs.values() if "label" in v]

    selected_sources = sources_selector_ncols(unique_sources)


    if "vocab_oversized_matching_results" not in st.session_state:
        st.session_state["vocab_oversized_matching_results"] = {}


    if "vocab_row_selection" not in st.session_state:
        st.session_state["vocab_row_selection"] = {}

    if "vocab_row_selection_status" not in st.session_state:
        st.session_state["vocab_row_selection_status"] = {}

    if "df_selection_keywords" not in st.session_state:
        st.session_state["df_selection_keywords"] = {}

    if st.button("Find Vocabulary Terms In Thesaury"):


        for key, meta_df in meta_dict.items():

            #TODO: investigate how to filter voc provider. READ; https://github.com/Undertone0809/langchain/blob/patch-duckduckgo/docs/modules/indexes/vectorstores/examples/faiss.ipynb
            with st.spinner(f"🔎 Finding nearest vocabulary terms - {key}..."):
                raw_result_listdict = find_nearest_vocab(
                    meta_df,                         # dataframe with searchterm. necesarry columns; ["Variable", "Description"]
                    _index_lib=index_vocabs,          # FAISS index
                    meta_data_dict=dict_vocabs,      # mapping dict
                    k_nearest=50
                )
            
            st.session_state["vocab_oversized_matching_results"][key] = raw_result_listdict
            st.session_state["vocab_row_selection"][key] = {}
            for result in raw_result_listdict:
                key_result = result.get('id')
                st.session_state["vocab_row_selection"][key][key_result]=False
            st.session_state["vocab_row_selection_status"][key]="initialised"

        st.success("✅ Matching complete!")

    if "vocab_matching_results" not in st.session_state:
        st.session_state["vocab_matching_results"] = {}


    if st.session_state["vocab_oversized_matching_results"]:

        if not selected_sources:
            st.warning("⚠️ Please select at least one vocabulary source.")
            st.stop()

        for key, k_dict in st.session_state["vocab_oversized_matching_results"].items():
            filtered_results = [
                hit for hit in k_dict
                if hit.get("source") in selected_sources
            ]

            grouped = defaultdict(list)
            for hit in filtered_results:
                grouped[hit["query"]].append(hit)

            # keep filtered-out items and return blanks
            original_headers = list(dict.fromkeys(hit.get("query") for hit in k_dict))
            
            grouped_incl_blank = {
                q: grouped.get(q, [make_blank_hit_smart(q, k_dict)])
                for q in original_headers
            }


            final_results = []
            keys_closest_filtered = []
            for query, hits in grouped_incl_blank.items():
                
                hits_sorted = sorted(
                    hits,
                    key=lambda x: x["Distance"],
                    reverse=False
                )
                keys_closest_filtered.append(hits_sorted[0]["id"])
                final_results.extend(hits_sorted[:4])


            st.session_state["vocab_matching_results"][key] = final_results

            if not st.session_state["vocab_row_selection_status"][key] =="initialised":
                continue
            for key_closest_filtered in keys_closest_filtered:
                st.session_state["vocab_row_selection"][key][key_closest_filtered] = True
            st.session_state["vocab_row_selection_status"][key] = "first proces done"
            


    if st.session_state["vocab_matching_results"]:

        tab_labels_matching = list(st.session_state["vocab_matching_results"].keys())
        tabs_matching = st.tabs(tab_labels_matching)

        st.divider()
        for tab, key in zip(tabs_matching, tab_labels_matching):
            with tab:
                vocab_match_dict = st.session_state["vocab_matching_results"][key]
                result_df = pd.DataFrame(vocab_match_dict)

                ## Add a checkbox column if it doesn't exist
                # if "Selected" not in result_df.columns:
                #     result_df.insert(1, "Selected", False) 
                #     # Set "Selected" to True for the row with the lowest Distance in each group of "query"
                #     idx_min_distance = result_df.groupby("query")["Distance"].idxmin()
                #     result_df.loc[idx_min_distance, "Selected"] = True
                
                result_df["Selected"] = result_df["id"].map(
                    lambda rid: st.session_state["vocab_row_selection"][key].get(rid, False)
                )

                #TODO: process only once?
                styled_df = add_visual_row_group(result_df)

                #TODO: change headers? Also include " original header description"?
                column_list = styled_df.columns.tolist()
                disabling_columns= column_list.copy()
                disabling_columns.remove("Selected")
                priority_cols = [" ", "Selected", "query", "label", "definition"]
                new_order = [c for c in priority_cols if c in column_list] + [c for c in column_list if c not in priority_cols]
                new_order.remove("id")

                #BUG: every edit everything jumps. Probably because of rerun and new styled_df?
                edited_df = st.data_editor(
                    styled_df,
                    disabled =disabling_columns,
                    column_order = new_order,
                    num_rows="dynamic",
                    width='stretch',
                    key=f"editable_vocab_df_{key}",
                    hide_index=True
                )

                if not edited_df.equals(styled_df):
                    # update the selected items
                    for _, row in edited_df.iterrows():
                        st.session_state["vocab_row_selection"][key][row["id"]] = row["Selected"]
                    st.rerun()

                # Process the selected rows
                selected_rows = edited_df[edited_df["Selected"]].copy()

                st.session_state["df_selection_keywords"][key] = selected_rows

                # Drop duplicates while preserving original order of 'query'
                unique_queries = edited_df["query"].drop_duplicates()

                if not selected_rows.empty:
                    st.markdown("##### -> Selected Rows")
                    
                    summary_df = (
                        selected_rows
                        .groupby("query", sort=False, as_index=False)
                        .agg({
                            "label": lambda x: list(set(x)),  # use set() to keep unique labels
                            "uri": lambda x: list(set(x))     # use set() to keep unique URIs
                        })
                    )

                    orig_queries = list(unique_queries)
                    missing = [q for q in orig_queries if q not in summary_df["query"].values]
                    if missing:
                        df_missing = pd.DataFrame({
                            "query": missing,
                            "label": [ [] for _ in missing ],
                            "uri":   [ [] for _ in missing ]
                        })
                        summary_df = pd.concat([summary_df, df_missing], ignore_index=True)

                    summary_df["element_uri"] = summary_df.apply(
                        lambda row: {label: uri for label, uri in zip(row["label"], row["uri"])},
                        axis=1
                    )
                    # Reorder rows according to the original query order
                    summary_df["query"] = pd.Categorical(summary_df["query"], categories=unique_queries, ordered=True)
                    summary_df = summary_df.sort_values("query").reset_index(drop=True)
                    summary_df["label"] = summary_df["label"].apply(lambda x: ", ".join(x))
                    summary_df["uri"] = summary_df["uri"].apply(lambda x: " || ".join(x))
                    st.dataframe(summary_df,
                                    column_config={
                                        "uri": st.column_config.LinkColumn()
                                    })
                    
                    summary_df = summary_df.rename(columns={"query": "name","label":"element"})

                    st.session_state['metadata_df'] = apply_new_metadata_info({key : summary_df}, st.session_state['metadata_df'], overwrite='yes_incl_blanks')

                else:
                    st.info("No rows selected.")


    # Prep UoM section with Ansis links
    if st.session_state["vocab_oversized_matching_results"]:
        
        vocab_csv = r"data\partial_results.csv"
        df_vocabs = read_csv_with_sniffer(Path(vocab_csv))
        applicable_unit_map = df_vocabs.set_index("concept")["applicableUnit"].to_dict()
        quantity_kind_map   = df_vocabs.set_index("concept")["quantityKind"].to_dict()

        for key, meta_df in meta_dict.items():
            if "ansis_UoM" not in st.session_state:
                st.session_state["ansis_UoM"] = defaultdict(dict)

            dict_selection = st.session_state["vocab_row_selection"]
            st.session_state["ansis_UoM"] = get_UoM_Ansis_guess(st.session_state["vocab_oversized_matching_results"], dict_selection)




    st.divider()

    st.markdown("#### Current state of your metadata table:")

    # Create one tab per key
    tab_labels = list(st.session_state[meta_key].keys())
    tabs = st.tabs(tab_labels)
    error_handling_tabs = st.empty()
    error_tabs = []

    total_size = 0
    descriptions_completed = 0

    for tab, key in zip(tabs, tab_labels):
        df = st.session_state[meta_key][key]
        
        total_size=+len(df)
        count_completed = df["description"].notna() & (df["description"] != "")
        descriptions_completed=+ count_completed.sum()
        with tab:
            st.dataframe(st.session_state[meta_key][key])


# -------------------- reach us --------------------
add_Soilwise_contact_sidebar()