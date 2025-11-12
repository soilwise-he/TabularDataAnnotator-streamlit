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

st.logo(
    'Logo/SWlogocolor.svg',
    link="https://soilwise-he.eu/",
)

# Clear Cache Button aligned to the right
col1, col2 = st.columns([5, 1])  # Adjust column proportions as needed

with col2:
    if st.button("🔄 Clear Cache and Restart"):
        # Clear session state
        for key in st.session_state.keys():
            del st.session_state[key]
        # Rerun the app
        st.rerun()

st.set_page_config(page_title="Tabular Soil Data Annotation", layout="wide")

# -------------------- Helper data and functions --------------------

DATA_TYPE_OPTIONS = ["string", "numeric", "date"]


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

@st.cache_resource
def import_metadata_from_file(uploaded_file) -> pd.DataFrame:
    # Accept CSV or JSON (tableschema or csvw)
    name = uploaded_file.name.lower()
    text = uploaded_file.getvalue().decode("utf-8")
    try:
        if name.endswith('.csv'):
            df = pd.read_csv(io.StringIO(text))
            # Expect columns: name, element, unit, method, datatype, description
            if "name" not in df.columns:
                st.error("CSV metadata must contain a 'name' column matching column names in the data.")
                return None
            # Normalize
            result = df.rename(columns={c: c.lower() for c in df.columns})
            return result
        elif name.endswith('.json'):
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


def apply_new_metadata_info(metadata_df: pd.DataFrame, current_meta: pd.DataFrame, overwrite = False) -> pd.DataFrame:
    # Function to apply new metadata info from metadata_df to current_meta

    # metadata_df expected to contain 'name' column
    if metadata_df is None:
        return current_meta
    md = current_meta.copy()
    for _, row in metadata_df.iterrows():
        name = row.get('name')
        if name in md['name'].values:
            idx = md.index[md['name'] == name][0]
            for col in ['datatype', 'element', 'unit', 'method', 'description','element_uri']:
                current_value = md.at[idx, col]
                
                if not overwrite and pd.notna(current_value) and current_value not in [None, ""]:
                    continue
                if col in row and pd.notna(row[col]) and row[col] != "":
                    value = row[col]
                    if isinstance(value, dict):
                        md.loc[idx, col] = str(value)  # Convert dict to string
                    else:
                        md.loc[idx, col] = value
    return md


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

@st.cache_resource
def load_sentence_model(modelname="all-MiniLM-L6-v2"):
    with st.spinner(f"🔄 Loading embedding model '{modelname}'... This may take a few seconds."):
        model = SentenceTransformer(modelname)
    return model

@st.cache_resource
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

def find_nearest_vocab(df_descriptions, index_lib, meta_data_dict, k_nearest=4):
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
        D_match, I_match = index_lib.search(q_emb, k=k_nearest)

        # to get a 1D array
        indices = I_match.flatten()
        distances = D_match.flatten()

        # Combine all into a flat list of dicts
        combined = [
            {
                "query": query,
                "definition": definition,
                "id": int(idx),
                **meta_data_dict[int(idx)],  # unpack nested keys
                "Distance": float(dist)
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


# -------------------- UI --------------------
st.title("Tabular Soil Data Annotation")

with st.expander("How to use", expanded=False):
    st.markdown(
        """
### Getting Started
1. Choose your **input mode** of your datafile:
   - **Single**: Upload a single CSV file
   - **Linked**: Upload two related CSV files (sites and observations)
   - **Excel**: Upload an Excel workbook, wherein you can select a specific sheets with tabular data

### Data Import & Preview
2. Upload your data file in the **Import tabular data** section
3. Optionally import existing metadata (CSV or JSON TableSchema/CSVW) in the **Import existing metadata** section
4. Review your data in the preview section showing the first 5 rows

### Adding Descriptions
5. Use the **Preview Description** section to:
   - Manually edit descriptions for each column
   - Use AI assistance by:
     - Uploading context documents (PDF, DOC, DOCX)
     - Writing custom context
     - Generating AI-powered descriptions

### Vocabulary Matching
6. Use the **Element matcher** to:
   - Find matching vocabulary terms from established thesauri
   - Select appropriate matches for your columnheaders
   - Automatically update element names and URIs

### Metadata Management
7. Edit detailed column metadata in the **Column metadata** table:
   - Data type (auto-detected but adjustable)
   - Element name
   - Unit
   - Method
   - Description

### Export Options
8. Export your annotated metadata in various formats:
   - CSV format
   - TableSchema JSON
   - CSVW JSON
"""
    )


mode = st.radio("Input mode", options=["single", "linked", "excel"], index=0, horizontal=True)

## Different methods to convey the information
# st.caption("ℹ️ No data will be sent to any third party. All information stays on your device.")
# st.info('This is a purely informational message', icon="ℹ️")
container_2 = st.empty()
button_A = container_2.button('ℹ️')
if button_A:
    container_2.empty()
    button_B = container_2.button("ℹ️ No data will be sent to any third party. All uploaded information stays on your device.")

# Placeholders for data
uploaded_df = None
site_df = None
obs_df = None

with st.sidebar.expander("LLM Provider", expanded=True):
    st.write("Choose which LLM backend to use and check API key availability.")
    

    LLM_selection_info = {
        "OpenAI": {
            'key' : st.secrets.get("openai", {}).get("api_key"),
            'callfunction': get_response_OpenAI,
            'help_key_creation': 'https://platform.openai.com/account/api-keys',
        },
        "Apertus": {
            'key' : st.secrets.get("apertus", {}).get("api_key"),
            'callfunction': get_response_Apertus,
            'help_key_creation': 'https://platform.publicai.co/settings/api-keys',
        },
        "Test_Wrong": {
            'key' : st.secrets.get("test", {}).get("api_key"),
            'callfunction': None,
            'help_key_creation': 'https://example.com',
        },
    }

    provider_selected = st.radio("Provider", options=LLM_selection_info.keys(), index=0, key="llm_provider")

    if LLM_selection_info[provider_selected]['key']:
        key_in_use = LLM_selection_info[provider_selected]['key']
        st.success(f"✅ Found API key for {provider_selected} .")        
    else:
        placeholder_error = st.empty()
        manual_key = st.text_input(f"Enter {provider_selected} API Key", type="password", key="manual_api_key")
        if not manual_key:
            placeholder_error.error(f"❌ No API key for {provider_selected} found in st.secrets. Generete one here; {LLM_selection_info[provider_selected]['help_key_creation']}. And input manually if needed.")
        key_in_use = manual_key or None
        LLM_selection_info[provider_selected]['key']=key_in_use
    st.session_state["api_key"] = key_in_use

    # create provider-specific clients (store in session_state)
    if provider_selected == "OpenAI":
        st.session_state["openai_client"] = create_openai_client(key_in_use)
    elif provider_selected == "Apertus":
        st.session_state["apertus_client"] = create_apertus_client(key_in_use)
    else:
        # ensure no stale clients remain
        st.session_state.pop("openai_client", None)
        st.session_state.pop("apertus_client", None)


# File upload UI
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("**Import tabular data**")
    if mode == 'single':
        uploaded = st.file_uploader("Upload CSV file", type=['csv'], key='single_upload')
        filename = uploaded.name if uploaded else None
        if uploaded:
            try:
                uploaded_df = pd.read_csv(uploaded)
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")

    elif mode == 'linked':
        site_file = st.file_uploader("Upload Sites CSV (locations)", type=['csv'], key='site_upload')
        obs_file = st.file_uploader("Upload Observations CSV", type=['csv'], key='obs_upload')
        filename = obs_file.name if obs_file else None
        if site_file:
            try:
                site_df = pd.read_csv(site_file)
            except Exception as e:
                st.error(f"Failed to read sites CSV: {e}")
        if obs_file:
            try:
                obs_df = pd.read_csv(obs_file)
            except Exception as e:
                st.error(f"Failed to read observations CSV: {e}")

    elif mode == 'excel':
        excel_file = st.file_uploader("Upload Excel workbook", type=['xlsx', 'xls'], key='excel_upload')
        filename = excel_file.name if excel_file else None
        selected_sheet = None
        if excel_file:
            try:
                xls = pd.ExcelFile(excel_file)
                sheets = xls.sheet_names
                selected_sheet = st.selectbox("Select sheet", options=sheets)
                if selected_sheet:
                    uploaded_df = pd.read_excel(xls, sheet_name=selected_sheet, header=None)
                    # Try to detect header row
                    # Convert to list of lists and find row with many non-empty values
                    arr = uploaded_df.fillna('').astype(str)
                    header_row_idx = 0
                    for r in range(min(10, len(arr))):
                        row = arr.iloc[r]
                        non_empty = (row.str.strip() != '').sum()
                        if non_empty >= (len(row) / 2) and non_empty > 0:
                            header_row_idx = r
                            break
                    # rebuild dataframe with detected header
                    df_values = pd.read_excel(excel_file, sheet_name=selected_sheet, header=None)
                    headers = df_values.iloc[header_row_idx].fillna('').astype(str).apply(lambda x: x.strip() or f'col_{pd.util.hash_pandas_object(pd.Series([x])).iloc[0]}')
                    data = df_values.iloc[header_row_idx + 1 :].reset_index(drop=True)
                    data.columns = headers
                    uploaded_df = data
            except Exception as e:
                st.error(f"Failed to read Excel: {e}")

with col2:
    st.markdown("**Import existing metadata**")
    myinfo = st.empty()
    metadata_file = st.file_uploader("Upload metadata (CSV or JSON TableSchema/CSVW)", type=['csv', 'json'], key='meta_upload')
    if 'metadata_df' not in st.session_state or metadata_file is None:
            myinfo.info("The metadata file is optional but should be a tabular data file with at least a **'name'** column that matches the headers of the uploaded data file. Optionally, it can include columns such as **'datatype', 'element', 'unit', 'method', and 'description'** for additional annotations.")

# Handle linked mode linking columns
if mode == 'linked' and site_df is not None and obs_df is not None:
    st.write("### Select linking columns")
    site_cols = list(site_df.columns)
    obs_cols = list(obs_df.columns)
    site_id_col = st.selectbox("ID column (sites)", options=[''] + site_cols)
    obs_fk_col = st.selectbox("Foreign key column (observations)", options=[''] + obs_cols)
    if site_id_col and obs_fk_col:
        st.write("Preview of linked join (left on observations -> sites):")
        try:
            merged = obs_df.merge(site_df, left_on=obs_fk_col, right_on=site_id_col, how='left', suffixes=('_obs', '_site'))
            st.dataframe(merged.head(5))
            uploaded_df = merged  # use merged for building metadata
        except Exception as e:
            st.error(f"Failed to merge tables: {e}")

# If metadata import provided, parse it
imported_metadata_df = None
if metadata_file is not None:
    imported_metadata_df = import_metadata_from_file(metadata_file)
    if imported_metadata_df is not None:
        st.success("Imported metadata file parsed.")

# If a dataframe is present (uploaded or merged), show preview and build metadata
if uploaded_df is not None:

    st.markdown("### Data preview (first 5 rows)")
    st.dataframe(uploaded_df.head(5))
    if 'metadata_df' not in st.session_state: #or st.session_state.get('source_df_id') != id(uploaded_df):
        meta_df = build_metadata_df_from_df(uploaded_df)
        st.session_state['metadata_df'] = meta_df
        st.session_state['source_df_id'] = id(uploaded_df)

    # allow user to import metadata and apply to current
    if imported_metadata_df is not None:
        #if st.button("Apply imported metadata to current columns"):
        st.session_state['metadata_df'] = apply_new_metadata_info(imported_metadata_df, st.session_state['metadata_df'])



    st.divider()

    st.markdown("### Preview Description")
    st.caption("Please, feel free to edit the descriptions directly in the table below or use some AI-power to help you.")

    expander_status = st.session_state.get("my_expander_is_expanded", False)

    col1_desc, col2_desc = st.columns([3, 2])
    
    with col1_desc:
        meta_df_description = st.session_state.get('metadata_df')
        edited_description = st.data_editor(meta_df_description[['name','description']],
                                            num_rows="dynamic",
                                            #height ='stretch',
                                            )
    

    with col2_desc:
        with st.expander("Need some LLM help?", expanded=False):
            st.markdown("**add some context files to start generation of descriptions with the help of an LLM**")
            uploaded_contextfiles = st.file_uploader("Upload context files", type=['pdf','doc','docx'], key='context_upload', accept_multiple_files=True)
            # Option to create a custom context file
            st.markdown("**Or create your own context file below:**")
            custom_context = st.text_area(
                "Write your custom context here:",
                placeholder="Type your context here...",
                height=200,
                key="custom_context_input"
            )


    # Add a key to store the variable descriptions in session state
    if "variable_descriptions" not in st.session_state:
        st.session_state["variable_descriptions"] = None

    if uploaded_contextfiles or custom_context.strip():
        all_context_text = ""
        with st.expander("Loaded context", expanded=True):
            with st.spinner("🧠 Reading your lovely context files..."):
                st.subheader("Extracted Context")
                st.caption("Please, feel free to redact any unimportant information from the context.")
                if uploaded_contextfiles:
                    for i, contextfile in enumerate(uploaded_contextfiles):
                        context_text=""
                        if contextfile.type == "application/pdf":
                            reader = PdfReader(contextfile)
                            for page in reader.pages:
                                context_text += page.extract_text() + "\n"
                        elif contextfile.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                            doc = docx.Document(contextfile)
                            for para in doc.paragraphs:
                                context_text += para.text + "\n"
                        # Use a unique key for each text area to store its state
                        key = f"context_text_{i}"
                        context_text = st.session_state.get(key, context_text)  # Load existing state or default
                        updated_text = st.text_area(
                            f"Context: {contextfile.name} [characters: {len(context_text)}]",
                            value=context_text,
                            height=300,
                            key=key
                        )
                        all_context_text += updated_text + "\n"
                                # Add custom context to the combined text
                    
                if custom_context.strip():
                    key = f"context_text_freefield"
                    updated_text = st.text_area(
                        f"Free text: [characters: {len(custom_context)}]",
                        value=custom_context,
                        height=300,
                        key=key
                    )
                    all_context_text += updated_text.strip() + "\n"
                    # Combine all updated text areas into one variable

                st.session_state["all_context_text"] = all_context_text

            AI_descriptions = None
            LLM_generator = st.button("AI generate Descriptions", key="generate_descriptions_button")
            if LLM_generator:
                meta_df_description = st.session_state.get('metadata_df')
                variable_names = list(meta_df_description['name'])

                AI_descriptions = generate_descriptions_with_LLM(var_list = variable_names,
                                                                context=all_context_text,
                                                                human_description=meta_df_description['description'])
                description_df = pd.DataFrame(
                    list(AI_descriptions.items()), columns=["name", "description"]
                )
                st.session_state["AI_var_descriptions"] = description_df

            if "AI_var_descriptions" in st.session_state:
                AI_var_df = st.session_state.get('AI_var_descriptions')
                st.subheader("Structured Variable Descriptions")
                st.caption("Please, feel free to edit the generated descriptions before approving them. By deleting the content of a description, the original description will be kept.")
                edited_AI_Var = st.data_editor(AI_var_df, width='stretch')
                st.session_state["AI_var_descriptions"] = edited_AI_Var
                approve_AI = st.button("✅ Approve and overwrite description with generated content", key="copy_AI_descriptions_button")
                if approve_AI:
                    # overwrite descriptions in main metadata df
                    meta_df = st.session_state.get('metadata_df')
                    meta_df_added = apply_new_metadata_info(edited_AI_Var, meta_df,overwrite = True)
                    st.session_state['metadata_df'] = meta_df_added
                    st.info("✅ Descriptions updated in main metadata table.")

    st.divider()

    st.markdown("### Element matcher")

    modelname = "models_local/all-MiniLM-L6-v2"
    model = load_sentence_model(modelname = modelname)


    index_vocabs, dict_vocabs = load_vocab_indexes(modelname = modelname)
    vocab_list = [v["label"] for v in dict_vocabs.values() if "label" in v]

    # Add a key to store the vocabulary matching results in session state
    if "vocab_matching_results" not in st.session_state:
        st.session_state["vocab_matching_results"] = None


    if st.button("Find Vocabulary Terms In Thesaury"):
        meta_df = st.session_state.get('metadata_df')

        with st.spinner("🔎 Finding nearest vocabulary terms..."):
            result_df = find_nearest_vocab(
                meta_df,                         # dataframe with searchterm. necesarry columns; ["Variable", "Description"]
                index_lib=index_vocabs,          # FAISS index
                meta_data_dict=dict_vocabs,      # mapping dict
                k_nearest=4
            )

        st.session_state["vocab_matching_results"] = result_df
        st.success("✅ Matching complete!")


    if st.session_state["vocab_matching_results"] is not None:
        result_df = pd.DataFrame(st.session_state["vocab_matching_results"])
        # Add a checkbox column if it doesn't exist
        if "Selected" not in result_df.columns:
            result_df.insert(1, "Selected", False) 
            # Set "Selected" to True for the row with the lowest Distance in each group of "query"
            idx_min_distance = result_df.groupby("query")["Distance"].idxmin()
            result_df.loc[idx_min_distance, "Selected"] = True

        styled_df = add_visual_row_group(result_df)

        column_list = styled_df.columns.tolist()
        disabling_columns= column_list.copy()
        disabling_columns.remove("Selected")
        priority_cols = [" ", "Selected", "query", "label", "definition"]
        new_order = [c for c in priority_cols if c in column_list] + [c for c in column_list if c not in priority_cols]


        edited_df = st.data_editor(
            styled_df,
            disabled =disabling_columns,
            column_order = new_order,
            num_rows="dynamic",
            width='stretch',
            key="editable_vocab_df_v2",
            hide_index=True
        )

        # Process the selected rows
        selected_rows = edited_df[edited_df["Selected"]].copy()

        # Drop duplicates while preserving original order of 'query'
        unique_queries = selected_rows["query"].drop_duplicates()

        if not selected_rows.empty:
            st.markdown("#### -> Selected Rows")
            
            summary_df = (
                selected_rows
                .groupby("query", sort=False, as_index=False)
                .agg({
                    "label": lambda x: list(set(x)),  # use set() to keep unique labels
                    "uri": lambda x: list(set(x))     # use set() to keep unique URIs
                })
            )

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

            st.session_state['metadata_df'] = apply_new_metadata_info(summary_df, st.session_state['metadata_df'], overwrite=True)

        else:
            st.info("No rows selected.")

    st.divider()

    st.markdown("### Column metadata")
    meta_df = st.session_state.get('metadata_df')

    # show suggestions helper
    st.write("Tip: click a cell to finetune the metadata.")

    edited = st.data_editor(meta_df, num_rows="dynamic")

    # validate datatype column to be allowed values
    if 'datatype' in edited.columns:
        edited['datatype'] = edited['datatype'].apply(lambda x: x if x in DATA_TYPE_OPTIONS else 'string')

    # Save edited meta back to session state
    st.session_state['metadata_df'] = edited

    # download buttons
    st.markdown("### Export metadata")
    # CSV
    csv_buf = io.StringIO()
    out_df = st.session_state['metadata_df'].copy()
    out_df.to_csv(csv_buf, index=False)
    csv_bytes = csv_buf.getvalue().encode('utf-8')
    download_bytes(csv_bytes, 'metadata.csv', 'text/csv')

    # TableSchema
    if st.button("Generate TableSchema JSON"):
        schema = {"fields": [], "primaryKey": None}
        for _, r in st.session_state['metadata_df'].iterrows():
            f = {"name": r['name']}
            if r.get('datatype'):
                f['type'] = r['datatype']
            if r.get('description'):
                f['description'] = r['description']
            if r.get('unit'):
                f['unit'] = r['unit']
            if r.get('method'):
                f['method'] = r['method']
            if r.get('element'):
                f['title'] = r['element']
            schema['fields'].append(f)
        st.json(schema, expanded=True)
        download_bytes(json.dumps(schema, indent=2).encode('utf-8'), 'tableschema.json', 'application/json')

    # CSVW
    if st.button("Generate CSVW JSON"):
        csvw = {
            "@context": "http://www.w3.org/ns/csvw",
            "url": filename,
            "tableSchema": {"columns": []}
        }
        for _, r in st.session_state['metadata_df'].iterrows():
            col = {"name": r['name']}
            if r.get('element'):
                col['titles'] = [r['element']]
            if r.get('element_uri'):
                col['propertyUrl'] = [element for _,element in ast.literal_eval(r['element_uri']).items()]
            if r.get('unit'):
                col['unit'] = r['unit']
            if r.get('method'):
                col['method'] = r['method']
            if r.get('datatype'):
                col['datatype'] = r['datatype']
            if r.get('description'):
                col['description'] = r['description']
            csvw['tableSchema']['columns'].append(col)
        st.json(csvw, expanded=True)
        download_bytes(json.dumps(csvw, indent=2).encode('utf-8'), 'csvw.json', 'application/json')

else:
    st.info("No file loaded yet. Use the controls above to upload a CSV or Excel file.")

