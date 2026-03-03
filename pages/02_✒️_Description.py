import streamlit as st
import pandas as pd
import io
import json
from datetime import datetime
from typing import List, Dict, Optional
import os
from PyPDF2 import PdfReader
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
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from io import BytesIO


from docx import Document
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from docx.text.paragraph import Paragraph
from docx.table import Table
from docx.oxml.ns import nsmap


add_Soilwise_logo()
add_clear_cache_button(key_prefix="description_page")

# st.markdown("## DEBUG INPUT")
meta_key = f"metadata_df"
# st.json(st.session_state[meta_key])
# st.json(st.session_state)
# st.markdown("## END DEBUG INPUT")

st.set_page_config(page_title="Tabular Soil Data Annotation", layout="wide")

# -------------------- Helper data and functions --------------------

DATA_TYPE_OPTIONS = ["string", "numeric", "date"]


@dataclass
class ContextFile:
    name: str
    content: bytes
    mime_type: str
    source: str  # "url" | "upload"
    ext:str


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

@st.cache_resource()
def read_csv_with_sniffer(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
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


def apply_new_metadata_info(new_metadata_dict: dict, current_meta:dict, overwrite = 'no_overwrite') -> dict:
    # Function to apply new metadata info from metadata_df to current_meta

    # metadata_df expected to contain 'name' column
    if new_metadata_dict is None:
        return current_meta
    md_dict = current_meta.copy()
    for key, metadata_df in new_metadata_dict.items():

        if key not in md_dict:
            continue
        
        md= md_dict[key]
        for _, row in metadata_df.iterrows():
            name = row.get('name')
            if name in md['name'].values:
                idx = md.index[md['name'] == name][0]
                for col in ['datatype', 'element', 'unit', 'method', 'description','element_uri']:
                    current_value = md.at[idx, col]
                    
                    if overwrite == 'no_overwrite' and pd.notna(current_value) and current_value not in [None, ""]:
                        continue
                    if (overwrite == 'yes' and col in row and pd.notna(row[col]) and row[col] != "") or (overwrite == 'yes_incl_blanks'and col in row):
                        value = row[col]
                        if isinstance(value, dict):
                            md.loc[idx, col] = str(value)  # Convert dict to string
                        else:
                            md.loc[idx, col] = value
                        continue
    return md_dict


def _norm_mime(m: Optional[str]) -> str:
    """Normalize mime strings that sometimes come as 'a, b' etc."""
    if not m:
        return ""
    # take first part if multiple are present
    return m.split(",")[0].strip().lower()



def markdown_to_plain(md: str) -> str:
    md = re.sub(r"^#{1,6}\s*", "", md, flags=re.M)         # remove headings
    md = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1 (\2)", md)     # links: text (url)
    md = re.sub(r"`{1,3}.*?`{1,3}", "", md, flags=re.S)    # remove code fences/inline
    md = re.sub(r"\s+\n", "\n", md)                        # trailing spaces
    return md.strip()


def iter_block_items(doc):
    """
    Yield paragraphs and tables in document order.
    """
    for child in doc.element.body:
        if isinstance(child, CT_P):
            yield Paragraph(child, doc)
        elif isinstance(child, CT_Tbl):
            yield Table(child, doc)

def _tc_text(tc) -> str:
    # Gather all <w:t> text nodes inside this cell
    ts = tc.xpath(".//w:t")
    return "".join(t.text for t in ts if t.text).strip()


def iter_table_rows_text(table):
    """
    Yield rows as list[str], robust to irregular/merged/malformed tables.
    First try python-docx row.cells; if it crashes, fall back to XML.
    """
    for row in table.rows:
        try:
            # Normal path (works for well-formed tables)
            cells = row.cells
            row_text = [c.text.strip() for c in cells if c.text and c.text.strip()]
            yield row_text
        except ValueError:
            # Fallback path: read raw <w:tc> elements in that <w:tr>
            tr = row._tr
            tcs = tr.xpath("./w:tc")
            row_text = [_tc_text(tc) for tc in tcs]
            row_text = [t for t in row_text if t]
            yield row_text


@st.cache_resource
def read_context_file(contextfile: "ContextFile") -> str:
    """
    Reads ContextFile(content: bytes) and returns extracted text.
    Supports PDF, DOCX, Markdown, and plain text fallbacks.
    """
    context_text = ""
    mime = _norm_mime(getattr(contextfile, "mime_type", ""))
    suffix = getattr(contextfile, "ext", "").replace('.','')

    data = getattr(contextfile, "content", b"")
    bio = BytesIO(data)

    # --- PDF ---
    if mime == "application/pdf" or suffix == "pdf":
        reader = PdfReader(bio)
        parts = []
        for page in reader.pages:
            parts.append((page.extract_text() or "").strip())
        context_text = "\n\n".join([p for p in parts if p])

    # --- DOCX ---
    #BUG: ignores tables in files!! Test with zenodo; https://zenodo.org/records/17305831
    elif (
        mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        or suffix == "docx"
    ):
        doc = Document(bio)
  
        chunks = []

        for block in iter_block_items(doc):
            if isinstance(block, Paragraph):
                if block.text.strip():
                    chunks.append(block.text.strip())

            elif isinstance(block, Table):
                chunks.append("[TABLE]")
                for row_text in iter_table_rows_text(block):
                    if row_text:
                        chunks.append(" | ".join(row_text))
                chunks.append("[/TABLE]")

        context_text = "\n".join(chunks)

    # --- Older DOC (binary) ---
    # python-docx can't read .doc; you'd need libreoffice/antiword, etc.
    elif suffix == "doc":
        context_text = (
            "[DOC binary detected: cannot extract reliably with python-docx. "
            "Please convert to .docx or PDF.]"
        )

    # --- Markdown / plain text ---
    elif suffix in {"md", "markdown"} or mime.startswith("text/") or mime == "application/octet-stream":
        # Try utf-8 first; fallback to latin-1 if needed
        try:
            context_text = data.decode("utf-8")
        except UnicodeDecodeError:
            context_text = data.decode("latin-1", errors="replace")
        
        context_text = markdown_to_plain(context_text)
        

    else:
        # last-resort: attempt decode as text
        try:
            context_text = data.decode("utf-8")
        except UnicodeDecodeError:
            context_text = data.decode("latin-1", errors="replace")

    return context_text.strip()


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


def generate_descriptions_with_LLM(var_list: List[str],
                                   context: str,
                                   human_description: Dict,
                                   cache_feedback,
                                   tablename:str) -> Dict[str, str]:
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
        cache_feedback.info(f"✅ Found cached result (version {cached['version']}, saved {cached['timestamp']}) - {tablename}")
        raw_output = cached["response"]
    else:

        with st.spinner(f"🪶 The gnomes are working in colaboration with {provider_selected}..."):
            callfunc = LLM_selection_info.get(provider_selected, {}).get("callfunction")
            raw_output = callfunc(prompt)


        store_result(context, var_list, raw_output,provider_selected)
        cache_feedback.success(f"✅ Saved new response to cache for {tablename}")
        
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

@st.cache_resource
def load_file_from_url(url: str) -> ContextFile:
    response = requests.get(url)
    response.raise_for_status()

    name = filename_from_url(url)
    mime = response.headers.get("Content-Type", "application/octet-stream")
    extention = name.split('.')[-1]

    return ContextFile(
        name=name,
        content=response.content,
        mime_type=mime,
        source="url",
        ext=extention
    )

def filename_from_url(url: str) -> str:
    filename = urlparse(url).path.split('/')[-1]
    if filename=="content":
        url_strip = url[:-len("/content")]
        filename=filename_from_url(url_strip)
    return filename

# PIN -------------------- UI --------------------
st.title("📖 STEP 2 : Description of data")

st.markdown("""
            Next step, describing the variables in humanly understandable language. This will nog only help yourself to document your data but also others who might use it in the future. And even our computers will understand the variables better when we will try to link them to vocabularies in the next step.  \n
            You can add descriptions manually, or let an LLM help you out by providing some context files (e.g. a data dictionary, a manual, a scientific paper,...). The LLM will then try to extract relevant information from the context and generate concise descriptions for your variables.""")

st.markdown(""" --- """)

with st.sidebar.expander("LLM Provider", expanded=True):
    st.write("Choose which LLM backend to use and check API key availability.")
    

    LLM_selection_info = {
        "OpenAI": {
            'key' : st.secrets.get("openai", {}).get("api_key"),
            'callfunction': get_response_OpenAI,
            'help_key_creation': 'https://platform.openai.com/account/api-keys',
        },
        # "Apertus": {
        #     'key' : st.secrets.get("apertus", {}).get("api_key"),
        #     'callfunction': get_response_Apertus,
        #     'help_key_creation': 'https://platform.publicai.co/settings/api-keys',
        # },
        " Anthropic": {
            'key' : st.secrets.get("Claude", {}).get("api_key"),
            'callfunction': None,
            'help_key_creation': 'https://platform.claude.com/settings/keys',
        },
        "Test_Random": {
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
            placeholder_error.error(f"❌ No API key for {provider_selected} found in st.secrets. Generete one here; {LLM_selection_info[provider_selected]['help_key_creation']}. Once you have the key, provide it below.")
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


if meta_key not in st.session_state or st.session_state.get(meta_key) is None:
    st.info("No metadata found. Please upload a data file with recognizable columns or a metadata file in step 1 to proceed with descriptions.")
    
else:
    

    st.markdown("### Manual variable descriptions")
    st.caption("Want some LLM power to help? keep scrolling down!")
    meta_dict_description = st.session_state.get(meta_key)
    tab_labels = list(meta_dict_description.keys())
    tabs = st.tabs(tab_labels)
    
    for tab, key in zip(tabs, tab_labels):
        with tab:
            meta_df_description = meta_dict_description[key]
            edited_description = st.data_editor(meta_df_description[['name','description']],
                                        num_rows="dynamic",
                                        disabled=["name"],
                                        #height ='stretch',
                                        key=f"description_editor_{key}"
                                        )

            # update session state on change
            if not meta_df_description[['name','description']].equals(edited_description):
                st.session_state["metadata_df"] = apply_new_metadata_info(
                                                    {key: edited_description},
                                                    st.session_state.get("metadata_df"),
                                                    overwrite='yes'
                                                    )
                st.rerun() # Not uptimal, but necessary to update the edited df in session_state. Bug is know in streamlit community
            #BUG: when description is generated with LLM, and you change something here. Some eternal loop is created!!!

    st.markdown("### Augmented variable descriptions")
    with st.expander("Need some LLM help?", expanded=False):


        container_3 = st.empty()
        button_A_3 = container_3.button('ℹ️  ')
        if button_A_3:
            container_3.empty()
            button_B_3 = container_3.button("ℹ️ Attention: Your header names and context text will be securely sent to an external source to auto generate descriptions. Please avoid including sensitive or personal information in the context files or variable names.")
        
        
        if "context_files" not in st.session_state:
            st.session_state["context_files"] = []

        if "zenodo_loaded" not in st.session_state:
            st.session_state["zenodo_loaded"] = False

        if not st.session_state["zenodo_loaded"]:
            if st.session_state.get("zenodo_context_files_url"):
                for url in st.session_state["zenodo_context_files_url"]:
                    filename = filename_from_url(url)
                    if not any(f.name == filename for f in st.session_state["context_files"]):
                        st.session_state["context_files"].append(load_file_from_url(url))
            st.session_state["zenodo_loaded"] = True

    

        st.markdown("**add some context files to start generation of descriptions with the help of an LLM**")
        upload_key = st.session_state.get("upload_key", 0)
        uploaded_contextfiles = st.file_uploader("Upload context files",
                                                 type=['pdf','doc','docx','md'],
                                                 key=f'context_upload_{upload_key}',
                                                 accept_multiple_files=True)

        
        if uploaded_contextfiles:
            for f in uploaded_contextfiles:
                if not any(existing.name == f.name for existing in st.session_state["context_files"]):
                    st.session_state["context_files"].append(
                        ContextFile(
                            name=f.name,
                            content=f.getvalue(),
                            mime_type=f.type,
                            source="upload",
                            ext=f.name.split('.')[-1]
                        )
                    )

            # ✅ Clear uploader so its UI looks empty again
            # UI is duplicated under st.markdown("### Context files")
            if not upload_key in st.session_state:
                st.session_state["upload_key"] = 0
            st.session_state["upload_key"] += 1
            st.rerun()

        
        st.markdown("### Context files")

        for i, f in enumerate(st.session_state["context_files"]):
            col1, col2, col3 = st.columns([6, 2, 1])

            with col1:
                st.markdown(f"📄 **{f.name}**")

            with col2:
                st.caption("🔗 Zenodo" if f.source == "url" else "⬆️ Uploaded")

            with col3:
                if st.button("❌", key=f"remove_{i}"):
                    st.session_state["context_files"].pop(i)
                    st.rerun()


        if "zenodo_meta_loaded" not in st.session_state:
            st.session_state["zenodo_meta_loaded"] = False

        #if not st.session_state["zenodo_meta_loaded"]:

        if st.session_state.get("zenodo_context_metadata"):
            meta = st.session_state["zenodo_context_metadata"]

            fields = ("title", "description")
            suggested_text = "  \n".join(
                f"{k}: {meta.get(k, '')}"
                for k in fields
                if 
                meta.get(k) not in (None, "")
            )
            text_area_header = "Write your custom context here, we pre-filled it with the title and abstract from zenodo:"
        else:
            text_area_header = "Write your custom context here:"
            suggested_text = ""
            
            # st.session_state["zenodo_meta_loaded"] = True


        if st.session_state.get("zenodo_context_metadata"):
            text_area_header = "Write your custom context here, we pre-filled it with the title and abstract from zenodo:"
        else:
            text_area_header = "Write your custom context here:"


        # Option to create a custom context file
        st.markdown("**Or create your own context file below:**")
        custom_context = st.text_area(
            text_area_header,
            placeholder="Type your context here...",
            height=200,
            key="custom_context_input",
            value=suggested_text,
        )

        placeholder_loaded_context = st.empty()




    # Add a key to store the variable descriptions in session state
    if "variable_descriptions" not in st.session_state:
        st.session_state["variable_descriptions"] = None

    if st.session_state["context_files"] or custom_context.strip():
        all_context_text = ""
        expanded_load_context = False if custom_context==suggested_text and not st.session_state["context_files"] else True
        with placeholder_loaded_context.expander(" -> Loaded context", expanded=expanded_load_context):
            with st.spinner("🧠 Reading your lovely context files..."):
                st.subheader("Extracted Context")
                st.caption("Please, feel free to redact any unimportant information from the context.")
                if st.session_state["context_files"] :
                    for i, contextfile in enumerate(st.session_state["context_files"] ):
                        context_text=""
                        context_text = read_context_file(contextfile)
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

                meta_dict_description = st.session_state.get(meta_key)
                
                status_box = st.empty()
                cache_box = st.empty()
                completed = []

                for key,meta_df_description in meta_dict_description.items():
                    variable_names = list(meta_df_description['name'])

                    AI_descriptions = generate_descriptions_with_LLM(var_list = variable_names,
                                                                    context = all_context_text,
                                                                    human_description = meta_df_description['description'],
                                                                    cache_feedback  = cache_box,
                                                                    tablename = key)
                    description_df = pd.DataFrame(
                        list(AI_descriptions.items()), columns=["name", "description"]
                    )

                    completed.append(key)

                    
                    status_box.success(
                            "💪 Descriptions ready for:\n\n"
                            + "\n".join(f"- {k}" for k in completed)
                    )
                        


                    if "AI_var_descriptions" not in st.session_state:
                        st.session_state["AI_var_descriptions"] = {}
                    st.session_state["AI_var_descriptions"][key] = description_df

            if "AI_var_descriptions" in st.session_state:
                AI_var_dict = st.session_state.get('AI_var_descriptions')
                st.subheader("Structured Variable Descriptions")
                st.caption("Please, feel free to edit the generated descriptions before approving them. By deleting the content of a description, the original description will be kept.")
                
                tab_labels = list(AI_var_dict.keys())
                tabs = st.tabs(tab_labels)
                
                for tab, key in zip(tabs, tab_labels):
                    AI_var_df = AI_var_dict[key]

                    with tab:
                        try:
                            edited_AI_Var = st.data_editor(AI_var_df,
                                                           width='stretch',
                                                           disabled=["name"],
                                                           column_config={"name": st.column_config.Column(
                                                               width="small"
                                                                )},
                                                           key=f"preview_desc_editor_{key}")
                        except Exception as e:
                            st.error(f"Error displaying dataframe: {e}")
                            continue
                
                        if not edited_AI_Var.equals(AI_var_df):
                            st.session_state["AI_var_descriptions"][key] = edited_AI_Var
                            st.rerun() # Not uptimal, but necessary to update the edited df in session_state. Bug is know in streamlit community
                approve_AI = st.button("✅⏬ Approve and overwrite description with generated content", key="copy_AI_descriptions_button")
                if approve_AI:
                    # overwrite descriptions in main metadata df
                    meta_dict_description = st.session_state.get(meta_key)

                    meta_dict_added = apply_new_metadata_info(st.session_state["AI_var_descriptions"],
                                                            meta_dict_description,
                                                            overwrite = 'yes')
                    
                    st.session_state[meta_key] = meta_dict_added

                    st.info("✅ Descriptions updated in main metadata table.")

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
    
    st.space()

    percent_progress = descriptions_completed/total_size
    st.progress(percent_progress,f"{percent_progress*100:.1f} % of desciptions provided")
    if percent_progress<0.1:
        st.error("Descriptions help your dataset to be processed easier. Please provide some descriptions, we even build a LLM functionality to releave some of the heavy lifting")
    elif percent_progress <0.8:
        st.info("You're on a roll, almost completed the whole set. ")
    elif percent_progress<1:
        st.success("Nice, ")
    else:
        st.success("Your dataset is ready for it's next step 💪  \n⏭️ Go for it! The next page is waiting for you")

# -------------------- reach us --------------------
add_Soilwise_contact_sidebar()