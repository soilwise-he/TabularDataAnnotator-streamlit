import streamlit as st
import pandas as pd
import io
import json
from typing import Optional, Sequence
from types import SimpleNamespace
import os
import re

import requests

import csv
from ui.blocks import add_Soilwise_contact_sidebar,add_Soilwise_logo, add_clear_cache_button
from util.metadata import apply_new_metadata_info

import html
from bs4 import BeautifulSoup

from urllib.parse import urlparse
import zipfile


add_Soilwise_logo()
add_clear_cache_button(key_prefix="input_page")

st.set_page_config(page_title="Tabular Soil Data Annotation", layout="wide")

# -------------------- Helper data and functions --------------------

# !! in UoM procedure, it's hardcoded filtered on "numeric"
DATA_TYPE_OPTIONS = ["string", "numeric", "date"]


def filename_from_url(url: str) -> str:
    filename = urlparse(url).path.split('/')[-1]
    if filename=="content":
        url_strip = url[:-len("/content")]
        filename=filename_from_url(url_strip)
    return filename

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


FORMAT_REGEX_DATES = {
    "%Y-%m-%d": r"^\d{4}-\d{2}-\d{2}$",
    "%d/%m/%Y": r"^\d{2}/\d{2}/\d{4}$",
    "%d-%m-%Y": r"^\d{2}-\d{2}-\d{4}$",
    "%Y/%m/%d": r"^\d{4}/\d{2}/\d{2}$",
    "%Y-%m": r"^\d{4}-\d{2}$",
    "%Y/%m": r"^\d{4}/\d{2}$",
    "%m-%Y": r"^\d{2}-\d{4}$",
    "%m/%Y": r"^\d{2}/\d{4}$",
}

def detect_date_format_from_series(s: pd.Series, sample_size: int = 200) -> str:
    """Infer the dominant date format in a pandas Series.

    Returns a strptime format token like %Y-%m-%d, %d/%m/%Y, etc.
    If no consistent format is detected, returns an empty string.
    """
    series = s.dropna().astype(str).str.strip()
    if len(series) == 0:
        return ""
    series = series.head(sample_size)

    patterns = {fmt: re.compile(regex) for fmt, regex in FORMAT_REGEX_DATES.items()}

    counts = {k: 0 for k in patterns.keys()}
    valid_total = 0

    for val in series:
        if not val:
            continue

        matched = False
        for fmt, pat in patterns.items():
            if not pat.match(val):
                continue

            try:
                pd.to_datetime(val, format=fmt, errors="raise")
                counts[fmt] += 1
                valid_total += 1
                matched = True
                break
            except Exception:
                continue

        if not matched:
            # fallback parse without fixed format (handles month names, etc.)
            try:
                pd.to_datetime(val, errors="raise")
                valid_total += 1
            except Exception:
                pass

    if valid_total == 0:
        return ""

    best_fmt = max(counts, key=counts.get)
    best_count = counts[best_fmt]

    # Require a dominant pattern before assigning a format.
    return best_fmt if best_count / valid_total >= 0.6 else ""


def build_metadata_df_from_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = []
    for c in df.columns:
        dtype = detect_column_type_from_series(df[c])
        date_format = detect_date_format_from_series(df[c]) if dtype == "date" else ""
        cols.append({
            "name": c,
            "datatype": dtype,
            "date format": date_format,
            "element": "",
            "unit": "",
            "method": "",
            "description": "",
            "element_uri": ""
        })
    return pd.DataFrame(cols)

@st.cache_data()
def read_csvBytes_with_sniffer(raw:bytes) -> pd.DataFrame:
    
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        import chardet
        enc = chardet.detect(raw)["encoding"] or "cp1252"
        text = raw.decode(enc, errors="replace")

    sample = text[:65536]  # first 64KB is usually enough
    dialect_uploaded = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
    separator_uploaded = dialect_uploaded.delimiter
    df = pd.read_csv(io.StringIO(text), sep=separator_uploaded)
    return df

@st.cache_data()
def import_metadata_from_file(uploaded_file) -> pd.DataFrame:
    # Accept CSV or JSON (tableschema or csvw)
    name = uploaded_file.name.lower()

    #text = uploaded_file.getvalue().decode("utf-8")
    raw = uploaded_file.getvalue()
    
    try:
        if name.endswith('.csv'):
            raw = uploaded_file.getvalue()
            df = read_csvBytes_with_sniffer(raw)
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
    
@st.cache_data()
def get_record_id_from_Zenodo_url(url:str):
    # Extract the record ID from a Zenodo URL
    match_record = re.search(r"zenodo.org/records/(\d+)", url)
    if match_record:
        return match_record.group(1)
    
    matc_doi = re.search(r"10.5281/zenodo.(\d+)", url)
    if matc_doi:
        return matc_doi.group(1)
    
    raise ValueError(f"Could not extract Zenodo record ID for '{url}'. Please ensure it's a valid Zenodo record URL or DOI.")

def get_metadata_from_Zenodo_id(record_id:int) -> dict:
    """Given a Zenodo record ID, fetch the record metadata and return it as a dictionary."""
    request_url = f"https://zenodo.org/api/records/{record_id}"
    headers = {
            'Content-Type': 'application/json'
            }
    try:
        response = requests.get(request_url, headers=headers)
        response.raise_for_status()
        response_json = response.json()

        if 'metadata' not in response_json:
            st.error("No metadata found in the Zenodo record.")
            raise ValueError("No metadata in record")
    except Exception as e:
        st.write(e)
        st.error(f"Failed to fetch or read metadata from URL: {url_input}")
    
    return response_json["metadata"]

@st.cache_data()
def get_files_URL_from_Zenodo_id(record_id:int,extensions: Optional[Sequence[str]] = None) -> list[str]:
    """Given a Zenodo record ID, fetch the record metadata and return a list of file URLs with specified extensions."""

    request_url = f"https://zenodo.org/api/records/{record_id}"
    headers = {
            'Content-Type': 'application/json'
            }
    try:
        response = requests.get(request_url, headers=headers)
        response.raise_for_status()
        response_json = response.json()

        if 'files' not in response_json or len(response_json['files']) == 0:
            st.error("No files found in the Zenodo record.")
            raise ValueError("No files in record")
    except Exception as e:
        st.write(e)
        st.error(f"Failed to fetch or read tabular data from URL: {url_input}")
        return []
        
    

    files = response_json["files"]

    if extensions:
        exts = tuple(
            (e if e.startswith(".") else f".{e}").lower()
            for e in extensions
        )
        filtered_files = [f for f in files if f["key"].lower().endswith(exts)]
    else:
        filtered_files = list(files)  # no filtering

    return [f["links"]["self"] for f in filtered_files]

@st.cache_data()
def request_file_from_zenodo(url:str) -> tuple[requests.Response, str, str]:
    try:
        file_response_head = requests.head(url)
        file_response_head.raise_for_status()
    except Exception as e:
        st.error(f"Failed to read file {url} from Zenodo record: {e}")
        return None, None, None
    name_file = filename_from_url(url)                
    ext_file = os.path.splitext(name_file)[1].lower()


    file_response = requests.get(url)

    return file_response, name_file, ext_file

def get_excel(excel_file):
    xls = pd.ExcelFile(excel_file)
    sheets = xls.sheet_names
    data_export=dict()
    for sheet in sheets:

        uploaded_df = pd.read_excel(xls, sheet_name=sheet, header=None)
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
        assert len(arr) > 0, f"Sheet is empty: '{sheet}' - file: '{excel_file.name}'"


        # rebuild dataframe with detected header
        df_values = pd.read_excel(excel_file, sheet_name=sheet, header=None)
        headers = df_values.iloc[header_row_idx].fillna('').astype(str).apply(lambda x: x.strip() or f'col_{pd.util.hash_pandas_object(pd.Series([x])).iloc[0]}')
        data = df_values.iloc[header_row_idx + 1 :].reset_index(drop=True)
        data.columns = headers
        data_export[sheet] = data
    return data_export


def description_to_plain_text(raw: str) -> str:
    if not raw:
        return ""

    # 1) Unescape HTML entities (&lt;h2&gt; → <h2>)
    unescaped = html.unescape(raw)

    # 2) Parse HTML
    soup = BeautifulSoup(unescaped, "html.parser")

    # Remove non-content tags
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # 3) Extract readable text
    text = soup.get_text(separator="\n", strip=True)

    # 4) Normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)  # collapse excessive newlines
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def is_safe_member(member_name: str) -> bool:
    # Prevent zip-slip and weird absolute paths
    normalized = member_name.replace("\\", "/")
    if normalized.startswith("/") or normalized.startswith("../") or "/../" in normalized:
        return False
    return True

def iter_interesting_zip_members(zf: zipfile.ZipFile, exts: list[str]):
    for info in zf.infolist():
        # skip directories
        if info.is_dir():
            continue
        if not is_safe_member(info.filename):
            continue
        _, ext = os.path.splitext(info.filename)
        ext = ext.lower()
        if ext in exts:
            yield info
            
@st.cache_resource
def process_zip_from_url(file_url: str, tabular_dict: dict, context_files: list, filename_dict: dict,
                         tabular_exts:list[str], context_exts:list[str]):
    r = requests.get(file_url)
    r.raise_for_status()
    filename = filename_from_url(file_url)

    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        # 1) tabular members
        for info in iter_interesting_zip_members(zf, tabular_exts):
            member_basename = os.path.basename(info.filename)
            with zf.open(info) as f:
                raw = f.read()

            # Reuse your existing readers:
            if member_basename.lower().endswith(".csv"):
                df = read_csvBytes_with_sniffer(raw)
                tabular_dict[f"{member_basename} (from {filename})"] = df
                filename_dict[f"{member_basename} (from {filename})"] = file_url
            elif member_basename.lower().endswith((".xlsx", ".xls")):
                bio = io.BytesIO(raw)
                bio.name = member_basename
                df_dict = get_excel(bio)
                for sheet_name, df in df_dict.items():
                    tabular_dict[f"{member_basename} | {sheet_name} (from {filename})"] = df
                    filename_dict[f"{member_basename} | {sheet_name} (from {filename})"] = file_url
        # 2) context members (optional): stash bytes so you can later run PDF/DOCX extraction
        for info in iter_interesting_zip_members(zf, context_exts):
            with zf.open(info) as f:
                raw = f.read()
            context_files.append({
                "source_zip": filename_from_url(file_url),
                "member_path": info.filename,
                "bytes": raw,
            })

# -------------------- UI --------------------
st.title("📖 STEP 1 : input of information")

st.markdown("""
            Ok, let's start by the firste uploading your data and eventually some existing metadata. But be assured, this tool is processing the information in a way that it never sends the raw data to any third party.
            Later in the app, there is an option to augment the information, making use of an LLM. Also in that case, only the variable names and the context text (if you choose to upload some) will be used.
            """)

st.markdown(""" --- """)

mode = st.radio("Input mode", options=["single CSV", "linked", "excel", "url - zenodo"], index=0, horizontal=True)


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




# File upload UI
col1, col2 = st.columns([1, 1])

tabular_dict = dict()
zipped_context_files = []
filename_dict = dict()  # to keep track of original filenames for tabular data


with col1:
    st.markdown("**Import tabular data**")
    if mode == 'single CSV':
        uploaded = st.file_uploader("Upload CSV file", type=['csv'], key='single_upload')
        filename = uploaded.name if uploaded else None
        if uploaded:
            try:
                raw = uploaded.getvalue()
                uploaded_df = read_csvBytes_with_sniffer(raw)
                tabular_dict[filename] = uploaded_df
                filename_dict[filename] = filename
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")

    elif mode == 'linked':
        site_file = st.file_uploader("Upload Sites CSV (locations)", type=['csv'], key='site_upload')
        obs_file = st.file_uploader("Upload Observations CSV", type=['csv'], key='obs_upload')
        filename = obs_file.name if obs_file else None
        if site_file:
            try:
                raw = site_file.getvalue()
                site_df = read_csvBytes_with_sniffer(raw)
                filename_dict[site_file.name] = site_file.name
            except Exception as e:
                st.error(f"Failed to read sites CSV: {e}")
        if obs_file:
            try:
                raw = obs_file.getvalue()
                obs_df = read_csvBytes_with_sniffer(raw)
                filename_dict[obs_file.name] = obs_file.name
            except Exception as e:
                st.error(f"Failed to read observations CSV: {e}")

    elif mode == 'excel':
        excel_file = st.file_uploader("Upload Excel workbook", type=['xlsx', 'xls'], key='excel_upload')
        filename = excel_file.name if excel_file else None
        selected_sheet = None
        if excel_file:
            try:
                df_dict = get_excel(excel_file)
                for sheet_name, df in df_dict.items():
                    tabular_dict[f"{filename} | {sheet_name}"] = df
                    filename_dict[f"{filename} | {sheet_name}"] = f"{filename}"
            except Exception as e:
                st.error(f"Failed to read Excel: {e}")
    elif mode == 'url - zenodo':
        url_input = st.text_input("Enter URL to zenodo records. (prefered format 'https://zenodo.org/records/{ID_number}')", key='url_input',value="https://zenodo.org/records/8083652")
        
        if url_input:
            # PIN : Zenodo Test URLS
            # https://zenodo.org/records/14034500 -> empty excel
            # https://zenodo.org/records/14726493 -> csv
            # https://zenodo.org/records/8083652 -> dubble excel
            # https://zenodo.org/records/10028494 -> complex zip
            # https://zenodo.org/records/17305831 -> AI4SoilHealth SOC
            #######################################################
            record_id = get_record_id_from_Zenodo_url(url_input)
            



            # retrieve tabular datafiles based on record_id
            filtered_extensions_tabular=['.csv','.xlsx','.xls']
            files_url_tabular = get_files_URL_from_Zenodo_id(record_id,extensions = filtered_extensions_tabular)

            # retrieve tabular datafiles based on record_id
            filtered_extensions_zip=['.zip']
            files_url_zip = get_files_URL_from_Zenodo_id(record_id,extensions = filtered_extensions_zip)


            # retrieve context files based on record_id  
            filtered_extensions_context=['.doc','.docx','.pdf','.md']
            files_url_context = get_files_URL_from_Zenodo_id(record_id,extensions = filtered_extensions_context)
            st.session_state['zenodo_context_files_url'] = files_url_context



            # retrieve metadata and save in session state for later use in context
            metadata_context_full = get_metadata_from_Zenodo_id(record_id)
            metadata_context = {k: metadata_context_full[k] for k in {"title","description"} if k in metadata_context_full}
            if "description" in metadata_context:
                metadata_context["description"] = description_to_plain_text(metadata_context["description"])
            st.session_state['zenodo_context_metadata'] = metadata_context



            for file_url in files_url_tabular:

                
                file_response,name_file, ext_file = request_file_from_zenodo(file_url)

                if ext_file in ['.xlsx','.xls']:
                    bitesIO = io.BytesIO(file_response.content)
                    bitesIO.name = name_file
                    df_dict = get_excel(bitesIO)
                    for sheet_name, df in df_dict.items():
                        tabular_dict[f"{name_file} | {sheet_name}"] = df
                        filename_dict[name_file] = file_url
                elif ext_file == '.csv':
                    uploaded_df = read_csvBytes_with_sniffer(file_response.content)
                    tabular_dict[name_file] = uploaded_df
                    filename_dict[name_file] = file_url
            for file_url in files_url_zip:
                st.write(f"diving into zip; {file_url}")
                process_zip_from_url(file_url,
                                        tabular_dict,
                                        zipped_context_files,
                                        filename_dict,
                                        tabular_exts=filtered_extensions_tabular,
                                        context_exts=filtered_extensions_context)
    st.session_state["zenodo_context_files_from_zip"] = zipped_context_files

    if  mode != 'url - zenodo':
        st.session_state.pop("zenodo_context_files_url", None)
        st.session_state.pop("zenodo_context_metadata", None)



with col2:
    st.markdown("**Import existing metadata**")
    myinfo = st.empty()
    metadata_file = st.file_uploader("Upload metadata (CSV or JSON TableSchema/CSVW)", type=['csv', 'json'], key='meta_upload')
    if 'metadata_df' not in st.session_state or metadata_file is None:
            myinfo.info("The metadata file is optional but should be a tabular data file with at least a **'name'** column that matches the headers of the uploaded data file. Optionally, it can include columns such as **'datatype'**, **'element'**, **'unit'**, **'method'**, and **'description'** for additional annotations.")

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
            tabular_dict[filename] = uploaded_df
        except Exception as e:
            st.error(f"Failed to merge tables: {e}")

meta_key = f"metadata_df"
st.session_state[meta_key] = tabular_dict



with col2:
    #TODO: change to dict !!!!
    # If metadata import provided, parse it
    imported_metadata_df = None
    if metadata_file is not None:
        imported_metadata_df = import_metadata_from_file(metadata_file)
        if imported_metadata_df is not None:
            st.success("Imported metadata file parsed.")



# If a dataframe is present (uploaded or merged), show preview and build metadata
if tabular_dict:

    st.markdown(""" --- """)
    st.markdown(f"### Data preview")
    st.caption("HINT: SHIFT+scroll to navigate the tabs horizontally if needed.")
    st.markdown("""
            <style>
                .stTabs [data-baseweb="tab-list"] {
                    gap: 2px;
                }

                .stTabs [data-baseweb="tab"] {
                    height: 50px;
                    background-color: #F0F2F6;
                    border-radius: 4px 4px 0px 0px;
                    padding-left: 12px;
                    padding-right: 12px;
                }

                .stTabs [aria-selected="true"] {
                    background-color: #FFFBF1;
                }

            </style>""", unsafe_allow_html=True)
    # Create one tab per key
    tab_labels = list(tabular_dict.keys())
    tabs = st.tabs(tab_labels)
    error_handling_tabs = st.empty()
    error_tabs = []
    error_tabs_explain = []
    tables_to_discard = set()
    tables_to_context = set()
    if "context_files" not in st.session_state:
        st.session_state["context_files"] = []

    for tab, key in zip(tabs, tab_labels):
        df = tabular_dict[key]

        with tab:


            discard_toggle_key = f"discard_table_{key}"
            move_toggle_key = f"move_to_context_{key}"

            discard_table = st.toggle(
                "Discard table for further annotation and processing",
                key=discard_toggle_key,
            )
            if not discard_table:
                st.session_state.pop(discard_toggle_key, None)

            move_to_context = st.toggle(
                "Move table to context tables",
                help = "If this table doesn't contain core tabular data but rather contextual information in a tabular format, you can move it to the context tables. It will later help the LLM to understand the context better.",
                key=move_toggle_key,
            )
            if not move_to_context:
                st.session_state.pop(move_toggle_key, None)

            if discard_table and move_to_context:
                st.warning("Both toggles are enabled. This table will be discarded from further investigation.")
                move_to_context = False
                st.session_state.pop(move_toggle_key, None)

            if discard_table:
                tables_to_discard.add(key)
                context_name = key if key.lower().endswith(".csv") else f"{key}.csv"
                st.session_state["context_files"] = [
                    f for f in st.session_state["context_files"]
                    if not (
                        getattr(f, "source", None) == "table"
                        and getattr(f, "name", None) == context_name
                    )
                ]
                st.info("This table will be excluded from metadata and downstream tabular processing.")
                continue

            if move_to_context:
                tables_to_context.add(key)
                st.info("This table will be added as a CSV context file and excluded from tabular processing.")
                continue
            else:
                # Toggle is off: remove any stale context-table entry for this table.
                context_name = key if key.lower().endswith(".csv") else f"{key}.csv"
                st.session_state["context_files"] = [
                    f for f in st.session_state["context_files"]
                    if not (
                        getattr(f, "source", None) == "table"
                        and getattr(f, "name", None) == context_name
                    )
                ]
            
            try:
                st.dataframe(df.head(5))
            except Exception as e:
                error_tabs.append(key)
                error_tabs_explain.append(e)
                tabular_dict.pop(key, None)
                filename_dict.pop(key, None)

                error_handling_tabs.error(
                            "🙄 couldn't load the data properly for  the following tab(s):\n\n"
                            + "\n".join(f"- {k}:  \n {e}" for k,e in zip(error_tabs,error_tabs_explain))
                            + "\n\n"
                            + "Handled this gracefully by completly ignoring these tabs. These will not be taken into account in further processing"
                    )
                st.error(f"Error displaying dataframe: {e}")
                continue

            source_id = id(df)
            meta_key = f"metadata_df"
            source_id_key = f"source_df_id"
            if meta_key not in st.session_state:
                st.session_state[meta_key] = {}

        
            if (meta_key not in st.session_state or st.session_state.get(source_id_key) != source_id):
                st.session_state[meta_key][key] = build_metadata_df_from_df(df)
                st.session_state[source_id_key] = source_id

            st.markdown("#### Metadata")
            st.caption("Change the datatype and date format as needed.")
            original_metadata_df = st.session_state[meta_key][key]
            all_columns = original_metadata_df.columns.tolist()
            edited_df = st.data_editor(
                            original_metadata_df,
                            width='stretch',
                            key=f"editor_{key}",
                            disabled=[col for col in all_columns if col != "datatype" and col != "date format"],
                            column_config={
                                "datatype": st.column_config.SelectboxColumn(
                                    options=DATA_TYPE_OPTIONS,
                                ),
                            },
                        )
            if not edited_df.equals(original_metadata_df):
                    # update the selected items
                    st.session_state[meta_key] = apply_new_metadata_info(
                                                    {key: edited_df},
                                                    st.session_state.get(meta_key),
                                                    overwrite='yes_incl_blanks'
                                                    )
                    st.rerun()               

    context_tables_added = 0
    if tables_to_context:
        existing_context_names = {
            getattr(f, "name", None)
            for f in st.session_state["context_files"]
            if getattr(f, "source", None) == "table"
        }

        for key in tables_to_context:
            df = tabular_dict.get(key)
            if df is None:
                continue

            context_name = key if key.lower().endswith(".csv") else f"{key}.csv"
            if context_name in existing_context_names:
                continue

            context_bytes = df.to_csv(index=False).encode("utf-8")
            st.session_state["context_files"].append(
                SimpleNamespace(
                    name=context_name,
                    content=context_bytes,
                    mime_type="text/csv",
                    source="table",
                    ext="csv",
                )
            )
            existing_context_names.add(context_name)
            context_tables_added += 1

    # Keep only context tables that are currently toggled on.
    desired_context_names = {
        k if k.lower().endswith(".csv") else f"{k}.csv"
        for k in tables_to_context
    }
    st.session_state["context_files"] = [
        f for f in st.session_state["context_files"]
        if (
            getattr(f, "source", None) != "table"
            or getattr(f, "name", None) in desired_context_names
        )
    ]

    tables_to_remove = tables_to_discard.union(tables_to_context)
    for key in tables_to_remove:
        tabular_dict.pop(key, None)
        filename_dict.pop(key, None)
        st.session_state[meta_key].pop(key, None)


    st.session_state["tabular_data_dict"] = {key: df.copy() for key, df in tabular_dict.items()}

    if "filename_dict" not in st.session_state:
            st.session_state["filename_dict"] = {}
    st.session_state["filename_dict"] = filename_dict

    # allow user to import metadata and apply to current
    #TODO: if import is changed to dict with keys matching the dataframes, change this part as well!!!!!!!!
    # if imported_metadata_df is not None or st.session_state.get('metadata_df_id') != id(imported_metadata_df):
    #     #if st.button("Apply imported metadata to current columns"):
    #     st.session_state['metadata_df'] = apply_new_metadata_info(imported_metadata_df, st.session_state['metadata_df'],overwrite='yes')
    #     st.session_state['metadata_df_id'] = id(imported_metadata_df)
        # st.info("""Metadata has been generated based on the uploaded dataset. \n \n ⏭️ You're ready for the next step""", icon="✅")
    st.info("""Metadata has been generated based on the uploaded dataset. You can still upload previous work on metadata and apply it to the generated metadata. \n \n ⏭️ You're ready for the next step on the following page""", icon="✅")

    st.markdown("## DEBUG OUTPUT")
    st.json(st.session_state[meta_key])
# -------------------- reach us --------------------
add_Soilwise_contact_sidebar()