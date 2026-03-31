import streamlit as st
import pandas as pd
import io
import json
import ast
import zipfile
import re
import socket
import threading
import tempfile
import time
import webbrowser
from functools import partial
from pathlib import Path
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

from ui.blocks import add_Soilwise_logo, add_Soilwise_contact_sidebar, add_clear_cache_button

from csvwlib import CSVWConverter

add_Soilwise_logo()
add_Soilwise_contact_sidebar()
add_clear_cache_button(key_prefix="export_page")
meta_key = "metadata_df"

st.title("💾 Export Metadata")

st.markdown("""
Export your annotated metadata in various standardized formats:
- **CSV**: Simple tabular format for spreadsheets
- **TableSchema JSON**: Frictionless Data standard format
- **CSVW JSON**: W3C CSV on the Web format
""")

# Helper function for download
def download_bytes(content: bytes, filename: str, mime: str = 'application/octet-stream'):
    st.download_button(label=f"📥 Download {filename}", data=content, file_name=filename, mime=mime)


def _safe_filename_component(value: str, fallback: str = "export") -> str:
    text = str(value) if value is not None else ""
    # Windows-invalid filename chars: <>:"/\\|?* plus control chars.
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", text)
    text = text.strip().strip(".")
    return text or fallback


def _parse_element_uri(raw_value):
    if not raw_value:
        return None
    try:
        parsed = ast.literal_eval(raw_value) if isinstance(raw_value, str) else raw_value
    except (ValueError, SyntaxError):
        return None
    if isinstance(parsed, dict):
        #TODO: list items can't be handled in CSVW
        _temp = list(parsed.values())
        return _temp[0] if _temp else None
    return None


def _csvw_column_from_row(row):
    col = {"name": row['name']}

    if row.get('element'):
        col['titles'] = row['element']

    property_urls = _parse_element_uri(row.get('element_uri'))
    if property_urls:
        col['propertyUrl'] = property_urls

    if row.get('unit'):
        col['schema:unitCode'] = row['unit']
    if row.get('method'):
        col['dc:method'] = row['method']
    if row.get('datatype'):
        if row['datatype'] == 'date':
            date_format = row.get('date format')
            col['datatype'] = {"base": "date", "format": date_format} if date_format else "date"
        else:
            col['datatype'] = row['datatype']
    if row.get('description'):
        col['dc:description'] = row['description']

    return col


def _build_csvw_table(table_df, url, foreign_keys=None):
    pk_rows = table_df[table_df.get("primary key", pd.Series(False, index=table_df.index)).astype(bool)]
    pk = pk_rows["name"].iloc[0] if not pk_rows.empty else None

    table_schema = {
        "columns": [_csvw_column_from_row(r) for _, r in table_df.iterrows()],
    }
    if pk:
        table_schema["primaryKey"] = pk
    
    if foreign_keys:
        table_schema["foreignKeys"] = foreign_keys

    return {
        "url": url,
        "tableSchema": table_schema,
    }


def _build_foreign_keys(table_key: str, relationships_summary_df: pd.DataFrame, filename_dict: dict) -> list:
    """Build foreignKeys array for a given table based on table relationships.
    
    Args:
        table_key: The table name for which to build foreign keys
        relationships_summary_df: DataFrame containing all table relationships
        filename_dict: Dictionary mapping table names to their filenames
        
    Returns:
        List of foreign key definitions
    """
    foreign_keys = []
    
    if relationships_summary_df is None or relationships_summary_df.empty:
        return foreign_keys
    
    # Filter relationships where this table is the left table (the referencing table)
    relevant_rels = relationships_summary_df[
        (relationships_summary_df["left_table"] == table_key) & 
        (relationships_summary_df["relation"] != "not linked") &
        (relationships_summary_df["left_id"].notna()) &
        (relationships_summary_df["left_id"] != "") &
        (relationships_summary_df["right_id"].notna()) &
        (relationships_summary_df["right_id"] != "")
    ]
    
    for _, rel in relevant_rels.iterrows():
        right_table = rel["right_table"]
        resource_url = filename_dict.get(right_table, f"{_safe_filename_component(right_table, 'table')}.csv")
        
        foreign_key = {
            "columnReference": rel["left_id"],
            "reference": {
                "resource": resource_url,
                "columnReference": rel["right_id"]
            }
        }
        foreign_keys.append(foreign_key)
    
    return foreign_keys


def _coerce_table_to_dataframe(table_obj, metadata_df: pd.DataFrame) -> pd.DataFrame | None:
    """Normalize supported table payloads to DataFrame for CSV broadcast.

    Supports DataFrame, raw CSV bytes, and raw CSV string.
    """
    if isinstance(table_obj, pd.DataFrame):
        return table_obj

    if isinstance(table_obj, (bytes, bytearray)):
        try:
            return pd.read_csv(io.BytesIO(table_obj))
        except Exception:
            return None

    if isinstance(table_obj, str):
        try:
            return pd.read_csv(io.StringIO(table_obj))
        except Exception:
            return None

    return None


def _normalize_table_key(value: str) -> str:
    text = str(value or "").strip().lower()
    if text.endswith(".csv"):
        text = text[:-4]
    return text


def _looks_like_metadata_df(df: pd.DataFrame) -> bool:
    metadata_columns = {
        "name", "datatype", "description", "unit", "method", "element", "element_uri", "date format"
    }
    cols = {str(c).strip().lower() for c in df.columns}
    return "name" in cols and (len(cols & metadata_columns) >= 3)


def _resolve_data_table_for_metadata(table_key: str, metadata_df: pd.DataFrame, data_by_table: dict) -> pd.DataFrame | None:
    """Find the matching source data table for a metadata table key.

    Tries direct key, normalized key aliases, then column-overlap matching.
    Rejects metadata-like DataFrames to avoid exporting metadata rows as CSV data.
    """
    metadata_column_names = set()
    if isinstance(metadata_df, pd.DataFrame) and "name" in metadata_df.columns:
        metadata_column_names = {
            str(v).strip() for v in metadata_df["name"].dropna().tolist() if str(v).strip()
        }

    # 1) direct lookup
    direct_candidate = _coerce_table_to_dataframe(data_by_table.get(table_key), metadata_df)
    if isinstance(direct_candidate, pd.DataFrame) and not _looks_like_metadata_df(direct_candidate):
        return direct_candidate

    # 2) normalized key lookup (with/without .csv)
    normalized_target = _normalize_table_key(table_key)
    for candidate_key, candidate_obj in data_by_table.items():
        if _normalize_table_key(candidate_key) != normalized_target:
            continue
        candidate_df = _coerce_table_to_dataframe(candidate_obj, metadata_df)
        if isinstance(candidate_df, pd.DataFrame) and not _looks_like_metadata_df(candidate_df):
            return candidate_df

    # 3) fallback by maximum overlap with expected column names
    best_df = None
    best_overlap = -1
    for _, candidate_obj in data_by_table.items():
        candidate_df = _coerce_table_to_dataframe(candidate_obj, metadata_df)
        if not isinstance(candidate_df, pd.DataFrame) or _looks_like_metadata_df(candidate_df):
            continue

        candidate_cols = {str(c).strip() for c in candidate_df.columns}
        overlap = len(metadata_column_names & candidate_cols)
        if overlap > best_overlap:
            best_overlap = overlap
            best_df = candidate_df

    if best_df is not None and best_overlap > 0:
        return best_df

    return None


def build_csvw_frame(metadata_by_table, fallback_filename, filename_dict, relationships_summary_df=None):

    context = [
        "http://www.w3.org/ns/csvw",
        {"qudt": "http://qudt.org/vocab/unit"}
    ]

    table_entries = []
    for table_key, metadata_df in metadata_by_table.items():
        table_url = filename_dict.get(table_key, fallback_filename)
        foreign_keys = _build_foreign_keys(table_key, relationships_summary_df, filename_dict)
        table_entries.append(_build_csvw_table(metadata_df, table_url, foreign_keys if foreign_keys else None))

    if len(table_entries) == 1:
        table = table_entries[0]
        return {
            "@context": context,
            "url": table["url"],
            "tableSchema": table["tableSchema"]
        }

    return {
        "@context": context,
        "tables": table_entries
    }


def _find_free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _unique_csv_name(table_key: str, used_names: set[str]) -> str:
    base = _safe_filename_component(table_key, fallback="table")
    candidate = f"{base}.csv"
    idx = 2
    while candidate in used_names:
        candidate = f"{base}_{idx}.csv"
        idx += 1
    used_names.add(candidate)
    return candidate


def _generate_rdf_ttl_local(metadata_by_table: dict, data_by_table: dict) -> str:
    if not metadata_by_table:
        raise ValueError("No metadata available for RDF export.")

    with tempfile.TemporaryDirectory(prefix="csvw_export_") as tmpdir:
        tmp_path = Path(tmpdir)

        used_names = set()
        table_entries = []
        missing_data_tables = []

        for table_key, metadata_df in metadata_by_table.items():
            table_df = _resolve_data_table_for_metadata(table_key, metadata_df, data_by_table)
            if table_df is None:
                missing_data_tables.append(table_key)
                continue

            csv_name = _unique_csv_name(table_key, used_names)
            (tmp_path / csv_name).write_text(
                table_df.to_csv(index=False),
                encoding="utf-8",
            )
            table_entries.append(_build_csvw_table(metadata_df, csv_name))

        if missing_data_tables:
            raise ValueError(
                "Missing tabular data for: " + ", ".join(map(str, missing_data_tables))
            )

        csvw_local = {
            "@context": [
                "http://www.w3.org/ns/csvw",
                {"qudt": "http://qudt.org/vocab/unit"},
            ],
            "tables": table_entries,
        }

        metadata_name = "csvw-metadata.json"
        metadata_path = tmp_path / metadata_name
        metadata_path.write_text(
            json.dumps(csvw_local, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        port = _find_free_port()
        handler = partial(SimpleHTTPRequestHandler, directory=str(tmp_path))
        server = ThreadingHTTPServer(("127.0.0.1", port), handler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()

        try:
            metadata_url = f"http://127.0.0.1:{port}/{metadata_name}"
            csv_url = f"http://127.0.0.1:{port}/{table_entries[0]['url']}"

            # # Debug helper: open generated local CSV in default browser.
            # if DEBUG_OPEN_CSV_URL_DURING_RDF:
            #     try:
            #         webbrowser.open_new_tab(csv_url)
            #         if DEBUG_OPEN_CSV_PAUSE_SECONDS > 0:
            #             time.sleep(DEBUG_OPEN_CSV_PAUSE_SECONDS)
            #     except Exception:
            #         pass

            ttl = CSVWConverter.to_rdf(csv_url=csv_url, metadata_url=metadata_url, format="ttl")
            if isinstance(ttl, bytes):
                return ttl.decode("utf-8")

            return str(ttl)
            
        finally:
            server.shutdown()
            server.server_close()


# Check if metadata exists in session state
if meta_key not in st.session_state or not st.session_state[meta_key]:
    st.warning("⚠️ No metadata available. Please annotate your data first on the previous pages.")
    st.stop()

st.markdown("### Preview of Metadata")
meta_tabs = st.tabs(list(st.session_state[meta_key].keys()))
for tab, table_key in zip(meta_tabs, st.session_state[meta_key].keys()):
    with tab:
        metadata_df = st.session_state[meta_key][table_key]
        st.dataframe(metadata_df, width='stretch')

st.divider()
st.markdown("### Export Options")

# TODO; check usage and implementation for mulitpple tables
# Get the filename from session state if available
filename = st.session_state.get('uploaded_filename', 'data.csv')

# ==================== CSV EXPORT ====================
st.markdown("#### 1️⃣ CSV Export")
st.caption("Simple comma-separated values format")

csv_payloads = {}
for table_key, metadata_df in st.session_state[meta_key].items():
    safe_table_key = _safe_filename_component(table_key, fallback="table")
    csv_buf = io.StringIO()
    metadata_df.copy().to_csv(csv_buf, index=False)
    csv_payloads[f'{safe_table_key}_metadata.csv'] = csv_buf.getvalue().encode('utf-8')

if len(csv_payloads) > 1:
    column_zip, column_individual = st.columns([2,5])
    # Primary action: download all metadata CSV files as a single ZIP archive.
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        for export_filename, export_bytes in csv_payloads.items():
            zf.writestr(export_filename, export_bytes)
    with column_zip:
        download_bytes(zip_buf.getvalue(), 'metadata_csv_exports.zip', 'application/zip')

    with column_individual:
        with st.expander("Download individual CSV files", expanded=False):
            for export_filename, export_bytes in csv_payloads.items():
                download_bytes(export_bytes, export_filename, 'text/csv')
else:
    for export_filename, export_bytes in csv_payloads.items():
        download_bytes(export_bytes, export_filename, 'text/csv')

st.divider()

# ==================== TABLESCHEMA JSON ====================
st.markdown("#### 2️⃣ TableSchema JSON")
st.caption("Frictionless Data standard format - [Learn more](https://specs.frictionlessdata.io/table-schema/)")

if st.button("Generate TableSchema JSON", key="tableschema_button"):
    schema = {"fields": [], "primaryKey": None}
    for _, r in metadata_df.iterrows():
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

st.divider()

# ==================== CSVW JSON ====================
st.markdown("#### 3️⃣ CSVW JSON")
st.caption("W3C CSV on the Web format - [Learn more](https://csvw.org/standards.html)")

if st.button("Generate CSVW JSON", key="csvw_button"):
    filename_dict = st.session_state.get("filename_dict", {})
    relationships_summary_df = st.session_state.get("table_relationships_summary_df", pd.DataFrame())
    
    csvw_frame = build_csvw_frame(
        metadata_by_table=st.session_state[meta_key],
        fallback_filename=filename,
        filename_dict=filename_dict,
        relationships_summary_df=relationships_summary_df if not relationships_summary_df.empty else None,
    )
    
    st.json(csvw_frame)
    download_bytes(json.dumps(csvw_frame, indent=2).encode('utf-8'), '_SoilWise.json', 'application/json')

st.divider()

# ==================== RDF ====================
st.markdown("#### 4️⃣ RDF")
st.caption("Resource Description Framework format - [Learn more](https://www.w3.org/RDF/)")

if st.button("Generate RDF", key="rdf_button"):
    tabular_data_dict = st.session_state.get("tabular_data_dict", {})
    rdf_payloads = {}
    rdf_errors = []

    for table_key, metadata_df in st.session_state[meta_key].items():
        safe_table_key = _safe_filename_component(table_key, fallback="table")
        one_table_metadata = {table_key: metadata_df}


        try:
            ttl_text = _generate_rdf_ttl_local(
                metadata_by_table=one_table_metadata,
                data_by_table=tabular_data_dict,
            )
            rdf_payloads[f"{safe_table_key}.ttl"] = ttl_text.encode("utf-8")
        except Exception as e:
            rdf_errors.append(f"{table_key}: {e}")

    if rdf_errors:
        st.error(
            "Failed to generate RDF for:\n\n"
            + "\n".join(f"- {err}" for err in rdf_errors)
        )

    if not rdf_payloads:
        st.warning("No RDF output generated.")
    elif len(rdf_payloads) > 1:
        column_zip, column_individual = st.columns([2, 5])
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
            for export_filename, export_bytes in rdf_payloads.items():
                zf.writestr(export_filename, export_bytes)

        with column_zip:
            download_bytes(zip_buf.getvalue(), 'metadata_rdf_exports.zip', 'application/zip')

        with column_individual:
            with st.expander("Download individual RDF files", expanded=False):
                for export_filename, export_bytes in rdf_payloads.items():
                    download_bytes(export_bytes, export_filename, 'text/turtle')
    else:
        for export_filename, export_bytes in rdf_payloads.items():
            st.code(export_bytes.decode("utf-8"), language="turtle")
            download_bytes(export_bytes, export_filename, 'text/turtle')


st.markdown("### 💡 Tips")
st.info("""
- **CSV**: Best for sharing with colleagues or importing into spreadsheet applications
- **TableSchema**: Use when sharing data that follows Frictionless Data standards
- **CSVW**: Ideal for semantic web and linked data applications

All formats preserve your metadata annotations and are ready for FAIR data publication.
""")
