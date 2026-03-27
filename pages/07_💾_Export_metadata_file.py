import streamlit as st
import pandas as pd
import io
import json
import ast
import zipfile
import re

from ui.blocks import add_Soilwise_logo, add_Soilwise_contact_sidebar, add_clear_cache_button

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
        return list(parsed.values())
    return None


def _csvw_column_from_row(row):
    col = {"name": row['name']}

    if row.get('element'):
        col['titles'] = row['element']

    property_urls = _parse_element_uri(row.get('element_uri'))
    if property_urls:
        col['propertyUrl'] = property_urls

    if row.get('unit'):
        col['qudt:hasUnit'] = row['unit']
    if row.get('method'):
        col['method'] = row['method']
    if row.get('datatype'):
        if row['datatype'] == 'date':
            date_format = row.get('date format')
            col['datatype'] = {"base": "date", "format": date_format} if date_format else "date"
        else:
            col['datatype'] = row['datatype']
    if row.get('description'):
        col['dc:description'] = row['description']

    return col


def _build_csvw_table(table_df, url):
    return {
        "url": url,
        "tableSchema": {
            "columns": [_csvw_column_from_row(r) for _, r in table_df.iterrows()]
        }
    }


def build_csvw_frame(metadata_by_table, fallback_filename, filename_dict):
    context = [
        "http://www.w3.org/ns/csvw",
        {"qudt": "http://qudt.org/vocab/unit"}
    ]

    table_entries = []
    for table_key, metadata_df in metadata_by_table.items():
        table_url = filename_dict.get(table_key, fallback_filename)
        table_entries.append(_build_csvw_table(metadata_df, table_url))

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
    csvw_frame = build_csvw_frame(
        metadata_by_table=st.session_state[meta_key],
        fallback_filename=filename,
        filename_dict=filename_dict,
    )
    
    st.json(csvw_frame)
    download_bytes(json.dumps(csvw_frame, indent=2).encode('utf-8'), '_SoilWise.json', 'application/json')

st.divider()

st.markdown("### 💡 Tips")
st.info("""
- **CSV**: Best for sharing with colleagues or importing into spreadsheet applications
- **TableSchema**: Use when sharing data that follows Frictionless Data standards
- **CSVW**: Ideal for semantic web and linked data applications

All formats preserve your metadata annotations and are ready for FAIR data publication.
""")
