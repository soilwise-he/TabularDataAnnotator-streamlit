import streamlit as st
import pandas as pd
import io
import json
import ast
import zipfile
import re
import socket
import uuid
from urllib.parse import quote as _url_quote
import threading
import tempfile
import time
import webbrowser
from datetime import date
from functools import partial
from pathlib import Path
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import yaml

from ui.blocks import add_Soilwise_logo, add_Soilwise_contact_sidebar, add_clear_cache_button

from csvwlib import CSVWConverter

RDF_ROW_WARNING_THRESHOLD = 50000
RDF_LIMITED_ROW_COUNT = 1000

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
- **MCF YAML**: pygeometa Metadata Control File (ISO 19115 / OGC)
""")

# -------------  Helper functions ------------- 
def download_bytes(content: bytes, filename: str, mime: str = 'application/octet-stream'):
    st.download_button(label=f"📥 Download {filename}", data=content, file_name=filename, mime=mime)


def _safe_filename_component(value: str, fallback: str = "export") -> str:
    text = str(value) if value is not None else ""
    # Windows-invalid filename chars: <>:"/\\|?* plus control chars.
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", text)
    text = text.strip().strip(".")
    return text or fallback



def _csvw_column_from_row(row):
    col = {"name": row['name']}

    if row.get('concept'):
        col['titles'] = row['concept']


    if row.get('element'):
        col['type'] = row.get('element')

        if row.get('concept_uri'):
            col[row.get('element')] = row.get('concept_uri')

    if row.get('unit'):
        col['schema:unitCode'] = row['unit']
    if row.get('conversionMultiplier') is not None and row.get('conversionMultiplier') != '':
        try:
            col['qudt:conversionMultiplier'] = float(row['conversionMultiplier'])
        except (TypeError, ValueError):
            pass
    if row.get('conversionOffset') is not None and row.get('conversionOffset') != '':
        try:
            col['qudt:conversionOffset'] = float(row['conversionOffset'])
        except (TypeError, ValueError):
            pass
    
    if row.get('method') and not row['method']=="null":
        col['sosa:usedProcedure'] = row['method']
    if row.get('datatype'):
        if row['datatype']  in ['date', 'dateTime', 'time']:
            date_format = row.get('dateTime format')
            col['datatype'] = {"base": row['datatype'], "format": date_format} if date_format else row['datatype']
        else:
            col['datatype'] = row['datatype']
    if row.get('description'):
        col['dc:description'] = row['description']

    return col


def _build_csvw_table(table_df, url, foreign_keys=None):
    pk_rows = table_df[table_df.get("primary key", pd.Series(False, index=table_df.index)).astype(bool)]
    pk = pk_rows["name"].iloc[0] if not pk_rows.empty else None

    foi_rows = table_df[
        table_df.get("element", pd.Series("", index=table_df.index)).astype(str).str.strip()
        == "sosa:FeatureOfInterest"
    ]
    foi_column_name = foi_rows["name"].iloc[0] if not foi_rows.empty else None

    table_schema = {
        "columns": [_csvw_column_from_row(r) for _, r in table_df.iterrows()],
    }
    if pk:
        table_schema["primaryKey"] = pk
    
    if foreign_keys:
        table_schema["foreignKeys"] = foreign_keys

    # When available, identify each row resource by its Feature of Interest value.
    table_schema["aboutUrl"] = f"{{{foi_column_name}}}" if foi_column_name else r"{_row}"

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
        "name", "datatype", "description", "unit_symbol","unit_uri", "method", "element", "concept", "concept_uri", "dateTime format", "quantity kind_uri"
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
        # {"qudt": "http://qudt.org/vocab/unit"}
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
            metadata_url = f"http://127.0.0.1:{port}/{_url_quote(metadata_name)}"
            csv_url = f"http://127.0.0.1:{port}/{_url_quote(table_entries[0]['url'])}"

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


def _prepare_rdf_source_tables(metadata_by_table: dict, data_by_table: dict, row_limit: int | None = None) -> tuple[dict, dict, list]:
    prepared_tables = {}
    table_sizes = {}
    missing_tables = []

    for table_key, metadata_df in metadata_by_table.items():
        table_df = _resolve_data_table_for_metadata(table_key, metadata_df, data_by_table)
        if table_df is None:
            missing_tables.append(table_key)
            continue

        table_sizes[table_key] = len(table_df)
        if row_limit is not None and len(table_df) > row_limit:
            prepared_tables[table_key] = table_df.head(row_limit).copy()
        else:
            prepared_tables[table_key] = table_df

    return prepared_tables, table_sizes, missing_tables


def _generate_rdf_payloads(metadata_by_table: dict, data_by_table: dict) -> tuple[dict, list]:
    rdf_payloads = {}
    rdf_errors = []

    for table_key, metadata_df in metadata_by_table.items():
        safe_table_key = _safe_filename_component(table_key, fallback="table")
        one_table_metadata = {table_key: metadata_df}

        try:
            ttl_text = _generate_rdf_ttl_local(
                metadata_by_table=one_table_metadata,
                data_by_table=data_by_table,
            )
            rdf_payloads[f"{safe_table_key}.ttl"] = ttl_text.encode("utf-8")
        except Exception as e:
            rdf_errors.append(f"{table_key}: {e}")

    return rdf_payloads, rdf_errors


def _generate_rdf_ttl_sosa(metadata_by_table: dict, data_by_table: dict, base_url: str) -> str:
    """Like _generate_rdf_ttl_local but uses SOSA virtual-column CSVW for the metadata."""
    if not metadata_by_table:
        raise ValueError("No metadata available for RDF export.")

    with tempfile.TemporaryDirectory(prefix="csvw_sosa_export_") as tmpdir:
        tmp_path = Path(tmpdir)

        used_names: set = set()
        table_entries: list = []
        missing_data_tables: list = []
        filename_dict = st.session_state.get("filename_dict", {})

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
            # Use the SOSA-aware table builder with the local csv_name as URL
            table_entries.append(
                _build_csvw_sosa_table(table_key, metadata_df, csv_name, base_url)
            )

        if missing_data_tables:
            raise ValueError(
                "Missing tabular data for: " + ", ".join(map(str, missing_data_tables))
            )

        csvw_local = {"@context": _SOSA_CONTEXT, "tables": table_entries}

        metadata_name = "csvw-sosa-metadata.json"
        (tmp_path / metadata_name).write_text(
            json.dumps(csvw_local, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        port = _find_free_port()
        handler = partial(SimpleHTTPRequestHandler, directory=str(tmp_path))
        server = ThreadingHTTPServer(("127.0.0.1", port), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()

        try:
            metadata_url = f"http://127.0.0.1:{port}/{_url_quote(metadata_name)}"
            csv_url      = f"http://127.0.0.1:{port}/{_url_quote(table_entries[0]['url'])}"
            ttl = CSVWConverter.to_rdf(csv_url=csv_url, metadata_url=metadata_url, format="ttl")
            return ttl.decode("utf-8") if isinstance(ttl, bytes) else str(ttl)
        finally:
            server.shutdown()
            server.server_close()


def _generate_rdf_payloads_sosa(
    metadata_by_table: dict, data_by_table: dict, base_url: str
) -> tuple[dict, list]:
    rdf_payloads: dict = {}
    rdf_errors:   list = []

    for table_key, metadata_df in metadata_by_table.items():
        safe_table_key = _safe_filename_component(table_key, fallback="table")
        one_table_metadata = {table_key: metadata_df}

        try:
            ttl_text = _generate_rdf_ttl_sosa(
                metadata_by_table=one_table_metadata,
                data_by_table=data_by_table,
                base_url=base_url,
            )
            rdf_payloads[f"{safe_table_key}_sosa.ttl"] = ttl_text.encode("utf-8")
        except Exception as e:
            rdf_errors.append(f"{table_key}: {e}")

    return rdf_payloads, rdf_errors


# ==================== SOSA-aware CSVW ====================

_SOSA_CONTEXT = [
    "http://www.w3.org/ns/csvw",
    {
        "sosa": "http://www.w3.org/ns/sosa/",
        "qudt": "http://qudt.org/1.1/schema/qudt#",
        "dcterms": "http://purl.org/dc/terms/",
        "geo": "http://www.w3.org/2003/01/geo/wgs84_pos#",
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    },
]

_SPATIAL_ROLE_TO_PROPERTY = {
    "X": "geo:long",
    "Y": "geo:lat",
    "Z": "geo:alt",
    "WKT geometry": "geo:asWKT",
    "BBOX": "geo:asWKT",
}


def _sosa_virtual_cluster(
    obs_col: str,
    foi_col: str | None,
    base_url: str,
    unit_uri: str | None,
    concept_uri: str | None,
    method_uri: str | None,
    result_time_default: str | None,
) -> list:
    """Return the CSVW virtual-column cluster for one Observed Property column.

    Mirrors the pattern in soil-observation-data-encodings example 3:
      Observation → hasResult → QuantityValue
                  → hasFeatureOfInterest → FOI
                  → observedProperty → concept
                  → usedProcedure → method
                  → resultTime (default or empty)
      QuantityValue → rdf:type, qudt:hasUnit
    """
    if foi_col:
        foi_node = f"{base_url}{{{foi_col}}}"
        obs_node = f"{base_url}{{{foi_col}}}/{obs_col}"
        qv_node  = f"{base_url}{{{foi_col}}}/{obs_col}/QV"
    else:
        foi_node = f"{base_url}{{_row}}"
        obs_node = f"{base_url}{{_row}}/{obs_col}"
        qv_node  = f"{base_url}{{_row}}/{obs_col}/QV"

    vc: list = []

    # --- Observation node ---
    vc.append({"virtual": True, "propertyUrl": "rdf:type",
               "aboutUrl": obs_node, "valueUrl": "sosa:Observation"})
    vc.append({"virtual": True, "propertyUrl": "sosa:hasFeatureOfInterest",
               "aboutUrl": obs_node, "valueUrl": foi_node})
    if concept_uri:
        vc.append({"virtual": True, "propertyUrl": "sosa:observedProperty",
                   "aboutUrl": obs_node, "valueUrl": concept_uri})
    if method_uri:
        vc.append({"virtual": True, "propertyUrl": "sosa:usedProcedure",
                   "aboutUrl": obs_node, "valueUrl": method_uri})
    if result_time_default is not None:
        entry = {"virtual": True, "propertyUrl": "sosa:resultTime",
                 "aboutUrl": obs_node, "datatype": "dateTime"}
        if result_time_default:
            entry["default"] = result_time_default
        vc.append(entry)
    vc.append({"virtual": True, "propertyUrl": "sosa:hasResult",
               "aboutUrl": obs_node, "valueUrl": qv_node})

    # --- QuantityValue node ---
    vc.append({"virtual": True, "propertyUrl": "rdf:type",
               "aboutUrl": qv_node, "valueUrl": "qudt:QuantityValue"})
    if unit_uri:
        vc.append({"virtual": True, "propertyUrl": "qudt:hasUnit",
                   "aboutUrl": qv_node, "valueUrl": unit_uri})

    return vc


def _build_csvw_sosa_table(
    table_key: str,
    metadata_df: pd.DataFrame,
    url: str,
    base_url: str,
    foreign_keys: list | None = None,
) -> dict:
    """Build a SOSA-aware CSVW table with virtual columns.

    Real columns are classified by their bucket assignment (page 3):
      • FOI-ID              → propertyUrl dcterms:identifier on the FOI node
      • FOI - Spatial Info  → geo:lat / geo:long / etc. on the FOI node
      • FOI - Attribute     → concept_uri propertyUrl on the FOI node
      • Temporal (phenom.)  → sosa:phenomenonTime on the FOI node
      • Temporal (result)   → sosa:resultTime on the obs node (single obs col) or FOI node
      • Observed Property   → qudt:value on the QuantityValue node
                              + full virtual cluster per obs column
      • Unsorted            → flat passthrough (no propertyUrl)
    """
    if not base_url.endswith("/"):
        base_url += "/"

    meta_by_name: dict = {str(r["name"]): r for _, r in metadata_df.iterrows()}

    buckets         = st.session_state.get("column_buckets", {}).get(table_key, {})
    foi_cols        = buckets.get("Feature of Interest (FOI) - ID", [])
    obs_cols        = buckets.get("Observed Property", [])
    temporal_cols   = buckets.get("Temporal", [])
    spatial_cols    = buckets.get("FOI - Spatial Information", [])
    attr_cols       = buckets.get("FOI - Attribute", [])
    unsorted_cols   = buckets.get("Unsorted", [])

    foi_col = foi_cols[0] if foi_cols else None

    temporal_deepdive = st.session_state.get("temporal_deepdive", {}).get(table_key, {})
    spatial_types      = st.session_state.get("spatial_types", {}).get(table_key, {})
    spatial_fit_for_all        = st.session_state.get("spatial_fit_for_all", {}).get(table_key, {})

    # Unpack fit-for-all temporal ({"2024-01T00:00Z": "sosa:resultTime"} style)
    fit_all_time_value: str | None = None
    fit_all_time_role: str | None  = None
    fit_all = temporal_deepdive.get("__fit_for_all__")
    if isinstance(fit_all, dict):
        for val, role in fit_all.items():
            fit_all_time_value = str(val)
            fit_all_time_role  = str(role)
            break

    real_cols: list    = []
    virtual_cols: list = []

    # ------------------------------------------------------------------ #
    # 1. FOI-ID columns
    # ------------------------------------------------------------------ #
    for col in foi_cols:
        row = meta_by_name.get(col, {})
        c: dict = {"name": col, "propertyUrl": "dcterms:identifier"}
        if row.get("concept"):
            c["titles"] = str(row["concept"])
        if row.get("datatype"):
            c["datatype"] = str(row["datatype"])
        if row.get("description"):
            c["dc:description"] = str(row["description"])
        real_cols.append(c)

    # ------------------------------------------------------------------ #
    # 2. Spatial columns (real, on FOI node)
    # ------------------------------------------------------------------ #
    for col in spatial_cols:
        row  = meta_by_name.get(col, {})
        role = spatial_types.get(col)
        c = {
            "name": col,
            "propertyUrl": _SPATIAL_ROLE_TO_PROPERTY.get(role, "geo:location") if role else "geo:location",
        }
        if row.get("concept"):
            c["titles"] = str(row["concept"])
        if row.get("datatype"):
            c["datatype"] = str(row["datatype"])
        if row.get("description"):
            c["dc:description"] = str(row["description"])
        real_cols.append(c)

    # Spatial fit-for-all → virtual columns on FOI node
    for role, value in spatial_fit_for_all.items():
        if not value:
            continue
        foi_about = f"{base_url}{{{foi_col}}}" if foi_col else f"{base_url}{{_row}}"
        if "reference system" in role.lower():
            try:
                epsg = int(value)
                virtual_cols.append({
                    "virtual": True,
                    "propertyUrl": "dcterms:conformsTo",
                    "aboutUrl": foi_about,
                    "valueUrl": f"http://www.opengis.net/def/crs/EPSG/0/{epsg}",
                })
            except (ValueError, TypeError):
                pass
        elif role in _SPATIAL_ROLE_TO_PROPERTY:
            virtual_cols.append({
                "virtual": True,
                "propertyUrl": _SPATIAL_ROLE_TO_PROPERTY[role],
                "aboutUrl": foi_about,
                "default": str(value),
            })

    # ------------------------------------------------------------------ #
    # 3. Temporal columns — classify by role
    # ------------------------------------------------------------------ #
    result_time_cols:    list = []
    phenomenon_time_cols: list = []

    for col in temporal_cols:
        role = temporal_deepdive.get(col, fit_all_time_role or "sosa:phenomenonTime")
        if isinstance(role, dict):
            role = next(iter(role.values()), "sosa:phenomenonTime")
        row = meta_by_name.get(col, {})
        dt  = str(row.get("datatype") or "dateTime")
        fmt = str(row.get("dateTime format") or "")
        dt_val = {"base": dt, "format": fmt} if dt in ("date", "dateTime", "time") and fmt else dt

        if role == "sosa:resultTime":
            result_time_cols.append(col)
        else:
            phenomenon_time_cols.append(col)
            c = {"name": col, "propertyUrl": "sosa:phenomenonTime", "datatype": dt_val}
            if row.get("description"):
                c["dc:description"] = str(row["description"])
            real_cols.append(c)

    # ------------------------------------------------------------------ #
    # 4. Attribute columns (real, on FOI node)
    # ------------------------------------------------------------------ #
    for col in attr_cols:
        row = meta_by_name.get(col, {})
        c   = {"name": col}
        prop = str(row.get("concept_uri") or row.get("element") or "").strip()
        if prop:
            c["propertyUrl"] = prop
        if row.get("concept"):
            c["titles"] = str(row["concept"])
        if row.get("datatype"):
            c["datatype"] = str(row["datatype"])
        if row.get("description"):
            c["dc:description"] = str(row["description"])
        real_cols.append(c)

    # ------------------------------------------------------------------ #
    # 5. Observed Property columns — real column on QV node + virtual cluster
    # ------------------------------------------------------------------ #
    for obs_col in obs_cols:
        row = meta_by_name.get(obs_col, {})

        if foi_col:
            qv_about  = f"{base_url}{{{foi_col}}}/{obs_col}/QV"
            obs_about = f"{base_url}{{{foi_col}}}/{obs_col}"
        else:
            qv_about  = f"{base_url}{{_row}}/{obs_col}/QV"
            obs_about = f"{base_url}{{_row}}/{obs_col}"

        # Real column: value lives on the QuantityValue node
        c: dict = {
            "name": obs_col,
            "aboutUrl": qv_about,
            "propertyUrl": "qudt:value",
            "datatype": str(row.get("datatype") or "number"),
        }
        if row.get("concept"):
            c["titles"] = str(row["concept"])
        if row.get("description"):
            c["dc:description"] = str(row["description"])
        real_cols.append(c)

        # Derive annotation URIs
        unit_uri    = str(row.get("unit_uri") or row.get("unit") or "").strip() or None
        concept_uri = str(row.get("concept_uri") or "").strip() or None
        method_uri  = str(row.get("method") or "").strip()
        if method_uri in ("", "null", "None"):
            method_uri = None

        # resultTime default: fit-for-all value if role matches, else None
        result_time_default: str | None = None
        if fit_all_time_value and fit_all_time_role in ("sosa:resultTime", None):
            result_time_default = fit_all_time_value
        elif result_time_cols and len(obs_cols) > 1:
            # Multiple obs cols + real resultTime col → emit empty default
            # (CSVW virtual cols cannot reference cell values of other real cols)
            result_time_default = ""

        virtual_cols.extend(
            _sosa_virtual_cluster(
                obs_col=obs_col,
                foi_col=foi_col,
                base_url=base_url,
                unit_uri=unit_uri,
                concept_uri=concept_uri,
                method_uri=method_uri,
                result_time_default=result_time_default,
            )
        )

    # ------------------------------------------------------------------ #
    # 6. Result-time real column
    #    Single obs col: map time column aboutUrl to the obs node.
    #    Multiple obs cols: demote to phenomenonTime on FOI node.
    # ------------------------------------------------------------------ #
    for tc in result_time_cols:
        row = meta_by_name.get(tc, {})
        dt  = str(row.get("datatype") or "dateTime")
        fmt = str(row.get("dateTime format") or "")
        dt_val = {"base": dt, "format": fmt} if dt in ("date", "dateTime", "time") and fmt else dt

        if len(obs_cols) == 1 and foi_col:
            tc_about = f"{base_url}{{{foi_col}}}/{obs_cols[0]}"
            c = {"name": tc, "aboutUrl": tc_about,
                 "propertyUrl": "sosa:resultTime", "datatype": dt_val}
        else:
            c = {"name": tc, "propertyUrl": "sosa:phenomenonTime", "datatype": dt_val}

        if row.get("description"):
            c["dc:description"] = str(row["description"])
        real_cols.append(c)

    # ------------------------------------------------------------------ #
    # 7. Unsorted columns — flat passthrough
    # ------------------------------------------------------------------ #
    for col in unsorted_cols:
        row = meta_by_name.get(col, {})
        c   = {"name": col}
        if row.get("datatype"):
            c["datatype"] = str(row["datatype"])
        if row.get("description"):
            c["dc:description"] = str(row["description"])
        real_cols.append(c)

    # ------------------------------------------------------------------ #
    # 8. Optional: rdf:type virtual column for FOI node
    # ------------------------------------------------------------------ #
    if foi_col:
        foi_type = str(meta_by_name.get(foi_col, {}).get("concept_uri") or "").strip()
        if foi_type:
            virtual_cols.insert(0, {
                "virtual": True,
                "propertyUrl": "rdf:type",
                "aboutUrl": f"{base_url}{{{foi_col}}}",
                "valueUrl": foi_type,
            })

    # ------------------------------------------------------------------ #
    # Assemble tableSchema
    # ------------------------------------------------------------------ #
    table_schema: dict = {"columns": real_cols + virtual_cols}
    if foi_col:
        table_schema["aboutUrl"]   = f"{base_url}{{{foi_col}}}"
        table_schema["primaryKey"] = foi_col
    if foreign_keys:
        table_schema["foreignKeys"] = foreign_keys

    return {"url": url, "tableSchema": table_schema}


def build_csvw_sosa_frame(
    metadata_by_table: dict,
    fallback_filename: str,
    filename_dict: dict,
    base_url: str,
    relationships_summary_df=None,
) -> dict:
    """Build a full CSVW document with SOSA virtual columns for all tables."""
    table_entries = []
    for table_key, mdf in metadata_by_table.items():
        table_url   = filename_dict.get(table_key, fallback_filename)
        foreign_keys = _build_foreign_keys(table_key, relationships_summary_df, filename_dict) \
                       if relationships_summary_df is not None else []
        table_entries.append(
            _build_csvw_sosa_table(table_key, mdf, table_url, base_url, foreign_keys or None)
        )

    if len(table_entries) == 1:
        return {
            "@context": _SOSA_CONTEXT,
            "url": table_entries[0]["url"],
            "tableSchema": table_entries[0]["tableSchema"],
        }
    return {"@context": _SOSA_CONTEXT, "tables": table_entries}


# ==================== MCF YAML ====================

_MCF_TYPE_MAP = {
    "integer": "integer", "int": "integer",
    "number": "number", "float": "number", "double": "number",
    "decimal": "number", "numeric": "number",
    "boolean": "boolean",
}


def _build_mcf_dict(table_key: str, metadata_df: pd.DataFrame) -> dict:
    """Build a pygeometa MCF 2.0 dict for one table from session state."""
    today = date.today().isoformat()

    # --- discovery / context info ---
    zenodo_meta = st.session_state.get("zenodo_context_metadata", {})
    title = str(zenodo_meta.get("title") or table_key)
    abstract = str(zenodo_meta.get("description") or "")

    # --- spatial extent ---
    spatial_fit_for_all = st.session_state.get("spatial_fit_for_all", {}).get(table_key, {})
    spatial_col_roles = {
        v for k, v in st.session_state.get("spatial_types", {}).get(table_key, {}).items()
        if k != "__fit_for_all__" and v
    }
    has_spatial = bool(spatial_col_roles) or bool(spatial_fit_for_all)

    raw_crs = spatial_fit_for_all.get("XY reference system", 4326)
    try:
        crs = int(raw_crs) if raw_crs else 4326
    except (ValueError, TypeError):
        crs = 4326

    bbox = [-180, -90, 180, 90]  # default world bbox
    bbox_raw = spatial_fit_for_all.get("BBOX")
    if bbox_raw:
        # 1. Explicit BBOX fit-for-all string
        try:
            parsed = [float(x) for x in str(bbox_raw).split(",")]
            if len(parsed) == 4:
                bbox = parsed
        except (ValueError, TypeError):
            pass
    else:
        # 2. Combine column data (min/max) with fit-for-all point values per axis.
        #    Column data takes priority; fit-for-all fills in any missing axis.
        _spatial_deepdive = st.session_state.get("spatial_deepdive", {}).get(table_key, {})
        _x_col = next((c for c, r in _spatial_deepdive.items() if r == "X" and c != "__fit_for_all__"), None)
        _y_col = next((c for c, r in _spatial_deepdive.items() if r == "Y" and c != "__fit_for_all__"), None)
        _full_df = st.session_state.get("tabular_data_dict", {}).get(table_key) if (_x_col or _y_col) else None

        # Resolve X axis
        _xmin, _xmax = None, None
        if _x_col and _full_df is not None and _x_col in _full_df.columns:
            try:
                _xs = pd.to_numeric(_full_df[_x_col], errors="coerce").dropna()
                if not _xs.empty:
                    _xmin, _xmax = float(_xs.min()), float(_xs.max())
            except (ValueError, TypeError):
                pass
        if _xmin is None and spatial_fit_for_all.get("X") is not None:
            try:
                _xmin = _xmax = float(spatial_fit_for_all["X"])
            except (ValueError, TypeError):
                pass

        # Resolve Y axis
        _ymin, _ymax = None, None
        if _y_col and _full_df is not None and _y_col in _full_df.columns:
            try:
                _ys = pd.to_numeric(_full_df[_y_col], errors="coerce").dropna()
                if not _ys.empty:
                    _ymin, _ymax = float(_ys.min()), float(_ys.max())
            except (ValueError, TypeError):
                pass
        if _ymin is None and spatial_fit_for_all.get("Y") is not None:
            try:
                _ymin = _ymax = float(spatial_fit_for_all["Y"])
            except (ValueError, TypeError):
                pass

        if _xmin is not None and _ymin is not None:
            bbox = [_xmin, _ymin, _xmax, _ymax]

    # --- temporal extent ---
    temporal_cols = (
        st.session_state.get("column_buckets", {})
        .get(table_key, {})
        .get("Temporal", [])
    )

    # --- file URL / distribution ---
    filename_dict = st.session_state.get("filename_dict", {})
    file_url = str(filename_dict.get(table_key, f"{_safe_filename_component(table_key)}.csv"))

    # #TODO: take keys from zenodo (?)
    # # --- observed property column names as keywords ---
    # obs_cols = (
    #     st.session_state.get("column_buckets", {})
    #     .get(table_key, {})
    #     .get("Observed Property", [])
    # )

    # --- content_info attributes from metadata_df ---
    attributes = []
    for _, row in metadata_df.iterrows():
        attr: dict = {"name": str(row.get("name", ""))}
        concept = str(row.get("concept") or row.get("name", "")).strip()
        description = str(row.get("description") or "").strip()
        usedProcedure = str(row.get("method") or "").strip()
        inforamationRole = str(row.get("element") or "").strip()

        if inforamationRole:
            attr["informationRole"] = inforamationRole
        if concept:
            attr["title"] = {"en": concept}
        if description:
            attr["abstract"] = {"en": description}
        if usedProcedure:
            attr["usedProcedure"] = usedProcedure
        raw_type = str(row.get("datatype") or "").lower().strip()

        # TODO: does this realy needs to be mapped?
        if raw_type:
            attr["type"] = _MCF_TYPE_MAP.get(raw_type, "string")
        unit = str(row.get("unit_uri") or row.get("unit") or "").strip()
        if unit:
            attr["units"] = unit
        unit_uri = str(row.get("concept_uri") or "").strip()
        if unit_uri:
            attr["url"] = unit_uri
        attributes.append(attr)

    # --- assemble MCF dict ---
    identification: dict = {
        "title": title,
        "abstract": abstract or "No description available.",
        "url": file_url,
        "status": "",
        "rights": "",
        "extents": {
            "spatial": [{"bbox": bbox, "crs": crs}],
        },
    }


    _tex = st.session_state.get("temporal_extent", {}).get(table_key, {})
    if temporal_cols or _tex.get("begin") or _tex.get("end"):
        identification["extents"]["temporal"] = [{"begin": _tex.get("begin") or "", "end": _tex.get("end") or ""}]

    # if obs_cols:
    #     identification["keywords"] = {
    #         "observed_properties": {
    #             "keywords": {"en": obs_cols},
    #             "keywords_type": "theme",
    #         }
    #     }

    # TODO: Detection
    spatial_section: dict = {"datatype": "textTable"}
    if has_spatial:
        spatial_section["geomtype"] = "point"

    mcf: dict = {
        "mcf": {"version": "2.0"},
        "metadata": {
            "identifier": str(uuid.uuid4()),
            "language": "en",
            "charset": "utf8",
            "hierarchylevel": "dataset",
            "dates": {"creation": today},
        },
        "spatial": spatial_section,
        "identification": identification,
        # "contact": {
        #     "pointOfContact": {
        #         "organization": "",
        #         "url": "",
        #         "individualname": "",
        #         "positionname": "",
        #         "phone": "",
        #         "fax": "",
        #         "address": "",
        #         "city": "",
        #         "administrativearea": "",
        #         "postalcode": "",
        #         "country": "",
        #         "email": "",
        #     }
        # },
        # "distribution": {
        #     _safe_filename_component(table_key, fallback="dataset"): {
        #         "url": file_url,
        #         "type": "WWW:LINK",
        #         "name": table_key,
        #         "description": title,
        #         "function": "download",
        #     }
        # },
        "content_info": {
            "type": "feature_catalogue",
            "attributes": attributes,
        },
    }
    return mcf

# ==================== START UI ====================

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
        c_1, c_2 = st.columns([1,1])
        c_2.json(st.session_state["spatial_deepdive"][table_key])
        c_1.dataframe(st.session_state["temporal_deepdive"][table_key])

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

_sosa_mode = st.checkbox(
    "Enhanced SOSA CSVW export",
    value=False,
    help=(
        "Generates SOSA-aligned virtual columns: each Observed Property column is encoded "
        "as a full `sosa:Observation → qudt:QuantityValue` sub-graph with unit, observed "
        "property URI, and procedure links — matching the soil-observation-data-encodings "
        "example 3 pattern. Requires bucket assignments from the Column Sorting page."
    ),
)

if _sosa_mode:
    # Derive a sensible default base URL from the first known file URL
    _fl = st.session_state.get("filename_dict", {})
    _default_base = re.sub(r"[^/\\]+$", "", str(next(iter(_fl.values()), ""))) if _fl else ""
    _base_url = st.text_input(
        "Base URL for resource identifiers",
        value=_default_base,
        placeholder="https://example.org/dataset/",
        help=(
            "URI prefix for all Observation / FOI / QuantityValue nodes.  \n"
            "Example: `https://soilwise.example.com/mydata/` produces nodes like  \n"
            "`https://soilwise.example.com/mydata/{ID}/{column}/QV`"
        ),
    )

if st.button("Generate CSVW JSON", key="csvw_button"):
    filename_dict = st.session_state.get("filename_dict", {})
    _rels = st.session_state.get("table_relationships_summary_df", pd.DataFrame())
    _rels = _rels if not _rels.empty else None

    if _sosa_mode:
        csvw_frame   = build_csvw_sosa_frame(
            metadata_by_table=st.session_state[meta_key],
            fallback_filename=filename,
            filename_dict=filename_dict,
            base_url=_base_url,
            relationships_summary_df=_rels,
        )
        out_filename = "_SoilWise_sosa.json"
    else:
        csvw_frame   = build_csvw_frame(
            metadata_by_table=st.session_state[meta_key],
            fallback_filename=filename,
            filename_dict=filename_dict,
            relationships_summary_df=_rels,
        )
        out_filename = "_SoilWise.json"

    st.json(csvw_frame)
    download_bytes(json.dumps(csvw_frame, indent=2).encode("utf-8"), out_filename, "application/json")

st.divider()

# ==================== RDF ====================
st.markdown("#### 4️⃣ RDF")
st.caption("Resource Description Framework format - [Learn more](https://www.w3.org/RDF/)")

_rdf_sosa_mode = st.checkbox(
    "Enhanced SOSA RDF export",
    value=False,
    key="rdf_sosa_mode",
    help=(
        "Generates RDF using the SOSA virtual-column CSVW as input, producing a full "
        "`sosa:Observation → qudt:QuantityValue` graph. "
        "Requires the same bucket assignments as the Enhanced SOSA CSVW export."
    ),
)

if _rdf_sosa_mode:
    _fl_rdf = st.session_state.get("filename_dict", {})
    _default_base_rdf = re.sub(r"[^/\\]+$", "", str(next(iter(_fl_rdf.values()), ""))) if _fl_rdf else ""
    _rdf_base_url = st.text_input(
        "Base URL for resource identifiers",
        value=_default_base_rdf,
        placeholder="https://example.org/dataset/",
        key="rdf_base_url",
        help=(
            "URI prefix used to construct Observation / FOI / QuantityValue node URIs.  "
            "Same value as used in the Enhanced SOSA CSVW export above."
        ),
    )

tabular_data_dict = st.session_state.get("tabular_data_dict", {})
rdf_table_sizes = {
    table_key: len(table_df)
    for table_key, table_df in tabular_data_dict.items()
    if isinstance(table_df, pd.DataFrame)
}

large_rdf_tables = {
    table_key: row_count
    for table_key, row_count in rdf_table_sizes.items()
    if row_count >= RDF_ROW_WARNING_THRESHOLD
}
has_large_rdf_tables = bool(large_rdf_tables)

if large_rdf_tables:
    large_tables_text = ", ".join(
        f"{table_key} ({row_count:,} rows)"
        for table_key, row_count in large_rdf_tables.items()
    )
    st.warning(
        "Large source tables detected. RDF generation will automatically use a limited dataset for speed. "
        f"Affected tables: {large_tables_text}."
    )

rdf_auto_col, rdf_force_col = st.columns(2)
run_auto_rdf = rdf_auto_col.button("Generate RDF", key="rdf_button")
if large_rdf_tables:
    run_force_full_rdf = rdf_force_col.button(
        "Generate RDF (full dataset)",
        key="rdf_button_force_full",
        help="Bypass automatic limiting and use full tables.",
    )
else:
    run_force_full_rdf = False

if run_auto_rdf or run_force_full_rdf:
    row_limit = None
    if run_auto_rdf and has_large_rdf_tables:
        row_limit = RDF_LIMITED_ROW_COUNT

    rdf_input_tables, _, rdf_missing_tables = _prepare_rdf_source_tables(
        metadata_by_table=st.session_state[meta_key],
        data_by_table=tabular_data_dict,
        row_limit=row_limit,
    )

    if rdf_missing_tables:
        st.info(
            "RDF export is missing source data for: "
            + ", ".join(map(str, rdf_missing_tables))
        )

    if row_limit is not None:
        st.caption(
            f"Generating RDF from at most {RDF_LIMITED_ROW_COUNT:,} rows per table for faster preview."
        )

    if _rdf_sosa_mode:
        rdf_payloads, rdf_errors = _generate_rdf_payloads_sosa(
            metadata_by_table=st.session_state[meta_key],
            data_by_table=rdf_input_tables,
            base_url=_rdf_base_url,
        )
        zip_filename = "metadata_rdf_sosa_exports.zip"
    else:
        rdf_payloads, rdf_errors = _generate_rdf_payloads(
            metadata_by_table=st.session_state[meta_key],
            data_by_table=rdf_input_tables,
        )
        zip_filename = "metadata_rdf_exports.zip"

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
            download_bytes(zip_buf.getvalue(), zip_filename, 'application/zip')

        with column_individual:
            with st.expander("Download individual RDF files", expanded=False):
                for export_filename, export_bytes in rdf_payloads.items():
                    download_bytes(export_bytes, export_filename, 'text/turtle')
    else:
        for export_filename, export_bytes in rdf_payloads.items():
            st.code(export_bytes.decode("utf-8"), language="turtle")
            download_bytes(export_bytes, export_filename, 'text/turtle')


st.divider()

# ==================== MCF YAML EXPORT ====================
st.markdown("#### 5️⃣ MCF YAML")
st.caption(
    "pygeometa Metadata Control File format - "
    "[Learn more](https://geopython.github.io/pygeometa/reference/mcf/)"
)

if st.button("Generate MCF YAML", key="mcf_button"):
    mcf_payloads = {}
    mcf_errors = []
    for table_key, tbl_metadata_df in st.session_state[meta_key].items():
        safe_key = _safe_filename_component(table_key, fallback="table")
        try:
            mcf_dict = _build_mcf_dict(table_key, tbl_metadata_df)
            mcf_yaml = yaml.dump(
                mcf_dict,
                allow_unicode=True,
                default_flow_style=False,
                sort_keys=False,
            )
            mcf_payloads[f"{safe_key}.mcf.yml"] = mcf_yaml.encode("utf-8")
        except Exception as e:
            mcf_errors.append(f"{table_key}: {e}")

    if mcf_errors:
        st.error("Failed to generate MCF for:\n\n" + "\n".join(f"- {err}" for err in mcf_errors))

    if not mcf_payloads:
        st.warning("No MCF output generated.")
    elif len(mcf_payloads) > 1:
        column_zip, column_individual = st.columns([2, 5])
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for export_filename, export_bytes in mcf_payloads.items():
                zf.writestr(export_filename, export_bytes)
        with column_zip:
            download_bytes(zip_buf.getvalue(), "metadata_mcf_exports.zip", "application/zip")
        with column_individual:
            with st.expander("Download individual MCF files", expanded=False):
                for export_filename, export_bytes in mcf_payloads.items():
                    download_bytes(export_bytes, export_filename, "text/yaml")
        # Preview first table
        first_key = next(iter(mcf_payloads))
        st.code(mcf_payloads[first_key].decode("utf-8"), language="yaml")
    else:
        for export_filename, export_bytes in mcf_payloads.items():
            st.code(export_bytes.decode("utf-8"), language="yaml")
            download_bytes(export_bytes, export_filename, "text/yaml")


st.markdown("### 💡 Tips")
st.info("""
- **CSV**: Best for sharing with colleagues or importing into spreadsheet applications
- **TableSchema**: Use when sharing data that follows Frictionless Data standards
- **CSVW**: Ideal for semantic web and linked data applications
- **MCF YAML**: pygeometa Metadata Control File for ISO 19115 / OGC metadata catalogues

All formats preserve your metadata annotations and are ready for FAIR data publication.
""")
