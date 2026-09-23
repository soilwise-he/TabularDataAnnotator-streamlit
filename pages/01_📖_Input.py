import ast
import streamlit as st
import pandas as pd
import io
import json
from typing import Any, Optional, Sequence, List, Dict
from types import SimpleNamespace
import os
import re
import hashlib
import unicodedata
from dateutil import parser as dateutil_parser

from csvwlib.utils.NumericUtils import NumericUtils
from csvwlib.utils.datatypeutils import is_compatible_with_datatype

import requests

import csv
from ui.blocks import add_Soilwise_contact_sidebar,add_Soilwise_logo, add_clear_cache_button
from util.metadata import (
    METADATA_EXPORT_FILENAME_SUFFIX,
    apply_new_metadata_info,
    build_filename_match_tokens,
    normalize_metadata_columns,
)

import html
from bs4 import BeautifulSoup

from urllib.parse import urlparse, quote
import zipfile


add_Soilwise_logo()
add_clear_cache_button(key_prefix="input_page")

st.set_page_config(page_title="Tabular Soil Data Annotation", layout="wide")

# -------------------- Helper data and functions --------------------

# !! in UoM procedure, it's hardcoded filtered on "numeric" types
# https://www.w3.org/TR/tabular-data-primer/?ref=stevenfirth.com#datatypes
DATA_TYPE_OPTIONS = ['codelist',
                     'anyURI', 'base64Binary', 'boolean', 'date',
                     'dateTime', 'dateTimeStamp', 'decimal',
                     'integer', 'long', 'int', 'short', 'byte',
                     'nonNegativeInteger', 'positiveInteger', 'unsignedLong',
                     'unsignedInt', 'unsignedShort', 'unsignedByte',
                     'nonPositiveInteger', 'negativeInteger', 'double',
                     'number', 'duration', 'dayTimeDuration', 'yearMonthDuration',
                     'float', 'gDay', 'gMonth', 'gMonthDay', 'gYear', 'gYearMonth',
                     'hexBinary', 'QName', 'string', 'normalizedString', 'token',
                     'language', 'Name', 'NMTOKEN', 'time', 'xml', 'html', 'json']

SESSION_RESET_KEYS = [
    "metadata_df",
    "source_df_id",
    "metadata_df_id",
    "tabular_data_dict",
    "filename_dict",
    "uploaded_filename",
    "remote_context_files_url",
    "remote_context_metadata",
    "remote_context_files_from_zip",
    "zenodo_loaded",
    "primary_keys",
    "context_files",
    "vocab_oversized_matching_results",
    "vocab_row_selection",
    "vocab_row_selection_status",
    "df_selection_keywords",
    "_linked_site_df",
    "_linked_obs_df",
    "_linked_obs_filename",
    "imported_metadata_by_filename",
    "imported_metadata_by_filename_remote",

]


SESSION_RESET_PREFIXES = [
    "editor_",
    "discard_table_",
    "move_to_context_",
]

GITHUB_INGEST_SCHEMA_VERSION = "v2-path-keys"

IGNORED_REMOTE_CSV_FILENAMES = {
    "table_linking_summary.csv",
    "fit_for_all_temporal_spatial.csv",
}


def _clear_dependent_session_state_for_new_input():
    preserve_zenodo_cache_key = st.session_state.get("_active_zenodo_cache_key")
    preserve_github_cache_key = st.session_state.get("_active_github_cache_key")

    for key in SESSION_RESET_KEYS:
        st.session_state.pop(key, None)

    for key in list(st.session_state.keys()):
        if any(key.startswith(prefix) for prefix in SESSION_RESET_PREFIXES):
            st.session_state.pop(key, None)

    for key in list(st.session_state.keys()):
        if key.startswith("_zenodo_ingest_cache_") and key != preserve_zenodo_cache_key:
            st.session_state.pop(key, None)

    for key in list(st.session_state.keys()):
        if key.startswith("_github_ingest_cache_") and key != preserve_github_cache_key:
            st.session_state.pop(key, None)

    # # Keep non-table context files (e.g. user-provided docs) but remove stale table-derived context.
    # if "context_files" in st.session_state:
    #     st.session_state["context_files"] = [
    #         f for f in st.session_state["context_files"]
    #         if getattr(f, "source", None) != "table"
    #     ]


def _build_input_signature(mode: str, upload_tokens: list[str], url_input: str = "") -> str:
    signature_raw = "|".join([mode, *(sorted(upload_tokens)), (url_input or "")])
    return hashlib.sha256(signature_raw.encode("utf-8")).hexdigest()

def filename_from_url(url: str) -> str:
    filename = urlparse(url).path.split('/')[-1]
    if filename=="content":
        url_strip = url[:-len("/content")]
        filename=filename_from_url(url_strip)
    return filename


def normalize_github_file_url(url: str) -> str:
    """Normalize GitHub file URLs to directly downloadable raw URLs."""
    parsed = urlparse(url.strip())
    host = parsed.netloc.lower()

    if host == "raw.githubusercontent.com":
        return url.strip()

    if host in {"github.com", "www.github.com"}:
        parts = parsed.path.strip("/").split("/")
        # Expected: /owner/repo/blob/branch/path/to/file.csv
        if len(parts) >= 5 and parts[2] == "blob":
            owner, repo, _, branch = parts[:4]
            file_path = "/".join(parts[4:])
            return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{file_path}"

    raise ValueError(
        "Invalid GitHub file URL. Use a raw.githubusercontent.com URL or a github.com/.../blob/... file URL."
    )


def normalize_github_input_url(url: str) -> str:
    """Validate GitHub input URL for file or folder/repository usage."""
    parsed = urlparse(url.strip())
    host = parsed.netloc.lower()
    parts = parsed.path.strip("/").split("/") if parsed.path.strip("/") else []

    if host == "raw.githubusercontent.com":
        if len(parts) >= 5:
            return url.strip()
        raise ValueError("Invalid raw GitHub URL.")

    if host in {"github.com", "www.github.com"}:
        if len(parts) >= 2:
            # Supports repo root, tree folder URLs, and blob file URLs.
            return f"https://github.com/{'/'.join(parts)}"

    raise ValueError(
        "Invalid GitHub URL. Use a repository/folder URL, raw.githubusercontent.com URL, or github.com/.../blob/... file URL."
    )


@st.cache_data()
def get_default_branch_from_github(owner: str, repo: str) -> str:
    request_url = f"https://api.github.com/repos/{owner}/{repo}"
    response = requests.get(request_url, headers={"Accept": "application/vnd.github+json"})
    response.raise_for_status()
    response_json = response.json()
    default_branch = response_json.get("default_branch")
    if not default_branch:
        raise ValueError(f"Could not determine default branch for {owner}/{repo}.")
    return default_branch


@st.cache_data()
def resolve_github_input_to_file_urls(
    url: str,
    extensions: Optional[Sequence[str]] = None,
    recursive: bool = True,
) -> list[str]:
    """Resolve GitHub file/folder/repo URL(s) to raw downloadable file URLs."""
    parsed = urlparse(url.strip())
    host = parsed.netloc.lower()
    exts = None
    if extensions:
        exts = tuple((e if e.startswith(".") else f".{e}").lower() for e in extensions)

    if host == "raw.githubusercontent.com":
        raw_url = url.strip()
        if exts and not raw_url.lower().endswith(exts):
            return []
        return [raw_url]

    if host not in {"github.com", "www.github.com"}:
        raise ValueError("Only github.com and raw.githubusercontent.com URLs are supported.")

    parts = parsed.path.strip("/").split("/")
    if len(parts) < 2:
        raise ValueError("GitHub URL must include owner and repository.")

    owner = parts[0]
    repo = parts[1]

    # File URL: /owner/repo/blob/branch/path/to/file
    if len(parts) >= 5 and parts[2] == "blob":
        raw_url = normalize_github_file_url(url)
        if exts and not raw_url.lower().endswith(exts):
            return []
        return [raw_url]

    # Folder URL: /owner/repo/tree/branch/path/to/folder
    if len(parts) >= 4 and parts[2] == "tree":
        branch = parts[3]
        folder_path = "/".join(parts[4:])
    # Repository root URL: /owner/repo
    elif len(parts) >= 2:
        branch = get_default_branch_from_github(owner, repo)
        folder_path = ""
    else:
        raise ValueError("Unsupported GitHub URL format.")

    resolved_files: list[str] = []
    stack = [folder_path]

    while stack:
        current_path = stack.pop()
        api_url = f"https://api.github.com/repos/{owner}/{repo}/contents"
        if current_path:
            api_url = f"{api_url}/{quote(current_path, safe='/')}"

        response = requests.get(
            api_url,
            headers={"Accept": "application/vnd.github+json"},
            params={"ref": branch},
        )
        response.raise_for_status()
        data = response.json()

        entries = data if isinstance(data, list) else [data]
        for entry in entries:
            entry_type = entry.get("type")
            if entry_type == "file":
                download_url = entry.get("download_url")
                entry_path = entry.get("path", "")
                if not download_url:
                    continue
                if exts and not entry_path.lower().endswith(exts):
                    continue
                resolved_files.append(download_url)
            elif entry_type == "dir" and recursive:
                entry_path = entry.get("path")
                if entry_path:
                    stack.append(entry_path)

    return resolved_files

def detect_csvw_datatype_from_series(s: pd.Series, sample_size: int = 200) -> str:
    """Detect the most specific CSVW datatype for a pandas Series using csvwlib.

    Probes each sampled value with csvwlib's NumericUtils.is_numeric (more robust
    than float() — handles E notation, %, ‰) and is_compatible_with_datatype for
    structural checks, then falls back to pandas for date/dateTime distinction.

    Returns one of: 'integer', 'float', 'date', 'dateTime', 'time', 'boolean', 'json', 'string'.
    """
    series = s.dropna().astype(str).str.strip()
    if len(series) == 0:
        return "string"
    series = series.head(sample_size)

    counts = {"integer": 0, "float": 0, "dateTime": 0, "date": 0, "time": 0, "boolean": 0, "json": 0}
    total = 0

    for val in series:
        if val == "":
            continue
        total += 1

        # --- numeric (csvwlib NumericUtils, supports E, %, ‰, +/-) ---
        v = val.replace(',', '.')

        if NumericUtils.is_numeric(v):
            try:
                f = float(v)
                if f == int(f):
                    counts["integer"] += 1
                else:
                    counts["float"] += 1
            except (ValueError, OverflowError):
                counts["float"] += 1
            continue

        # --- boolean (csvwlib is_compatible_with_datatype) ---
        if is_compatible_with_datatype(val, "boolean"):
            # only treat as boolean if the value is literally true/false/1/0
            if val.lower() in ("true", "false", "1", "0"):
                counts["boolean"] += 1
                continue

        # --- json (object or array) ---

        if val.strip().startswith(('{', '[')):
            try:
                json.loads(val)
                counts["json"] += 1
                continue
            except (ValueError, json.JSONDecodeError):
                pass

        # --- date / dateTime / time via pandas (csvwlib delegates to dateutil anyway) ---
        if any(sep in val for sep in ['/', '-', '.', ':']):
            try:
                parsed = pd.to_datetime(val)
                # distinguish dateTime (has non-midnight time) from plain date
                if parsed.hour != 0 or parsed.minute != 0 or parsed.second != 0:
                    counts["dateTime"] += 1
                else:
                    counts["date"] += 1
                continue
            except Exception:
                pass
            # time-only strings  e.g. "14:30:00"
            try:
                pd.to_datetime(val, format="%H:%M:%S")
                counts["time"] += 1
                continue
            except Exception:
                pass

    if total == 0:
        return "string"

    THRESHOLD = 0.8
    # Order: most specific / least ambiguous first

    for dt in ("integer", "float", "boolean", "dateTime", "date", "time", "json"):
        if counts[dt] / total >= THRESHOLD:
            if dt == "integer" and counts["float"] > 0:
                return "float"
            return dt
    return "string"


# Comprehensive date/datetime format candidates (strptime tokens).
FORMAT_CANDIDATES_DATETIME = [
    # ---- Date-only: numeric ----
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%Y/%m/%d",
    "%m/%d/%Y",
    "%m-%d-%Y",
    "%d.%m.%Y",
    "%Y.%m.%d",
    # Year-month
    "%Y-%m",
    "%Y/%m",
    "%m-%Y",
    "%m/%Y",
    # ---- Date-only: month names ----
    "%d %b %Y",
    "%d %B %Y",
    "%b %d, %Y",
    "%B %d, %Y",
    "%d-%b-%Y",
    "%d/%b/%Y",
    "%b %Y",
    "%B %Y",
    # ---- DateTime: space separator ----
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%d-%m-%Y %H:%M:%S",
    "%d-%m-%Y %H:%M",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d %H:%M",
    "%d.%m.%Y %H:%M:%S",
    "%d.%m.%Y %H:%M",
    # ---- DateTime: ISO 8601 ----
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S.%f",
    # ---- DateTime: month names + time ----
    "%d %b %Y %H:%M:%S",
    "%d %B %Y %H:%M:%S",
    "%d %b %Y %H:%M",
    "%d %B %Y %H:%M",
    "%b %d, %Y %H:%M:%S",
    "%B %d, %Y %H:%M:%S",
    "%b %d, %Y %H:%M",
    "%B %d, %Y %H:%M",
    # ---- Time-only ----
    "%H:%M:%S",
    "%H:%M",
    "%H:%M:%S.%f",
]

# Non-English month name → English for broader locale support.
_MONTH_TRANSLATION: Dict[str, str] = {
    # German
    "januar": "January", "jänner": "January", "februar": "February",
    "feber": "February", "märz": "March", "mai": "May",
    "juni": "June", "juli": "July", "oktober": "October",
    "dezember": "December",
    "mär": "Mar", "okt": "Oct", "dez": "Dec",
    # French
    "janvier": "January", "février": "February", "mars": "March",
    "avril": "April", "juin": "June", "juillet": "July",
    "août": "August", "septembre": "September", "octobre": "October",
    "novembre": "November", "décembre": "December",
    "janv": "Jan", "févr": "Feb", "avr": "Apr",
    "juil": "Jul", "sept": "Sep", "déc": "Dec",
    # Spanish
    "enero": "January", "febrero": "February", "marzo": "March",
    "mayo": "May", "junio": "June", "julio": "July",
    "agosto": "August", "septiembre": "September", "octubre": "October",
    "noviembre": "November", "diciembre": "December",
    # Italian
    "gennaio": "January", "febbraio": "February", "aprile": "April",
    "maggio": "May", "giugno": "June", "luglio": "July",
    "settembre": "September", "ottobre": "October",
    # Dutch
    "januari": "January", "februari": "February", "maart": "March",
    "mei": "May",
    # Portuguese
    "fevereiro": "February", "março": "March", "maio": "May",
    "junho": "June", "julho": "July", "setembro": "September",
    "outubro": "October", "dezembro": "December",
}

_MONTH_PATTERN = re.compile(
    "|".join(re.escape(k) for k in sorted(_MONTH_TRANSLATION, key=len, reverse=True)),
    re.IGNORECASE,
)


def _normalize_datetime_string(val: str) -> str:
    """Normalize a datetime string for broader format matching.

    Handles fullwidth digits, CJK date separators (年月日時分秒),
    and non-English month names.
    """
    val = unicodedata.normalize("NFKC", val)
    # CJK date / time separators → standard delimiters
    val = val.replace("年", "-").replace("月", "-").replace("日", "")
    val = val.replace("時", ":").replace("分", ":").replace("秒", "")
    val = val.strip(" -:T")
    # Translate non-English month names
    val = _MONTH_PATTERN.sub(lambda m: _MONTH_TRANSLATION[m.group(0).lower()], val)
    return val.strip()


def _strip_tz_suffix(val: str) -> str:
    """Remove trailing timezone indicators for strptime matching."""
    val = val.rstrip()
    if val.endswith("Z") or val.endswith("z"):
        return val[:-1]
    if re.search(r'[+-]\d{2}:\d{2}$', val):
        return val[:-6]
    if re.search(r'[+-]\d{4}$', val):
        return val[:-5]
    return val

@st.cache_data()
def detect_date_format_from_series(s: pd.Series, sample_size: int = 200) -> str:
    """Infer the dominant date/datetime format in a pandas Series.

    Returns a strptime format token (e.g. %Y-%m-%d, %d/%m/%Y %H:%M:%S).
    Handles date-only, datetime, time-only, and Unicode-formatted strings
    (CJK separators, fullwidth digits, non-English month names).
    If no consistent format is detected, returns an empty string.
    """
    series = s.dropna().astype(str).str.strip()
    if len(series) == 0:
        return ""
    series = series.head(sample_size)

    counts: Dict[str, int] = {fmt: 0 for fmt in FORMAT_CANDIDATES_DATETIME}
    valid_total = 0
    # Adaptive ordering: promote the last successful format to the front
    # so subsequent values (usually the same format) match on the first try.
    last_hit: Optional[str] = None

    for val in series:
        if not val:
            continue

        normalized = _strip_tz_suffix(_normalize_datetime_string(val))

        matched = False

        # Try the previously successful format first
        if last_hit is not None:
            try:
                pd.to_datetime(normalized, format=last_hit, errors="raise")
                counts[last_hit] += 1
                valid_total += 1
                matched = True
            except Exception:
                pass

        if not matched:
            for fmt in FORMAT_CANDIDATES_DATETIME:
                if fmt == last_hit:
                    continue  # already tried
                try:
                    pd.to_datetime(normalized, format=fmt, errors="raise")
                    counts[fmt] += 1
                    valid_total += 1
                    last_hit = fmt
                    matched = True
                    break
                except Exception:
                    continue

        if not matched:
            # Fallback: dateutil handles many additional formats
            try:
                dateutil_parser.parse(normalized)
                valid_total += 1
            except (ValueError, OverflowError):
                pass

    if valid_total == 0:
        return ""

    best_fmt = max(counts, key=counts.get)
    best_count = counts[best_fmt]

    # Require a dominant pattern before assigning a format.
    return best_fmt if best_count / valid_total >= 0.6 else ""

@st.cache_data()
def build_metadata_df_from_df(df_origin: pd.DataFrame) -> pd.DataFrame:
    df = df_origin.copy()
    cols = []
    for c in df.columns:

        

        dtype = detect_csvw_datatype_from_series(df[c])

        Date_Time_format = detect_date_format_from_series(df[c]) if dtype in ("date", "dateTime", "time") else ""
        cols.append({
            "name": c,
            "column_type": dtype,
            "column_format": Date_Time_format,
            "concept_type": "",
            "concept": "",
            "concept_uri": "",
            "unit_symbol": "",
            "unit_uri": "",
            "quantity_kind_uri": "",
            "method": "",
            "method_uri": "",
            "description": "",
        })

    return pd.DataFrame(cols)

@st.cache_data()
def read_csvBytes_with_sniffer(raw:bytes) -> pd.DataFrame:

    def _best_delimiter(lines: list[str]) -> tuple[str, int]:
        """Return (delimiter, expected_field_count) using the most consistent split."""
        candidates = [",", ";", "\t", "|"]
        best_sep = ","
        best_fields = 1
        best_support = -1

        for sep in candidates:
            split_counts = [line.count(sep) for line in lines if line.count(sep) > 0]
            if not split_counts:
                continue

            mode_count = max(set(split_counts), key=split_counts.count)
            support = split_counts.count(mode_count)
            if support > best_support or (support == best_support and mode_count > (best_fields - 1)):
                best_support = support
                best_sep = sep
                best_fields = mode_count + 1

        return best_sep, best_fields

    def _looks_like_datetime(text_token: str) -> bool:
        token = str(text_token or "").strip()
        if len(token) < 6:
            return False
        try:
            dateutil_parser.parse(token)
            return True
        except Exception:
            return False

    def _looks_like_number(text_token: str) -> bool:
        token = str(text_token or "").strip()
        if token == "":
            return False
        try:
            float(token)
            return True
        except Exception:
            return False

    def _looks_like_data_row(parts: list[str]) -> bool:
        non_empty = [p.strip() for p in parts if str(p).strip() != ""]
        if not non_empty:
            return False

        matches = 0
        for token in non_empty:
            normalized = token.lower()
            if normalized in {"true", "false", "0", "1"}:
                matches += 1
                continue
            if _looks_like_number(token) or _looks_like_datetime(token):
                matches += 1

        return (matches / len(non_empty)) >= 0.6

    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        import chardet
        enc = chardet.detect(raw)["encoding"] or "cp1252"
        text = raw.decode(enc, errors="replace")

    sample = text[:65536]  # first 64KB is usually enough
    try:
        dialect_uploaded = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        separator_uploaded = dialect_uploaded.delimiter
    except Exception:
        separator_uploaded, _ = _best_delimiter([line for line in text.splitlines() if line.strip()])

    try:
        return pd.read_csv(io.StringIO(text), sep=separator_uploaded)
    except pd.errors.ParserError:
        # Fallback for files that have a metadata preamble before tabular rows.
        lines = [line for line in text.splitlines() if line.strip()]
        if not lines:
            return pd.DataFrame()

        sep, expected_fields = _best_delimiter(lines)
        expected_delims = max(0, expected_fields - 1)

        start_idx = 0
        min_streak = 2
        for i in range(len(lines)):
            if lines[i].count(sep) != expected_delims:
                continue
            streak = 0
            for j in range(i, min(i + 5, len(lines))):
                if lines[j].count(sep) == expected_delims:
                    streak += 1
            if streak >= min_streak:
                start_idx = i
                break

        tabular_text = "\n".join(lines[start_idx:])
        if not tabular_text.strip():
            return pd.DataFrame()

        first_parts = [p.strip() for p in lines[start_idx].split(sep)]
        second_parts = []
        if start_idx + 1 < len(lines):
            second_parts = [p.strip() for p in lines[start_idx + 1].split(sep)]

        first_is_data = _looks_like_data_row(first_parts)
        second_is_data = _looks_like_data_row(second_parts) if second_parts else False
        has_header = not (first_is_data and second_is_data)

        if has_header:
            return pd.read_csv(
                io.StringIO(tabular_text),
                sep=sep,
                engine="python",
                on_bad_lines="skip",
            )

        df = pd.read_csv(
            io.StringIO(tabular_text),
            sep=sep,
            header=None,
            engine="python",
            on_bad_lines="skip",
        )
        df.columns = [f"column_{idx + 1}" for idx in range(df.shape[1])]
        return df

def _canonicalize_metadata_column_name(column_name: str) -> str:
    key = str(column_name or "").strip().lower().replace("-", "_").replace(" ", "_")
    alias_map = {
        "datatype": "column_type",
        "data_type": "column_type",
        "result_type": "column_type",
        "date_time_format": "column_format",
        "datetime_format": "column_format",
        "type": "column_type",
        "element uri": "concept_uri",
        "elementuri": "concept_uri",
        "concepturi": "concept_uri",
        "unit_symbol": "unit_symbol",
        "unit_symbol_": "unit_symbol",
        "unit": "unit_symbol",
        "unit_uri": "unit_uri",
        "unituri": "unit_uri",
        "quantity kind_uri": "quantity_kind_uri",
        "quantity_kinduri": "quantity_kind_uri",
        "quantitykind_uri": "quantity_kind_uri",
        "quantity_kind": "quantity_kind_uri",
        "quantity_kind uri": "quantity_kind_uri",
        "methode_uri": "method_uri",
        "methodeuri": "method_uri",
    }
    return alias_map.get(key, key).strip()


def _is_metadata_export_filename(filename: str) -> bool:
    return str(filename or "").strip().lower().endswith(METADATA_EXPORT_FILENAME_SUFFIX)


def _is_ignored_remote_csv_filename(filename: str) -> bool:
    return str(filename or "").strip().lower() in IGNORED_REMOTE_CSV_FILENAMES


def _parse_metadata_csv_bytes(raw: bytes, source_filename: str, show_errors: bool = True) -> dict[str, pd.DataFrame]:
    df = read_csvBytes_with_sniffer(raw)
    if df.empty or "name" not in {str(c).strip().lower() for c in df.columns}:
        if show_errors:
            st.error("CSV metadata must contain a 'name' column matching column names in the data.")
        return {}

    normalized = df.copy()
    normalized.columns = [str(c).strip() for c in normalized.columns]
    normalized = normalized.rename(columns={c: _canonicalize_metadata_column_name(c) for c in normalized.columns})

    for col in ["column_type", "column_format", "concept_type", "concept", "concept_uri", "unit_symbol", "unit_uri", "quantity_kind_uri", "method", "method_uri", "description"]:
        if col not in normalized.columns:
            normalized[col] = ""

    normalized["filename"] = source_filename
    return {source_filename: normalized}


@st.cache_data()
def import_metadata_from_file(uploaded_file) -> dict[str, pd.DataFrame]:
    """Parse a single metadata upload and return a dict keyed by filename.

    This is the multi-file-ready shape: each file yields one keyed entry, so later
    imports can be matched to the correct table using filename-aware logic.
    """
    if isinstance(uploaded_file, (list, tuple)):
        result: dict[str, pd.DataFrame] = {}
        for uploaded in uploaded_file:
            parsed = import_metadata_from_file(uploaded)
            if isinstance(parsed, dict):
                result.update(parsed)
        return result

    name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()
    source_filename = getattr(uploaded_file, "name", "") or "metadata"

    try:
        if name.endswith('.csv'):
            return _parse_metadata_csv_bytes(raw, source_filename=source_filename, show_errors=True)

        elif name.endswith('.json'):
            # TODO: check for ingestions of these kind of formats
            st.error('🚧 JSON metadata import not yet implemented.')
            # text = raw.decode("utf-8")
            # j = json.loads(text)

            # if isinstance(j, dict) and j.get('fields'):
            #     rows = []
            #     for f in j['fields']:
            #         rows.append({
            #             'name': f.get('name'),
            #             'column_type': f.get('type') or '',
            #             'column_format': '',
            #             'concept': f.get('concept') or f.get('title') or '',
            #             'element': f.get('title') or '',
            #             'element_uri': f.get('element_uri') or f.get('element_uri') or f.get('concept_uri') or '',
            #             'unit_symbol': f.get('unit_symbol') or f.get('unit') or '',
            #             'unit_uri': f.get('unit_uri') or f.get('unit uri') or '',
            #             'quantity_kind_uri': f.get('quantity_kind_uri') or f.get('quantity_kind_uri') or '',
            #             'method': f.get('method') or '',
            #             'description': f.get('description') or '',
            #         })
            #     return pd.DataFrame(rows)

            # if isinstance(j, dict) and j.get('tableSchema') and j['tableSchema'].get('columns'):
            #     rows = []
            #     for f in j['tableSchema']['columns']:
            #         rows.append({
            #             'name': f.get('name'),
            #             'column_type': f.get('datatype') or f.get('column_type') or '',
            #             'column_format': '',
            #             'concept': f.get('concept') or '',
            #             'element': (f.get('titles') or [''])[0] if isinstance(f.get('titles'), list) else (f.get('titles') or ''),
            #             'element_uri': f.get('element_uri') or f.get('element_uri') or f.get('concept_uri') or '',
            #             'unit_symbol': f.get('unit_symbol') or f.get('schema:unitCode') or f.get('unit') or '',
            #             'unit_uri': f.get('unit_uri') or f.get('unit uri') or '',
            #             'quantity_kind_uri': f.get('quantity_kind_uri') or f.get('quantity_kind_uri') or '',
            #             'method': f.get('method') or '',
            #             'description': f.get('dc:description') or f.get('description') or '',
            #         })
            #     return pd.DataFrame(rows)

            # st.error('Unrecognized JSON metadata format (expecting TableSchema or CSVW).')
            # return None
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
        file_response_head
        file_response_head.raise_for_status()
    except Exception as e:
        st.error(f"Failed to read file {url} from Zenodo record: {e}")
        return None, None, None
    name_file = filename_from_url(url)                
    ext_file = os.path.splitext(name_file)[1].lower()


    file_response = requests.get(url)

    return file_response, name_file, ext_file


@st.cache_data()
def request_file_from_github(url: str) -> tuple[Optional[requests.Response], Optional[str], Optional[str], Optional[str]]:
    try:
        raw_url = normalize_github_file_url(url)
        file_response = requests.get(raw_url)
        file_response.raise_for_status()
    except Exception as e:
        st.error(f"Failed to read file from GitHub URL '{url}': {e}")
        return None, None, None, None

    name_file = filename_from_url(raw_url)
    ext_file = os.path.splitext(name_file)[1].lower()
    return file_response, name_file, ext_file, raw_url


def _extract_github_scope_path(github_input_url: str) -> Optional[str]:
    """Return folder scope path from a GitHub tree URL (if present)."""
    try:
        parsed = urlparse(github_input_url)
        if parsed.netloc.lower() not in {"github.com", "www.github.com"}:
            return None

        parts = [p for p in parsed.path.split("/") if p]
        # github.com/{owner}/{repo}/tree/{branch}/{path...}
        if len(parts) >= 5 and parts[2] == "tree":
            return "/".join(parts[4:]).strip("/")
    except Exception:
        return None
    return None


def build_github_table_key(raw_url: str, source_url: Optional[str] = None, sheet_name: Optional[str] = None) -> str:
    """Create concise, collision-safe table keys for GitHub imports.

    If a folder URL was provided, keys are relative to that folder.
    """
    parsed = urlparse(raw_url)
    parts = [p for p in parsed.path.split("/") if p]

    # raw.githubusercontent.com/{owner}/{repo}/{branch}/{path...}
    if parsed.netloc.lower() == "raw.githubusercontent.com" and len(parts) >= 5:
        rel_path = "/".join(parts[3:])

        scope_path = _extract_github_scope_path(source_url) if source_url else None
        if scope_path:
            scope_prefix = f"{scope_path}/"
            if rel_path == scope_path:
                rel_path = filename_from_url(raw_url)
            elif rel_path.startswith(scope_prefix):
                rel_path = rel_path[len(scope_prefix):]

        base_key = rel_path
    else:
        base_key = filename_from_url(raw_url)

    return f"{base_key} | {sheet_name}" if sheet_name else base_key

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
                         tabular_exts:list[str], context_exts:list[str], imported_metadata_by_filename: dict[str, pd.DataFrame] | None = None):
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
                if _is_ignored_remote_csv_filename(member_basename):
                    continue
                if _is_metadata_export_filename(member_basename):
                    parsed_metadata = _parse_metadata_csv_bytes(raw, source_filename=member_basename, show_errors=False)
                    if parsed_metadata and imported_metadata_by_filename is not None:
                        imported_metadata_by_filename.update(parsed_metadata)
                    continue
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
            
# pick primary_key candidate from rows (list of dicts)
def pick_primary_key(headers: List[str], rows_sample: List[Dict[str,Any]]) -> Optional[str]:
    # compute unique ratio per header
    scores = []
    for h in headers:
        non_empty = 0
        seen = set()
        for r in rows_sample:
            v = r.get(h, "")
            s = "" if v is None else str(v).strip()
            if s == "":
                continue
            non_empty += 1
            seen.add(s)
        if non_empty == 0:
            uniq_ratio = 0.0
        else:
            uniq_ratio = len(seen) / non_empty
        scores.append((h, non_empty, len(seen), uniq_ratio))

    # prefer name hints
    hints = ['id', 'identifier', 'uuid', 'code', 'key']
    strong = [s for s in scores if s[1] >= 3 and s[3] >= 0.98]
    good = [s for s in scores if s[1] >= 3 and s[3] >= 0.8]
    perfect = [s for s in scores if s[1] >= 3 and s[2] == s[1]]

    def pick_by_hint(cands):
        for hint in hints:
            for c in cands:
                if hint in c[0].lower():
                    return c[0]
        return None
    
    if perfect:
        return pick_by_hint(perfect) or perfect[0][0]
    if len(strong) == 1:
        return strong[0][0]
    if len(strong) > 1:
        return pick_by_hint(strong) or sorted(strong, key=lambda x: -x[3])[0][0]
    if good:
        return pick_by_hint(good) or sorted(good, key=lambda x: -x[3])[0][0]

    return None

# -------------------- UI --------------------

st.title("📖 STEP 1 : input of information")

st.markdown("""
            Ok, let's start by the firste uploading your data and eventually some existing metadata. But be assured, this tool is processing the information in a way that it never sends the raw data to any third party.
            Later in the app, there is an option to augment the information, making use of an LLM. Also in that case, only the variable names and the context text (if you choose to upload some) will be used.
            """)

st.markdown(""" --- """)

mode = st.radio("Input mode", options=["single CSV", "linked", "excel", "url - zenodo", "url - github"], index=0, horizontal=True, key='input_mode')


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
remote_imported_metadata_by_filename: dict[str, pd.DataFrame] = {}
upload_tokens: list[str] = []
url_input = ""
_filename = None
_record_id = None
_uploaded = None
_site_file = None
_obs_file = None
_excel_file = None
_github_urls: list[str] = []
_github_urls_input = ""

# ---------- Phase 1: render widgets, collect tokens — no file I/O ----------
with col1:
    st.markdown("**Import tabular data**")
    if mode == 'single CSV':
        _uploaded = st.file_uploader("Upload CSV file", type=['csv'], key='single_upload')
        _filename = _uploaded.name if _uploaded else None
        if _uploaded:
            upload_tokens.append(f"csv:{_uploaded.name}:{_uploaded.size}")

    elif mode == 'linked':
        _site_file = st.file_uploader("Upload Sites CSV (locations)", type=['csv'], key='site_upload')
        _obs_file = st.file_uploader("Upload Observations CSV", type=['csv'], key='obs_upload')
        _filename = _obs_file.name if _obs_file else None
        if _site_file:
            upload_tokens.append(f"csv:{_site_file.name}:{_site_file.size}")
        if _obs_file:
            upload_tokens.append(f"csv:{_obs_file.name}:{_obs_file.size}")

    elif mode == 'excel':
        _excel_file = st.file_uploader("Upload Excel workbook", type=['xlsx', 'xls'], key='excel_upload')
        _filename = _excel_file.name if _excel_file else None
        if _excel_file:
            upload_tokens.append(f"excel:{_excel_file.name}:{_excel_file.size}")

    elif mode == 'url - zenodo':
        url_input = st.text_input(
            "Enter URL to zenodo records. (prefered format 'https://zenodo.org/records/{ID_number}')",
            key='url_input',
        )
        if url_input:
            try:
                _record_id = get_record_id_from_Zenodo_url(url_input)
                upload_tokens.append(f"zenodo:{_record_id}")
                # Keep the active cache key current so clear-on-new-input preserves the right entry
                st.session_state["_active_zenodo_cache_key"] = f"_zenodo_ingest_cache_{_record_id}"
            except ValueError as e:
                st.error(str(e))

    elif mode == 'url - github':
        _github_urls_input = st.text_input(
            "Enter a GitHub URL (file, folder, or repository)",
            key='github_urls_input',
        )

        if _github_urls_input.strip():
            candidate_urls = [u.strip() for u in re.split(r"[\n;,]", _github_urls_input) if u.strip()]
            normalized_urls = []
            for candidate in candidate_urls:
                try:
                    normalized_urls.append(normalize_github_input_url(candidate))
                except ValueError as e:
                    st.error(f"{candidate}: {e}")

            if normalized_urls:
                _github_urls = normalized_urls
                for gh_url in _github_urls:
                    upload_tokens.append(f"github:{gh_url}")
                github_cache_seed = "|".join(sorted(_github_urls) + [GITHUB_INGEST_SCHEMA_VERSION])
                github_cache_digest = hashlib.sha256(github_cache_seed.encode("utf-8")).hexdigest()
                st.session_state["_active_github_cache_key"] = f"_github_ingest_cache_{github_cache_digest}"
            else:
                st.session_state["_active_github_cache_key"] = None


    if mode not in ('url - zenodo', 'url - github'):
        st.session_state["_active_zenodo_cache_key"] = None
        st.session_state.pop("remote_context_files_url", None)
        st.session_state.pop("remote_context_metadata", None)

    if mode != 'url - github':
        st.session_state["_active_github_cache_key"] = None

# ---------- Phase 2: signature check ----------
has_active_input = bool(upload_tokens) or bool(url_input)
current_input_signature = _build_input_signature(mode, upload_tokens, url_input)
previous_input_signature = st.session_state.get("_input_signature")
_need_processing = has_active_input and (previous_input_signature != current_input_signature)

if _need_processing:
    _clear_dependent_session_state_for_new_input()
    st.session_state["_input_signature"] = current_input_signature
elif "tabular_data_dict" in st.session_state:
    # Signature unchanged or navigating back: skip all file I/O, restore from session state
    tabular_dict = {k: df.copy() for k, df in st.session_state["tabular_data_dict"].items()}
    filename_dict = dict(st.session_state.get("filename_dict", {}))
    zipped_context_files = list(st.session_state.get("remote_context_files_from_zip", []))
    remote_imported_metadata_by_filename = dict(st.session_state.get("imported_metadata_by_filename_remote", {}))
    if mode == 'linked':
        site_df = st.session_state.get("_linked_site_df")
        obs_df = st.session_state.get("_linked_obs_df")
        if _filename is None:
            _filename = st.session_state.get("_linked_obs_filename")

# ---------- Phase 3: file I/O — only runs when input actually changed ----------
if _need_processing:
    remote_imported_metadata_by_filename = {}
    if mode == 'single CSV':
        if _uploaded:
            try:
                raw = _uploaded.getvalue()
                uploaded_df = read_csvBytes_with_sniffer(raw)
                tabular_dict[_filename] = uploaded_df
                filename_dict[_filename] = _filename
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")

    elif mode == 'linked':
        if _site_file:
            try:
                raw = _site_file.getvalue()
                site_df = read_csvBytes_with_sniffer(raw)
                filename_dict[_site_file.name] = _site_file.name
                st.session_state["_linked_site_df"] = site_df
            except Exception as e:
                st.error(f"Failed to read sites CSV: {e}")
        if _obs_file:
            try:
                raw = _obs_file.getvalue()
                obs_df = read_csvBytes_with_sniffer(raw)
                filename_dict[_obs_file.name] = _obs_file.name
                st.session_state["_linked_obs_df"] = obs_df
                st.session_state["_linked_obs_filename"] = _obs_file.name
            except Exception as e:
                st.error(f"Failed to read observations CSV: {e}")

    elif mode == 'excel':
        if _excel_file:
            try:
                df_dict = get_excel(_excel_file)
                for sheet_name, df in df_dict.items():
                    tabular_dict[f"{_filename} | {sheet_name}"] = df
                    filename_dict[f"{_filename} | {sheet_name}"] = _filename
            except Exception as e:
                st.error(f"Failed to read Excel: {e}")

    elif mode == 'url - zenodo' and _record_id:
        # PIN : Zenodo Test URLS
        # https://zenodo.org/records/14034500 -> empty excel
        # https://zenodo.org/records/14726493 -> csv
        # https://zenodo.org/records/8083652 -> dubble excel
        # https://zenodo.org/records/10028494 -> complex zip
        # https://zenodo.org/records/17305831 -> AI4SoilHealth SOC
        # https://zenodo.org/records/19177539 -> connected CSV's
        # https://zenodo.org/records/19723699 -> Echo repo
        # https://zenodo.org/records/14610222 -> Hungarian Soil Degradation with readme in description

        #######################################################
        zenodo_cache_key = f"_zenodo_ingest_cache_{_record_id}"
        cached_zenodo = st.session_state.get(zenodo_cache_key)
        if cached_zenodo is not None:
            tabular_dict.update({k: v.copy() for k, v in cached_zenodo["tabular_dict"].items()})
            filename_dict.update(dict(cached_zenodo["filename_dict"]))
            st.session_state['remote_context_files_url'] = list(cached_zenodo["files_url_context"])
            st.session_state['remote_context_metadata'] = dict(cached_zenodo["metadata_context"])
            zipped_context_files.extend(list(cached_zenodo["zipped_context_files"]))
            remote_imported_metadata_by_filename = dict(cached_zenodo.get("imported_metadata_by_filename_remote", {}))
        else:
            filtered_extensions_tabular = ['.csv', '.xlsx', '.xls']
            files_url_tabular = get_files_URL_from_Zenodo_id(_record_id, extensions=filtered_extensions_tabular)

            filtered_extensions_zip = ['.zip']
            files_url_zip = get_files_URL_from_Zenodo_id(_record_id, extensions=filtered_extensions_zip)

            filtered_extensions_context = ['.doc', '.docx', '.pdf', '.md', '.txt']
            files_url_context = get_files_URL_from_Zenodo_id(_record_id, extensions=filtered_extensions_context)
            st.session_state['remote_context_files_url'] = files_url_context

            metadata_context_full = get_metadata_from_Zenodo_id(_record_id)
            metadata_context = {k: metadata_context_full[k] for k in {"title", "description"} if k in metadata_context_full}
            if "description" in metadata_context:
                metadata_context["description"] = description_to_plain_text(metadata_context["description"])
            st.session_state['remote_context_metadata'] = metadata_context

            for file_url in files_url_tabular:
                file_response, name_file, ext_file = request_file_from_zenodo(file_url)
                if ext_file in ['.xlsx', '.xls']:
                    bitesIO = io.BytesIO(file_response.content)
                    bitesIO.name = name_file
                    df_dict = get_excel(bitesIO)
                    for sheet_name, df in df_dict.items():
                        tabular_dict[f"{name_file} | {sheet_name}"] = df
                        filename_dict[name_file] = file_url
                elif ext_file == '.csv':
                    if _is_ignored_remote_csv_filename(name_file):
                        continue
                    if _is_metadata_export_filename(name_file):
                        parsed_metadata = _parse_metadata_csv_bytes(
                            file_response.content,
                            source_filename=name_file,
                            show_errors=False,
                        )
                        if parsed_metadata:
                            remote_imported_metadata_by_filename.update(parsed_metadata)
                    else:
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
                                        context_exts=filtered_extensions_context,
                                        imported_metadata_by_filename=remote_imported_metadata_by_filename)

            st.session_state[zenodo_cache_key] = {
                "tabular_dict": {k: v.copy() for k, v in tabular_dict.items()},
                "filename_dict": dict(filename_dict),
                "zipped_context_files": list(zipped_context_files),
                "files_url_context": list(st.session_state.get('remote_context_files_url', [])),
                "metadata_context": dict(st.session_state.get('remote_context_metadata', {})),
                "imported_metadata_by_filename_remote": dict(remote_imported_metadata_by_filename),
            }

    elif mode == 'url - github' and _github_urls:
        # PIN : Github Test URLS

        # https://github.com/soilwise-he/soil-observation-data-encodings/tree/main/EXAMPLES/example1
        # https://github.com/soilwise-he/soil-observation-data-encodings/tree/main/EXAMPLES/example2
        # https://github.com/soilwise-he/soil-observation-data-encodings/tree/main/EXAMPLES/example3
        # https://github.com/soilwise-he/soil-observation-data-encodings/tree/main/EXAMPLES/example4
        # https://github.com/soilwise-he/soil-observation-data-encodings/tree/main/EXAMPLES/example5/meetpunten_bodemlocatie_2021-032627_1912_CN_SWC

        #######################################################
        filtered_extensions_tabular = ['.csv', '.xlsx', '.xls']
        filtered_extensions_zip = ['.zip']
        filtered_extensions_context = ['.doc', '.docx', '.pdf', '.md', '.txt']
        github_cache_seed = "|".join(sorted(_github_urls) + [GITHUB_INGEST_SCHEMA_VERSION])
        github_cache_key = st.session_state.get("_active_github_cache_key") or f"_github_ingest_cache_{hashlib.sha256(github_cache_seed.encode('utf-8')).hexdigest()}"
        cached_github = st.session_state.get(github_cache_key)

        if cached_github is not None:
            tabular_dict.update({k: v.copy() for k, v in cached_github["tabular_dict"].items()})
            filename_dict.update(dict(cached_github["filename_dict"]))
            st.session_state['remote_context_files_url'] = list(cached_github["files_url_context"])
            zipped_context_files.extend(list(cached_github["zipped_context_files"]))
            remote_imported_metadata_by_filename = dict(cached_github.get("imported_metadata_by_filename_remote", {}))
        else:
            filtered_extensions_github = filtered_extensions_tabular + filtered_extensions_zip + filtered_extensions_context
            processed_raw_urls = set()
            github_context_urls = set()

            for github_input_url in _github_urls:
                try:
                    resolved_file_urls = resolve_github_input_to_file_urls(
                        github_input_url,
                        extensions=filtered_extensions_github,
                        recursive=True,
                    )
                except Exception as e:
                    st.error(f"Failed to resolve GitHub URL '{github_input_url}': {e}")
                    continue

                if not resolved_file_urls:
                    st.warning(f"No supported data files found at {github_input_url}")
                    continue

                for file_url in resolved_file_urls:
                    if file_url in processed_raw_urls:
                        continue
                    processed_raw_urls.add(file_url)

                    file_response, name_file, ext_file, raw_url = request_file_from_github(file_url)
                    if file_response is None or name_file is None or ext_file is None or raw_url is None:
                        continue

                    if ext_file in ['.xlsx', '.xls']:
                        bitesIO = io.BytesIO(file_response.content)
                        bitesIO.name = name_file
                        df_dict = get_excel(bitesIO)
                        for sheet_name, df in df_dict.items():
                            table_key = build_github_table_key(raw_url, source_url=github_input_url, sheet_name=sheet_name)
                            tabular_dict[table_key] = df
                            filename_dict[table_key] = raw_url
                    elif ext_file == '.csv':
                        if _is_ignored_remote_csv_filename(name_file):
                            continue
                        if _is_metadata_export_filename(name_file):
                            parsed_metadata = _parse_metadata_csv_bytes(
                                file_response.content,
                                source_filename=name_file,
                                show_errors=False,
                            )
                            if parsed_metadata:
                                remote_imported_metadata_by_filename.update(parsed_metadata)
                        else:
                            uploaded_df = read_csvBytes_with_sniffer(file_response.content)
                            table_key = build_github_table_key(raw_url, source_url=github_input_url)
                            tabular_dict[table_key] = uploaded_df
                            filename_dict[table_key] = raw_url
                    elif ext_file in filtered_extensions_zip:
                        process_zip_from_url(raw_url,
                                                tabular_dict,
                                                zipped_context_files,
                                                filename_dict,
                                                tabular_exts=filtered_extensions_tabular,
                                                context_exts=filtered_extensions_context,
                                                imported_metadata_by_filename=remote_imported_metadata_by_filename)
                    elif ext_file in filtered_extensions_context:
                        github_context_urls.add(raw_url)
                    else:
                        st.warning(f"Skipped unsupported file type '{ext_file}' from {file_url}")

            st.session_state['remote_context_files_url'] = list(github_context_urls)
            st.session_state[github_cache_key] = {
                "tabular_dict": {k: v.copy() for k, v in tabular_dict.items()},
                "filename_dict": dict(filename_dict),
                "zipped_context_files": list(zipped_context_files),
                "files_url_context": list(st.session_state.get('remote_context_files_url', [])),
                "imported_metadata_by_filename_remote": dict(remote_imported_metadata_by_filename),
            }

    st.session_state["remote_context_files_from_zip"] = zipped_context_files
    st.session_state["imported_metadata_by_filename_remote"] = remote_imported_metadata_by_filename



with col2:
    st.markdown("**Import existing metadata**")
    myinfo = st.empty()
    metadata_file = st.file_uploader(
        "Upload metadata files (CSV or JSON TableSchema/CSVW)",
        type=['csv', 'json'],
        accept_multiple_files=True,
        key='meta_upload',
    )
    if 'metadata_df' not in st.session_state or metadata_file is None:
            myinfo.info("The metadata files are optional but each file should contain at least a **'name'** column matching the headers of the uploaded data. Optionally, they can include columns such as **'column_type'**, **'concept'**, **'unit'**, **'method'**, and **'description'** for additional annotations.")

# Handle linked mode linking columns
if mode == 'linked' and site_df is not None and obs_df is not None:
    st.write("### Select linking columns")
    site_cols = list(site_df.columns)
    obs_cols = list(obs_df.columns)
    site_id_col = st.selectbox("ID column (sites)", options=[''] + site_cols, key='site_id_col_select')
    obs_fk_col = st.selectbox("Foreign key column (observations)", options=[''] + obs_cols, key='obs_fk_col_select')
    if site_id_col and obs_fk_col:
        st.write("Preview of linked join (left on observations -> sites):")
        try:
            merged = obs_df.merge(site_df, left_on=obs_fk_col, right_on=site_id_col, how='left', suffixes=('_obs', '_site'))
            st.dataframe(merged.head(5))
            uploaded_df = merged  # use merged for building metadata
            if _filename:
                tabular_dict[_filename] = uploaded_df
        except Exception as e:
            st.error(f"Failed to merge tables: {e}")

meta_key = f"metadata_df"
if meta_key not in st.session_state or not isinstance(st.session_state.get(meta_key), dict):
    st.session_state[meta_key] = {}
else:
    st.session_state[meta_key] = normalize_metadata_columns(st.session_state[meta_key])



with col2:
    imported_metadata_by_filename: dict[str, pd.DataFrame] = {}
    remote_metadata_files = dict(st.session_state.get("imported_metadata_by_filename_remote", {}))
    metadata_files = metadata_file if isinstance(metadata_file, list) else ([metadata_file] if metadata_file is not None else [])
    if metadata_files or remote_metadata_files:
        imported_metadata_by_filename = dict(remote_metadata_files)
        for uploaded_meta in metadata_files:
            imported_meta_dict = import_metadata_from_file(uploaded_meta)
            if isinstance(imported_meta_dict, dict):
                imported_metadata_by_filename.update(imported_meta_dict)
        if imported_metadata_by_filename:
            st.session_state["imported_metadata_by_filename"] = imported_metadata_by_filename
            st.success(f"Imported metadata files parsed ({len(imported_metadata_by_filename)} file(s)).")
        else:
            st.session_state.pop("imported_metadata_by_filename", None)
    else:
        st.session_state.pop("imported_metadata_by_filename", None)


# If a dataframe is present (uploaded or merged), show preview and build metadata
if tabular_dict:

    st.markdown(""" --- """)
    st.markdown(f"### Data preview")
    st.caption("HINT: SHIFT+scroll to navigate the tabs horizontally if needed.")

    # HTML for having the tabs colored for better UX and more visible + gap between them
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
    if "primary_keys" not in st.session_state:
        st.session_state["primary_keys"] = {}
    if "primary_keys_guess" not in st.session_state:
        st.session_state["primary_keys_guess"] = {}

    for tab, key in zip(tabs, tab_labels):
        df = tabular_dict[key]

        with tab:


            discard_toggle_key = f"discard_table_{key}"
            move_toggle_key = f"move_to_context_{key}"

            discard_table = st.toggle(
                "Discard this table, it doesn't contain tabular data to be annotated",
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
            
            



            meta_key = f"metadata_df"
            if meta_key not in st.session_state:
                st.session_state[meta_key] = {}


            if key not in st.session_state[meta_key]: 
                st.session_state[meta_key][key] = build_metadata_df_from_df(df)

            imported_metadata_by_filename = st.session_state.get("imported_metadata_by_filename", {})
            if imported_metadata_by_filename:
                matched_import_rows = []
                table_filename = str(st.session_state.get("filename_dict", {}).get(key, key))
                table_filename_tokens = build_filename_match_tokens(table_filename) | build_filename_match_tokens(key)

                for source_filename, imported_df in imported_metadata_by_filename.items():
                    if imported_df.empty:
                        continue
                    file_tokens = build_filename_match_tokens(source_filename)
                    filename_values = imported_df.get("filename", pd.Series([""] * len(imported_df)))
                    matches_filename = filename_values.astype(str).apply(
                        lambda value: bool(build_filename_match_tokens(value) & (file_tokens | table_filename_tokens))
                    )
                    if matches_filename.any():
                        matched_import_rows.append(imported_df.loc[matches_filename].copy())
                        continue
                    if file_tokens & table_filename_tokens or len(imported_metadata_by_filename) == 1:
                        matched_import_rows.append(imported_df.copy())

                if matched_import_rows:
                    imported_rows = pd.concat(matched_import_rows, ignore_index=True, sort=False)
                    imported_rows = imported_rows.drop(columns=["filename"], errors="ignore")
                    imported_rows = imported_rows.drop_duplicates(subset=["name"], keep="last")

                    imported_match_rows = imported_rows[
                        imported_rows["name"].astype(str).str.strip().isin(
                            st.session_state[meta_key][key]["name"].astype(str).str.strip()
                        )
                    ].copy()
                    if not imported_match_rows.empty:
                        st.session_state[meta_key][key] = apply_new_metadata_info(
                            {key: imported_match_rows},
                            {key: st.session_state[meta_key][key]},
                            overwrite='yes',
                        )[key]

            if key not in st.session_state["primary_keys_guess"]:
                st.session_state["primary_keys_guess"][key] = pick_primary_key(
                    df.columns.tolist(), df.head(200).to_dict(orient='records')
                )
            primary_keys_guess = st.session_state["primary_keys_guess"][key]
            pk_c1, pk_c2,_ = st.columns([2, 2, 6])
            pk_c1.markdown("Select [primary_key](https://en.wikipedia.org/wiki/Primary_key) column (if present)")
            st.session_state["primary_keys"][key]=pk_c2.selectbox(
                "primary_key",
                label_visibility = "collapsed",
                options=[''] + df.columns.tolist(),
                index=0 if primary_keys_guess is None else df.columns.get_loc(primary_keys_guess) + 1,
                key=f"primary_key_select_{key}",
                width=200,
            )
            
            # Add primary_key to metadata dataframe
            primary_key_col = st.session_state["primary_keys"][key]
            if primary_key_col:  # Only if a primary_key was selected
                if "primary_key" not in st.session_state[meta_key][key].columns:
                    st.session_state[meta_key][key]["primary_key"] = False
                st.session_state[meta_key][key]["primary_key"] = st.session_state[meta_key][key]['name']== primary_key_col
            else:
                st.session_state[meta_key][key]["primary_key"] = False


            # Preview the dataframe
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



            st.markdown("#### Datatype")
            st.caption("Change the datatype and date format as needed.")

            original_metadata_df = st.session_state[meta_key][key]
            all_columns = original_metadata_df.columns.tolist()
            edited_df = st.data_editor(
                            original_metadata_df,
                            width='stretch',
                            key=f"editor_{key}",
                            disabled=[col for col in all_columns if col != "column_type" and col != "column_format"],
                            column_config={
                                "column_type": st.column_config.SelectboxColumn(
                                    options=DATA_TYPE_OPTIONS,
                                ),
                            }, 
                        )

            if not edited_df.equals(original_metadata_df):
                # A data-editor change already causes Streamlit to rerun the script.
                # Triggering another rerun here can repeatedly replay retained editor
                # state after a different table has been discarded.
                st.session_state[meta_key][key] = edited_df.copy()

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
    st.session_state["tabular_data_dict_preview"] = {key: df.copy().head(10) for key, df in tabular_dict.items()}
    

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



# -------------------- reach us --------------------
add_Soilwise_contact_sidebar()
