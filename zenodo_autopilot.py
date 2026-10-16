from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import types
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

ROOT = Path(__file__).resolve().parent
BATCH_HELPER_FILE = ROOT / "zenodo_metadata_batch.py"
INPUT_PAGE_FILE = ROOT / "pages" / "01_📖_Input.py"
METADATA_UTIL_FILE = ROOT / "util" / "metadata.py"
QUDT_PAGE_FILE = ROOT / "pages" / "05_📐_Unit_of_measurement.py"
CONTEXT_PAGE_FILE = ROOT / "pages" / "02_✒️_Context.py"
SORTING_PAGE_FILE = ROOT / "pages" / "03_🪣_Column_Sorting.py"
METHOD_PAGE_FILE = ROOT / "pages" / "06_⚙️_Methode.py"
KEYWORD_MATCHER_FILE = ROOT / "pages" / "04_🕵️_Keyword_Matcher.py"
def _create_streamlit_stub() -> types.ModuleType:
    class _Dummy:
        def __call__(self, *args, **kwargs):
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def expander(self, *args, **kwargs):
            return self

        def button(self, *args, **kwargs):
            return False

        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

    class _SessionState(dict):
        def get(self, key, default=None):
            return super().get(key, default)

        def pop(self, key, default=None):
            return super().pop(key, default) if key in self else default

    def _first_option(options):
        if options is None:
            return None
        values = list(options)
        return values[0] if values else None

    stub = types.ModuleType("streamlit")
    stub.session_state = _SessionState()
    stub.secrets = {}
    stub.set_page_config = lambda *args, **kwargs: None
    stub.cache_data = lambda func=None, *args, **kwargs: (func if func else (lambda f: f))
    stub.cache_resource = lambda func=None, *args, **kwargs: (func if func else (lambda f: f))
    stub.error = lambda *args, **kwargs: None
    stub.warning = lambda *args, **kwargs: None
    stub.info = lambda *args, **kwargs: None
    stub.success = lambda *args, **kwargs: None
    stub.write = lambda *args, **kwargs: None
    stub.caption = lambda *args, **kwargs: None
    stub.markdown = lambda *args, **kwargs: None
    stub.text_input = lambda *args, **kwargs: ""
    stub.spinner = lambda *args, **kwargs: _Dummy()
    stub.file_uploader = lambda *args, **kwargs: None
    stub.radio = lambda *args, **kwargs: _first_option(kwargs.get("options", [None]))
    stub.empty = lambda *args, **kwargs: _Dummy()
    def _columns(*args, **kwargs):
        width = args[0] if args else 1
        if isinstance(width, (list, tuple)):
            count = len(width)
        else:
            count = int(width) if width is not None else 1
        return [_Dummy() for _ in range(count)]

    stub.columns = _columns
    stub.expander = lambda *args, **kwargs: _Dummy()
    stub.button = lambda *args, **kwargs: False
    stub.selectbox = lambda *args, **kwargs: _first_option(kwargs.get("options", [None]))
    stub.number_input = lambda *args, **kwargs: kwargs.get("value", kwargs.get("min_value", 0))
    stub.dataframe = lambda *args, **kwargs: None
    stub.multiselect = lambda *args, **kwargs: []
    stub.tabs = lambda *args, **kwargs: [_Dummy() for _ in range(len(args[0]))] if args else []
    stub.sidebar = _Dummy()
    stub.__getattr__ = lambda name: (lambda *args, **kwargs: None)
    return stub


def _install_streamlit_stub() -> types.ModuleType:
    stub = _create_streamlit_stub()
    existing = sys.modules.get("streamlit")
    if existing is not None:
        for name, value in vars(stub).items():
            if name.startswith("__"):
                continue
            setattr(existing, name, value)
    else:
        sys.modules["streamlit"] = stub
        existing = stub

    sortable_stub = types.ModuleType("streamlit_sortables")
    sortable_stub.sort_items = lambda *args, **kwargs: args[0] if args else None
    sys.modules.setdefault("streamlit_sortables", sortable_stub)
    return existing


def _load_module(module_name: str, file_path: Path, session_state_seed: dict[str, Any] | None = None):
    if not file_path.exists():
        raise FileNotFoundError(f"Missing required file: {file_path}")
    streamlit_mod = _install_streamlit_stub()
    if session_state_seed:
        try:
            streamlit_mod.session_state.update(session_state_seed)
        except Exception:
            pass
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_batch_helpers():
    if BATCH_HELPER_FILE.exists():
        batch_mod = _load_module("dataannot_batch", BATCH_HELPER_FILE)
        return batch_mod
    raise FileNotFoundError(f"Could not find batch helper: {BATCH_HELPER_FILE}")


def load_metadata_utils():
    return _load_module("dataannot_metadata_utils", METADATA_UTIL_FILE)


def load_qudt_helpers():
    return _load_module("dataannot_qudt_page", QUDT_PAGE_FILE)


def _normalize_link_list(raw_links: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    links: list[str] = []
    for item in raw_links:
        value = str(item).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        links.append(value)
    return links


def _read_links(path: str | Path) -> list[str]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Link list not found: {file_path}")
    return _normalize_link_list(file_path.read_text(encoding="utf-8").splitlines())


_BUCKET_TO_CONCEPT = {
    "Feature of Interest (FOI) - ID": "sosa:FeatureOfInterest",
    "Observed Property": "sosa:Property",
    "FOI - Spatial Information": "geo:Feature",
    "FOI - Attribute": "schema:Property",
    "Temporal": "sosa:phenomenonTime",
}

_SPATIAL_ROLE_TO_CONCEPT = {
    "X": "geo:long",
    "Y": "geo:lat",
    "Z": "geo:alt",
    "WKT geometry": "geo:asWKT",
    "BBOX": "geo:asWKT",
    "XY reference system": "geo:location",
    "Z reference system": "geo:alt",
}

_APP_TEMPORAL_DTYPES = {"date", "datetime", "time", "datetimestamp", "gday", "gyear", "gmonth", "gyearmonth", "gmonthday"}
_APP_SPATIAL_PATTERN = re.compile(
    r"\b(easting|longitude|lon|lng|long|x_coord|xcoord|x_pos|xpos|northing|latitude|lat|y_coord|ycoord|y_pos|ypos"
    r"|altitude|elevation|height|z_coord|zcoord|z_pos|zpos|crs|epsg|srid|proj|projection|reference.?system|coord.?ref"
    r"|geom|geometry|wkt|wkb)\b|(?<![a-z])[xyz](?![a-z])",
    re.IGNORECASE,
)
_APP_FOI_ID_PATTERN = re.compile(
    r"\b(id|ids|identifier|identifiers|site[._-]?id|sample[._-]?id|station[._-]?id|location[._-]?id|plot[._-]?id"
    r"|point[._-]?id|obs[._-]?id|observation[._-]?id|key|code|uuid|guid|pk|primary[._-]?key|ref|reference)\b",
    re.IGNORECASE,
)


def _load_sorting_module_for_batch(name: str | None, description: str | None, column_type: str | None):
    # Seed the minimum metadata expected by the page-level initialization.
    seed_df = pd.DataFrame(
        [
            {
                "name": str(name or ""),
                "column_type": str(column_type or "").lower(),
                "description": str(description or ""),
                "concept_type": "",
            }
        ]
    )
    seed_state = {
        "metadata_df": {"__batch__": seed_df},
        "AI_var_UoM": {},
        "AI_var_Methode": {},
        "column_buckets": {},
        "spatial_deepdive": {},
        "temporal_deepdive": {},
    }
    return _load_module(
        "dataannot_sorting_page_autopilot",
        SORTING_PAGE_FILE,
        session_state_seed=seed_state,
    )


def _guess_bucket_from_name(name: str | None, description: str | None = None, column_type: str | None = None) -> str:
    try:
        sorting_module = _load_sorting_module_for_batch(name, description, column_type)
    except Exception:
        sorting_module = None

    if sorting_module:
        info = {
            "name": str(name or "").lower(),
            "description": str(description or "").lower(),
            "column_type": str(column_type or "").lower(),
            "has_uom": bool(re.search(r"(mm|cm|m|km|kg|g|l|ml|mg|ug|pa|kpa|mpa|c|f|%|ppm|psu|ms/cm|s/m|g/cm3|g/m3|kg/m3|w/v|w/w)", str(name or "") + " " + str(description or ""), re.IGNORECASE)),
            "high_confidence_method": bool(_infer_method_hint(str(name or ""))),
            "concept_type": None,
        }
        if hasattr(sorting_module, "_preferred_bucket_for_column"):
            try:
                original_col_meta = getattr(sorting_module, "_col_meta", None)
                sorting_module._col_meta = {"__batch__": {str(name or ""): info}}
                bucket = sorting_module._preferred_bucket_for_column("__batch__", str(name or ""))
                if bucket:
                    return bucket
            except Exception:
                pass
            finally:
                if original_col_meta is None:
                    if hasattr(sorting_module, "_col_meta"):
                        delattr(sorting_module, "_col_meta")
                else:
                    sorting_module._col_meta = original_col_meta

        if hasattr(sorting_module, "_guess_bucket"):
            try:
                bucket = sorting_module._guess_bucket(info)
                if bucket:
                    return bucket
            except Exception:
                pass

    text_name = str(name or "")
    text_desc = str(description or "")
    text_joined = f"{text_name} {text_desc}".strip()

    if column_type and str(column_type).lower() in _APP_TEMPORAL_DTYPES:
        return "Temporal"
    if _APP_SPATIAL_PATTERN.search(text_joined):
        return "FOI - Spatial Information"
    if _APP_FOI_ID_PATTERN.search(text_joined):
        return "Feature of Interest (FOI) - ID"
    return "FOI - Attribute"


def _guess_concept_type_from_name(name: str | None, description: str | None = None, column_type: str | None = None) -> str:
    """Best-guess a concept URI using the same spatial deep-dive heuristics as the app."""
    try:
        sorting_module = _load_sorting_module_for_batch(name, description, column_type)
    except Exception:
        sorting_module = None

    if sorting_module:
        if hasattr(sorting_module, "_guess_spatial_role"):
            role = sorting_module._guess_spatial_role(str(name or ""))
            concept = _SPATIAL_ROLE_TO_CONCEPT.get(role)
            if concept:
                return concept

        if hasattr(sorting_module, "_preferred_bucket_for_column") or hasattr(sorting_module, "_guess_bucket"):
            info = {
                "name": str(name or "").lower(),
                "description": str(description or "").lower(),
                "column_type": str(column_type or "").lower(),
                "has_uom": bool(re.search(r"(mm|cm|m|km|kg|g|l|ml|mg|ug|pa|kpa|mpa|c|f|%|ppm|psu|ms/cm|s/m|g/cm3|g/m3|kg/m3|w/v|w/w)", str(name or "") + " " + str(description or ""), re.IGNORECASE)),
                "high_confidence_method": bool(_infer_method_hint(str(name or ""))),
                "concept_type": None,
            }
            if hasattr(sorting_module, "_preferred_bucket_for_column"):
                try:
                    original_col_meta = getattr(sorting_module, "_col_meta", None)
                    sorting_module._col_meta = {"__batch__": {str(name or ""): info}}
                    bucket = sorting_module._preferred_bucket_for_column("__batch__", str(name or ""))
                    if bucket:
                        concept = _BUCKET_TO_CONCEPT.get(bucket)
                        if concept:
                            return concept
                except Exception:
                    pass
                finally:
                    if original_col_meta is None:
                        if hasattr(sorting_module, "_col_meta"):
                            delattr(sorting_module, "_col_meta")
                    else:
                        sorting_module._col_meta = original_col_meta
            if hasattr(sorting_module, "_guess_bucket"):
                try:
                    bucket = sorting_module._guess_bucket(info)
                    concept = _BUCKET_TO_CONCEPT.get(bucket)
                    if concept:
                        return concept
                except Exception:
                    pass

    bucket = _guess_bucket_from_name(name, description, column_type)
    concept = _BUCKET_TO_CONCEPT.get(bucket)
    if concept:
        return concept

    text = " ".join(filter(None, [str(name or ""), str(description or "")])).lower()
    if re.search(r"\b(long|longitude|lon|easting|east|utm_x|xcoord|x_coordinate)\b", text):
        return "geo:long"
    if re.search(r"\b(lat|latitude|latit|northing|north|utm_y|ycoord|y_coordinate)\b", text):
        return "geo:lat"
    if re.search(r"\b(alt|altitude|elevation|height|depth)\b", text):
        return "geo:alt"
    if re.search(r"\b(wkt|geometry|bbox|polygon|boundary)\b", text):
        return "geo:asWKT"
    return "schema:Property"


def _heuristic_bucket_map(df: pd.DataFrame) -> dict[str, str]:
    buckets: dict[str, str] = {}
    column_types = {
        str(column): _infer_column_type_for_series(df[column])
        for column in df.columns
    }
    for column in list(df.columns):
        column_name = str(column)
        row = df[column] if isinstance(df, pd.DataFrame) else pd.Series(dtype=object)
        series = row.dropna().astype(str)
        sample = " ".join(series.head(10).tolist())
        if "name" in df.columns and column_name == "name" and not df.empty:
            continue
        buckets[column_name] = _guess_bucket_from_name(column_name, sample, column_types.get(column_name))

    # Match app behavior: keep only the leftmost FOI-ID candidate and demote
    # the rest to Unsorted.
    foi_bucket = "Feature of Interest (FOI) - ID"
    ordered_columns = [str(col) for col in list(df.columns) if str(col) in buckets]
    foi_candidates = [col for col in ordered_columns if buckets.get(col) == foi_bucket]
    if len(foi_candidates) > 1:
        for demoted in foi_candidates[1:]:
            buckets[demoted] = "Unsorted"
    return buckets


def _heuristic_concept_map(df: pd.DataFrame) -> dict[str, str]:
    concept_map: dict[str, str] = {}
    column_types = {
        str(column): _infer_column_type_for_series(df[column])
        for column in df.columns
    }
    for column in list(df.columns):
        column_name = str(column)
        row = df[column] if isinstance(df, pd.DataFrame) else pd.Series(dtype=object)
        series = row.dropna().astype(str)
        sample = " ".join(series.head(10).tolist())
        if "name" in df.columns and column_name == "name" and not df.empty:
            continue
        concept_map[column_name] = _guess_concept_type_from_name(column_name, sample, column_types.get(column_name))
    return concept_map


def _find_numeric_columns(df: pd.DataFrame, column_types: dict[str, str] | None = None) -> list[str]:
    if df.empty:
        return []
    if column_types is None:
        column_types = {
            str(column): _infer_column_type_for_series(df[column])
            for column in df.columns
        }
    candidates: list[str] = []
    for column in df.columns:
        if str(column).lower() in {"name", "description", "method", "reference", "unit_symbol", "unit_uri", "quantity_kind_uri"}:
            continue
        dtype = str(column_types.get(str(column), "")).lower()
        if dtype in {"integer", "float", "decimal", "number", "numeric"}:
            candidates.append(str(column))
    return candidates


def _suggest_unit_candidates(df: pd.DataFrame) -> dict[str, list[str]]:
    suggestions: dict[str, list[str]] = {}
    for column in df.columns:
        label = str(column).lower()
        if not any(token in label for token in ["unit", "uom", "kind", "temp", "moisture", "ph", "ec", "depth", "mass", "value"]):
            continue
        if "unit" in label or "uom" in label:
            suggestions[str(column)] = ["QUDT unit search recommended"]
        elif "kind" in label:
            suggestions[str(column)] = ["QUDT quantity-kind search recommended"]
    return suggestions


def _is_likely_codelist(values_list: list[str]) -> bool:
    values_series = pd.Series(values_list)
    if values_series.empty:
        return False

    n_unique = values_series.nunique()
    if n_unique == 0:
        return False

    unique_ratio = n_unique / len(values_series)
    top_frequency = values_series.value_counts(normalize=True).iloc[0]
    p = values_series.value_counts(normalize=True)
    entropy_val = -(p * np.log(p)).sum() if not p.empty else 0.0
    entropy_norm = entropy_val / np.log(len(p)) if len(p) > 1 else 0.0

    score = 0
    if unique_ratio < 0.05:
        score += 3
    elif unique_ratio < 0.2:
        score += 2
    elif unique_ratio > 0.8:
        score -= 5

    if n_unique < 50:
        score += 2

    if top_frequency > 0.05:
        score += 2

    avg_length = values_series.str.len().mean()
    if avg_length > 30:
        score -= 1

    if entropy_norm < 0.5:
        score += 2
    elif entropy_norm > 0.9:
        score -= 2
    else:
        score += 1

    return score > 3


def _infer_column_type_for_series(s: pd.Series, sample_size: int = 200) -> str:
    try:
        input_module = _load_module("dataannot_input_page_autopilot", INPUT_PAGE_FILE)
    except Exception:
        input_module = None

    if input_module and hasattr(input_module, "detect_csvw_datatype_from_series"):
        detected = input_module.detect_csvw_datatype_from_series(s, sample_size=sample_size)
        if detected:
            return str(detected)

    series = s.dropna().astype(str).str.strip()
    if len(series) == 0:
        return "string"
    series = series.head(sample_size)

    counts = {"integer": 0, "float": 0, "dateTime": 0, "date": 0, "time": 0, "boolean": 0, "json": 0}
    string_values: list[str] = []
    total = 0

    for val in series:
        if val == "":
            continue
        total += 1
        clean_val = val.replace(",", ".")

        try:
            value = float(clean_val)
        except ValueError:
            value = None

        if value is not None and float(value).is_integer():
            counts["integer"] += 1
            continue
        if value is not None:
            counts["float"] += 1
            continue

        lowered = val.lower().strip()
        if lowered in {"true", "false", "1", "0"}:
            counts["boolean"] += 1
            continue

        if val.strip().startswith(("{", "[")):
            try:
                json.loads(val)
                counts["json"] += 1
                continue
            except (TypeError, ValueError, json.JSONDecodeError):
                pass

        if any(sep in val for sep in ["/", "-", ".", ":"]):
            try:
                parsed = pd.to_datetime(val)
                if parsed.hour != 0 or parsed.minute != 0 or parsed.second != 0:
                    counts["dateTime"] += 1
                else:
                    counts["date"] += 1
                continue
            except Exception:
                pass

            try:
                pd.to_datetime(val, format="%H:%M:%S")
                counts["time"] += 1
                continue
            except Exception:
                pass

        string_values.append(val)

    if total == 0:
        return "string"

    threshold = 0.8
    for dtype in ("integer", "float", "boolean", "dateTime", "date", "time", "json"):
        if counts[dtype] / total >= threshold:
            return dtype

    if string_values and len(string_values) / total >= threshold and _is_likely_codelist(string_values):
        return "codelist"
    return "string"


def _clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return text


def _resolve_variable_description(df: pd.DataFrame, variable_name: str) -> str:
    clean_name = _clean_text(variable_name)
    if not clean_name:
        return ""

    if "name" in df.columns and "description" in df.columns:
        name_matches = df["name"].map(_clean_text) == clean_name
        if name_matches.any():
            matched = _clean_text(df.loc[name_matches, "description"].iloc[0])
            if matched:
                return matched

    return ""


def _infer_description_from_metadata(row: pd.Series) -> str:
    candidates = [
        row.get("description"),
        row.get("concept"),
        row.get("concept_type"),
        row.get("name"),
    ]
    for candidate in candidates:
        text = _clean_text(candidate)
        if text:
            return text
    return "No description available in the normalized metadata."


def _infer_unit_symbol(name: str) -> str:
    lowered = name.lower()
    if any(token in lowered for token in ["ph", "acidity", "alkalinity"]):
        return "pH"
    if "temp" in lowered or "temperature" in lowered:
        return "°C"
    if "moisture" in lowered:
        return "%"
    if "carbon" in lowered or "organic" in lowered:
        return "%"
    if "mass" in lowered or "weight" in lowered:
        return "kg"
    if "depth" in lowered:
        return "cm"
    if "conductivity" in lowered or "ec" in lowered:
        return "dS/m"
    if "elevation" in lowered or "altitude" in lowered:
        return "m"
    return ""


def _infer_method_hint(name: str) -> str:
    lowered = name.lower()
    if any(token in lowered for token in ["ph", "acidity", "alkalinity"]):
        return "ISO 10390:2005 - Soil quality - Determination of pH"
    if "organic carbon" in lowered or "soc" in lowered:
        return "ISO 10694:1995 - Soil quality - Determination of organic carbon"
    if "moisture" in lowered or "water content" in lowered:
        return "ISO 11465:1993 - Soil quality - Determination of dry matter and water content"
    if "bulk density" in lowered:
        return "ISO 11272:2017 - Soil quality - Determination of dry bulk density"
    if "texture" in lowered:
        return "ISO 11277:2009 - Soil quality - Determination of particle size distribution"
    if "phosph" in lowered:
        return "ISO 6878:2004 - Soil quality - Determination of phosphorus"
    if "nitrogen" in lowered:
        return "ISO 11261:1995 - Soil quality - Determination of total nitrogen"
    return ""


def _infer_relationships(normalized: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    table_keys = [key for key in normalized if isinstance(normalized[key], pd.DataFrame)]
    relationships: list[dict[str, Any]] = []
    for left_idx, left_key in enumerate(table_keys):
        for right_key in table_keys[left_idx + 1:]:
            left_df = normalized[left_key]
            right_df = normalized[right_key]
            left_names = [str(v).strip().lower() for v in (left_df["name"].tolist() if "name" in left_df.columns else left_df.columns)]
            right_names = [str(v).strip().lower() for v in (right_df["name"].tolist() if "name" in right_df.columns else right_df.columns)]
            left_candidates = [name for name in left_names if any(token in name for token in ["id", "site", "plot", "sample", "profile", "location", "station"]) or name.endswith("_id")]
            right_candidates = [name for name in right_names if any(token in name for token in ["id", "site", "plot", "sample", "profile", "location", "station"]) or name.endswith("_id")]
            left_id = left_candidates[0] if left_candidates else ""
            right_id = right_candidates[0] if right_candidates else ""
            if left_id and right_id:
                relationships.append({
                    "left_table": left_key,
                    "right_table": right_key,
                    "relation": "many-to-one",
                    "left_id": left_id,
                    "right_id": right_id,
                    "score": "heuristic",
                })
    return relationships


def build_record_summary(table_name: str, df: pd.DataFrame) -> dict[str, Any]:
    normalized_df = df.copy()
    if "name" in normalized_df.columns:
        names = [str(v) for v in normalized_df["name"].dropna().tolist() if str(v).strip()]
    else:
        names = list(normalized_df.columns)

    column_types = {
        str(column): _infer_column_type_for_series(normalized_df[column])
        for column in normalized_df.columns
    }

    summary = {
        "table_name": table_name,
        "shape": list(normalized_df.shape),
        "columns": list(normalized_df.columns),
        "column_types": column_types,
        "bucket_map": _heuristic_bucket_map(normalized_df),
        "concept_map": _heuristic_concept_map(normalized_df),
        "numeric_columns": _find_numeric_columns(normalized_df, column_types),
        "unit_candidates": _suggest_unit_candidates(normalized_df),
        "variable_names": names[:20],
    }
    return summary


def _is_uom_signal_for_sorting(value: str | None) -> bool:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return False
    if normalized in {"string", "categorical", "identifier", "unknown", "none", "n/a", "na"}:
        return False
    return True


def _build_sorting_metadata_df(
    df: pd.DataFrame,
    description_rows: list[dict[str, str]] | None = None,
) -> pd.DataFrame:
    description_by_name = {
        str(row.get("name", "")).strip(): _clean_text(row.get("description"))
        for row in (description_rows or [])
        if str(row.get("name", "")).strip()
    }
    rows: list[dict[str, str]] = []
    for column in list(df.columns):
        column_name = str(column)
        rows.append(
            {
                "name": column_name,
                "column_type": _infer_column_type_for_series(df[column]),
                "description": description_by_name.get(column_name, ""),
                "concept_type": "",
            }
        )
    return pd.DataFrame(rows)


def _build_bucket_map_with_sorting_page(
    table_name: str,
    df: pd.DataFrame,
    description_rows: list[dict[str, str]] | None = None,
    uom_rows: list[dict[str, str]] | None = None,
    method_rows: list[dict[str, str]] | None = None,
) -> dict[str, str]:
    metadata_df = _build_sorting_metadata_df(df, description_rows=description_rows)

    ai_uom_df = pd.DataFrame(
        [
            {
                "name": str(row.get("name", "")).strip(),
                "UoM": str(row.get("unit_symbol", "")).strip(),
            }
            for row in (uom_rows or [])
            if str(row.get("name", "")).strip() and _is_uom_signal_for_sorting(row.get("unit_symbol"))
        ]
    )
    ai_method_df = pd.DataFrame(
        [
            {
                "name": str(row.get("name", "")).strip(),
                "confidence": "",
            }
            for row in (method_rows or [])
            if str(row.get("name", "")).strip() and _clean_text(row.get("method"))
        ]
    )

    seed_state = {
        "metadata_df": {table_name: metadata_df},
        "AI_var_UoM": {table_name: ai_uom_df},
        "AI_var_Methode": {table_name: ai_method_df},
        "column_buckets": {},
        "spatial_deepdive": {},
        "temporal_deepdive": {},
    }

    try:
        sorting_module = _load_module(
            "dataannot_sorting_page_summary",
            SORTING_PAGE_FILE,
            session_state_seed=seed_state,
        )
        bucket_state = sorting_module.st.session_state.get("column_buckets", {}).get(table_name, {})
        bucket_map: dict[str, str] = {}
        for bucket_name, bucket_columns in bucket_state.items():
            for bucket_column in bucket_columns:
                bucket_map[str(bucket_column)] = str(bucket_name)
        if bucket_map:
            return bucket_map
    except Exception:
        pass

    return _heuristic_bucket_map(df)


def _concept_map_from_bucket_map(bucket_map: dict[str, str]) -> dict[str, str]:
    concept_map: dict[str, str] = {}
    for column_name, bucket_name in bucket_map.items():
        concept_map[str(column_name)] = _BUCKET_TO_CONCEPT.get(str(bucket_name), "")
    return concept_map


def build_human_in_the_loop_guide(metadata_dict: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Return a structured summary of the app workflow as a best-guess human review guide.

    This mirrors the page-by-page flow of the Streamlit app, but keeps the output in a
    machine-readable form that can be surfaced by the batch autopilot script for review.
    """
    workflow = [
        {
            "page": "Input",
            "goal": "Load the source tables and confirm the dataset is the intended one.",
            "human_action": "Review the file names, table boundaries, and source context before moving on.",
            "best_guess": "The uploaded files are the likely starting point; the workflow assumes the user has confirmed them.",
            "status": "provisional",
            "confidence": "medium",
        },
        {
            "page": "Context",
            "goal": "Derive descriptions and variable semantics from the dataset plus any supplied context.",
            "human_action": "Check the inferred description for each column and keep or correct any values that sound speculative.",
            "best_guess": "Variable meaning is inferred from names, descriptions, and nearby scientific conventions.",
            "status": "provisional",
            "confidence": "medium",
        },
        {
            "page": "Column Sorting",
            "goal": "Place every column into a soil-data role such as FOI, spatial, attribute, or temporal.",
            "human_action": "Validate the bucket assignment; reclassify any columns whose role is ambiguous.",
            "best_guess": "Names like lat, lon, site_id, temperature, and date are the most reliable bucket clues.",
            "status": "provisional",
            "confidence": "medium",
        },
        {
            "page": "Unit of measurement",
            "goal": "Identify candidate units and quantity kinds for numeric variables.",
            "human_action": "Review the proposed unit symbols and quantity kinds, especially for ambiguous measurements.",
            "best_guess": "Numeric columns are treated as measurement-like unless they are clearly identifiers or codes.",
            "status": "provisional",
            "confidence": "medium",
        },
        {
            "page": "Method",
            "goal": "Associate variables with a canonical method or standard when one is explicitly or implicitly implied.",
            "human_action": "Approve or replace the suggested method reference before exporting metadata.",
            "best_guess": "Only standards or established protocols are preferred; vague lab prose should be treated cautiously.",
            "status": "provisional",
            "confidence": "low",
        },
        {
            "page": "Linking tables",
            "goal": "Infer relationships between tables and identify primary keys or joinable identifiers.",
            "human_action": "Confirm table joins and relationship labels, since the raw heuristic is only a suggestion.",
            "best_guess": "Identifier-like columns (site_id, sample_id, profile_id, pk, location_id) are the strongest join signals.",
            "status": "provisional",
            "confidence": "medium",
        },
        {
            "page": "Export",
            "goal": "Bundle approved metadata, relationships, and table summaries into the final metadata package.",
            "human_action": "Review the final package before release; the bot should not silently finalize uncertain guesses.",
            "best_guess": "The export is strongest when the final user reviews and edits the proposed metadata rather than accepting it all as fact.",
            "status": "provisional",
            "confidence": "medium",
        },
    ]

    tables: dict[str, Any] = {}
    review_queue: list[dict[str, Any]] = []
    for table_name, df in metadata_dict.items():
        if not isinstance(df, pd.DataFrame):
            continue
        summary = build_record_summary(table_name, df)
        name_candidates = [str(value).strip() for value in (df["name"].tolist() if "name" in df.columns else df.columns) if str(value).strip()]
        first_three = name_candidates[:3]
        table_review = []
        for column in list(df.columns):
            name = str(column)
            reasons: list[str] = []
            if any(token in name.lower() for token in ["id", "site", "sample", "profile", "location", "station"]) and "name" not in name.lower():
                reasons.append("identifier-like column name")
            if summary["bucket_map"].get(column) in {"FOI - Attribute", "Observed Property"} and column in summary["numeric_columns"]:
                reasons.append("numeric field that may need a unit or quantity-kind check")
            if summary["unit_candidates"].get(column) in ({}, [], None):
                reasons.append("no reliable unit candidate inferred")
            if reasons:
                table_review.append({
                    "column": name,
                    "reasons": reasons,
                    "status": "provisional",
                    "confidence": "low" if len(reasons) > 1 else "medium",
                    "human_action": "Confirm the inferred meaning or unit before treating this as final metadata.",
                })
        if table_review:
            review_queue.append({
                "table": table_name,
                "items": table_review,
            })

        table_guide = {
            "status": "provisional",
            "confidence": "medium",
            "columns": list(df.columns),
            "column_types": {
                str(column): _infer_column_type_for_series(df[column])
                for column in df.columns
            },
            "context": {
                "best_guess": {
                    "likely_variable_names": summary["variable_names"],
                    "bucket_map": summary["bucket_map"],
                    "concept_map": summary.get("concept_map", {}),
                    "numeric_columns": summary["numeric_columns"],
                    "unit_candidates": summary["unit_candidates"],
                },
                "human_action": "Check whether the inferred bucket and unit suggestions match the actual scientific meaning of each variable.",
                "status": "provisional",
            },
            "sorting": {
                "best_guess": {column: summary["bucket_map"].get(column, "FOI - Attribute") for column in df.columns[:20]},
                "human_action": "Reclassify columns when the names are ambiguous or the dataset includes mixed semantics.",
                "status": "provisional",
            },
            "unit_measurement": {
                "best_guess": {column: summary["unit_candidates"].get(column, []) for column in df.columns[:20]},
                "human_action": "Confirm the proposed units and quantity kinds for all numeric or measurement-oriented variables.",
                "status": "provisional",
            },
            "method": {
                "best_guess": {
                    "sample_variables": first_three,
                    "notes": "The method stage should use context and standards to prefer explicit ISO/CEN/DOI evidence over generic lab language.",
                },
                "human_action": "Accept only methods that are explicitly justified by the supplied context or a known standard.",
                "status": "provisional",
            },
            "linking": {
                "best_guess": {
                    "candidate_identifiers": [name for name in name_candidates if any(token in name.lower() for token in ["id", "site", "sample", "profile", "location", "station"])],
                },
                "human_action": "Check that the inferred joins are real relationships rather than coincidental column names.",
                "status": "provisional",
            },
            "needs_human_review": table_review,
        }
        tables[table_name] = table_guide

    flattened_review_queue = [item for item in review_queue if item]
    return {
        "workflow": workflow,
        "tables": tables,
        "needs_human_review": flattened_review_queue,
        "summary": "This is a best-guess human-in-the-loop workflow: the script proposes likely metadata, but the human reviews and corrects each stage before export.",
        "status": "provisional",
    }


def _safe_table_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "metadata_table"


def _read_streamlit_secrets() -> dict[str, Any]:
    candidates = [
        ROOT / ".streamlit" / "secrets.toml",
        Path(".streamlit") / "secrets.toml",
        Path.home() / ".streamlit" / "secrets.toml",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            with path.open("rb") as fh:
                payload = tomllib.load(fh)
            if isinstance(payload, dict):
                return payload
        except Exception:
            continue
    return {}


def _resolve_openai_api_key() -> str | None:
    env_value = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY_1") or os.getenv("OPENAI_API_KEY_2")
    if env_value:
        return env_value
    secrets = _read_streamlit_secrets()
    if isinstance(secrets, dict):
        openai_cfg = secrets.get("openai", {})
        if isinstance(openai_cfg, dict):
            api_key = openai_cfg.get("api_key")
            if api_key:
                return str(api_key)
    return None


def _build_context_prompt_for_table(table_name: str, df: pd.DataFrame, extra_context: str = "") -> str:
    names = []
    if "name" in df.columns:
        names = [str(v).strip() for v in df["name"].dropna().tolist() if str(v).strip()]
    if not names:
        names = [str(v).strip() for v in df.columns if str(v).strip()]

    description_hints = []
    if "description" in df.columns:
        values = [str(v).strip() for v in df["description"].dropna().tolist() if str(v).strip()]
        if values:
            description_hints.append("Existing descriptions: " + "; ".join(values[:15]))

    lines = [
        f"Dataset table: {table_name}",
        "This is a metadata table for a soil or environmental dataset.",
        "Column names: " + ", ".join(names[:40]),
    ]
    if description_hints:
        lines.extend(description_hints)
    if extra_context.strip():
        lines.append("Additional context provided by the user: ")
        lines.append(extra_context.strip())
    return "\n".join(lines)


def _compose_context_text(metadata_dict: dict[str, pd.DataFrame], extra_context: str = "") -> str:
    pieces: list[str] = []
    for table_name, df in metadata_dict.items():
        if not isinstance(df, pd.DataFrame):
            continue
        names = [str(v).strip() for v in (df["name"].tolist() if "name" in df.columns else df.columns) if str(v).strip()]
        if not names:
            continue
        pieces.append(_build_context_prompt_for_table(table_name, df, extra_context=""))
    combined = "\n\n".join(pieces)
    if extra_context.strip():
        combined = f"{combined}\n\n{extra_context.strip()}" if combined else extra_context.strip()
    return combined


def _collect_context_text_for_record_url(record_url: str) -> str:
    """Reuse the app’s context-discovery flow for a Zenodo record URL.

    This mirrors the Input + Context pages: fetch the record metadata, gather all
    candidate context files from the record, load each file via the app helper, and
    extract text using the same reader logic as the app before composing the final
    batch LLM context.
    """
    if not record_url or not str(record_url).strip():
        return ""

    pieces: list[str] = []
    try:
        input_module = _load_module("dataannot_input_page_autopilot", INPUT_PAGE_FILE)
    except Exception:
        input_module = None

    try:
        context_module = _load_module("dataannot_context_page_autopilot", CONTEXT_PAGE_FILE)
    except Exception:
        context_module = None

    if input_module is None:
        return ""

    try:
        record_id = input_module.get_record_id_from_Zenodo_url(str(record_url))
    except Exception:
        return ""

    metadata_context: dict[str, Any] = {}
    try:
        metadata_context = input_module.get_metadata_from_Zenodo_id(record_id) or {}
    except Exception:
        metadata_context = {}

    if isinstance(metadata_context, dict):
        title = str(metadata_context.get("title") or "").strip()
        description = metadata_context.get("description")
        if isinstance(description, str):
            try:
                description = input_module.description_to_plain_text(description)
            except Exception:
                pass
            description = str(description).strip()
        elif not isinstance(description, str):
            description = ""
        if title:
            pieces.append(f"Zenodo title: {title}")
        if description:
            pieces.append(f"Zenodo description: {description}")

    context_urls: list[str] = []
    try:
        context_urls = list(input_module.get_files_URL_from_Zenodo_id(record_id, extensions=[".doc", ".docx", ".pdf", ".md", ".txt"]))
    except Exception:
        context_urls = []

    seen: set[str] = set()
    for url in context_urls:
        file_name = None
        try:
            file_name = input_module.filename_from_url(url)
        except Exception:
            file_name = os.path.basename(str(url))

        if not file_name or file_name in seen:
            continue
        seen.add(file_name)

        try:
            if context_module is not None and hasattr(context_module, "load_file_from_url"):
                context_file = context_module.load_file_from_url(url)
            else:
                response, name, ext = input_module.request_file_from_zenodo(url)
                if response is None:
                    continue
                context_file = context_module.ContextFile(
                    name=name or file_name,
                    content=getattr(response, "content", b""),
                    mime_type=(getattr(response, "headers", {}) or {}).get("Content-Type", "application/octet-stream"),
                    source="url",
                    ext=str(ext or os.path.splitext(name or file_name)[1]).lstrip("."),
                ) if context_module is not None else None
                if context_file is None:
                    continue
        except Exception:
            continue

        try:
            if context_module is not None and hasattr(context_module, "read_context_file"):
                extracted = context_module.read_context_file(context_file)
            else:
                extracted = ""
        except Exception:
            extracted = ""

        if extracted and str(extracted).strip():
            pieces.append(f"Context file: {getattr(context_file, 'name', file_name)}\n{str(extracted).strip()}")

    combined = "\n\n".join(piece for piece in pieces if piece and str(piece).strip())
    return combined


class _NoopFeedback:
    def info(self, *args, **kwargs):
        return None

    def success(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None


def _prepare_context_page_llm(context_module: Any, api_key: str | None) -> None:
    context_module.provider_selected = "OpenAI"
    context_module.LLM_selection_info = {
        "OpenAI": {
            "key": api_key,
            "callfunction": context_module.get_response_OpenAI,
            "help_key_creation": "https://platform.openai.com/account/api-keys",
        }
    }
    context_module.st.session_state["api_key"] = api_key
    if api_key:
        context_module.st.session_state["openai_client"] = context_module.create_openai_client(api_key)
    else:
        context_module.st.session_state.pop("openai_client", None)


def _generate_llm_descriptions_for_metadata(
    metadata_dict: dict[str, pd.DataFrame],
    api_key: str | None = None,
    context_text: str = "",
    debug_feedback: dict[str, Any] | None = None,
) -> dict[str, list[dict[str, str]]]:
    results: dict[str, list[dict[str, str]]] = {}

    def _record_debug(table_name: str, df: pd.DataFrame, **extra: Any) -> None:
        if debug_feedback is None:
            return
        table_context = _build_context_prompt_for_table(table_name, df, extra_context=context_text)
        debug_feedback[table_name] = {
            "prompt_context_preview": table_context[:300],
            **extra,
        }

    try:
        context_module = _load_module("dataannot_context_page_autopilot", CONTEXT_PAGE_FILE)
    except Exception as exc:
        for table_name, df in metadata_dict.items():
            if isinstance(df, pd.DataFrame):
                _record_debug(
                    table_name,
                    df,
                    status="skipped",
                    reason="context_module_load_failed",
                    error=str(exc),
                    llm_result_type="not_loaded",
                    llm_result_keys=[],
                    resolved_rows=[],
                )
        return results

    _prepare_context_page_llm(context_module, api_key)

    for table_name, df in metadata_dict.items():
        if not isinstance(df, pd.DataFrame):
            continue
        names = [str(v).strip() for v in (df["name"].tolist() if "name" in df.columns else df.columns) if str(v).strip()]
        if not names:
            _record_debug(table_name, df, status="skipped", reason="no_named_columns", llm_result_type="none", llm_result_keys=[], resolved_rows=[])
            continue

        table_context = _build_context_prompt_for_table(table_name, df, extra_context=context_text)
        human_description = {}
        if "description" in df.columns:
            if "name" in df.columns:
                source_names = df["name"].tolist()
            else:
                source_names = [str(col) for col in df.columns]
            for name, value in zip(source_names, df["description"].tolist()):
                clean_name = str(name).strip()
                clean_value = _clean_text(value)
                if clean_name and clean_value:
                    human_description[clean_name] = clean_value

        llm_result: Any = {}
        try:
            llm_result = context_module.generate_descriptions_with_LLM(
                var_list=names,
                context=table_context,
                human_description=human_description,
                cache_feedback=_NoopFeedback(),
                tablename=table_name,
            )
        except Exception as exc:
            llm_result = {}
            if debug_feedback is not None:
                debug_feedback[table_name] = {
                    "prompt_context_preview": table_context[:300],
                    "full_context_text": table_context,
                    "status": "error",
                    "reason": "llm_generation_failed",
                    "error": str(exc),
                    "llm_result_type": type(llm_result).__name__,
                    "llm_result_keys": [],
                    "resolved_rows": [],
                }

        rows: list[dict[str, str]] = []
        per_variable_debug: list[dict[str, str]] = []
        for name in names:
            clean_name = _clean_text(name)
            if not clean_name:
                continue
            description = _resolve_variable_description(df, clean_name)
            source = "metadata" if description else "empty"
            if isinstance(llm_result, dict):
                for candidate_key, candidate_value in llm_result.items():
                    if str(candidate_key).strip() == clean_name:
                        if isinstance(candidate_value, dict):
                            llm_description = _clean_text(candidate_value.get("description") or candidate_value.get("definition") or candidate_value.get("text"))
                        else:
                            llm_description = _clean_text(candidate_value)
                        if llm_description:
                            description = llm_description
                            source = "llm"
                        break
            rows.append({"name": clean_name, "description": description})
            per_variable_debug.append(
                {
                    "name": clean_name,
                    "source": source,
                    "description_preview": description[:160],
                }
            )
        results[table_name] = rows

        if debug_feedback is not None:
            llm_keys = []
            if isinstance(llm_result, dict):
                llm_keys = [str(key) for key in llm_result.keys()][:60]
            debug_feedback[table_name] = {
                "prompt_context_preview": table_context[:300],
                "full_context_text": table_context,
                "status": "ok" if llm_result else "no_result",
                "llm_result_type": type(llm_result).__name__,
                "llm_result_keys": llm_keys,
                "resolved_rows": per_variable_debug,
            }

    return results


def _generate_llm_uom_for_metadata(metadata_dict: dict[str, pd.DataFrame], api_key: str | None = None, context_text: str = "") -> dict[str, list[dict[str, str]]]:
    results: dict[str, list[dict[str, str]]] = {}
    try:
        context_module = _load_module("dataannot_context_page_autopilot", CONTEXT_PAGE_FILE)
    except Exception:
        return results

    _prepare_context_page_llm(context_module, api_key)
    for table_name, df in metadata_dict.items():
        if not isinstance(df, pd.DataFrame):
            continue
        names = [str(v).strip() for v in (df["name"].tolist() if "name" in df.columns else df.columns) if str(v).strip()]
        if not names:
            continue
        try:
            llm_result = context_module.generate_UoM_with_LLM(
                var_list=names,
                context=_build_context_prompt_for_table(table_name, df, extra_context=context_text),
                cache_feedback=_NoopFeedback(),
                tablename=table_name,
            )
        except Exception:
            llm_result = {}
        rows: list[dict[str, str]] = []
        for name in names:
            row_result = llm_result.get(name, {}) if isinstance(llm_result, dict) else {}
            if isinstance(row_result, dict):
                best_guess = row_result.get("UoM - Best guess") or row_result.get("UoM kind - Best guess") or ""
                rows.append({"name": name, "unit_symbol": str(best_guess)})
            else:
                rows.append({"name": name, "unit_symbol": ""})
        results[table_name] = rows
    return results


def _generate_llm_method_for_metadata(metadata_dict: dict[str, pd.DataFrame], api_key: str | None = None, context_text: str = "") -> dict[str, list[dict[str, str]]]:
    results: dict[str, list[dict[str, str]]] = {}
    try:
        context_module = _load_module("dataannot_context_page_autopilot", CONTEXT_PAGE_FILE)
    except Exception:
        return results

    _prepare_context_page_llm(context_module, api_key)
    for table_name, df in metadata_dict.items():
        if not isinstance(df, pd.DataFrame):
            continue
        names = [str(v).strip() for v in (df["name"].tolist() if "name" in df.columns else df.columns) if str(v).strip()]
        if not names:
            continue
        try:
            llm_result = context_module.generate_methode_with_LLM(
                var_list=names,
                context=_build_context_prompt_for_table(table_name, df, extra_context=context_text),
                cache_feedback=_NoopFeedback(),
                tablename=table_name,
            )
        except Exception:
            llm_result = {}
        rows: list[dict[str, str]] = []
        for name in names:
            row_result = llm_result.get(name, {}) if isinstance(llm_result, dict) else {}
            if isinstance(row_result, dict):
                method_name = row_result.get("method_name") or row_result.get("reference") or ""
                rows.append({"name": name, "method": str(method_name)})
            else:
                rows.append({"name": name, "method": ""})
        results[table_name] = rows
    return results


def _match_keyword_candidates_direct(
    candidate_df: pd.DataFrame,
    index_lib: Any,
    vocab_dict: dict[int, dict[str, Any]],
    model: Any,
    source_filter: set[str] | None = None,
    k_nearest: int = 50,
) -> dict[str, list[dict[str, Any]]]:
    """Batch-safe equivalent of the app matcher that does not depend on UI globals."""
    matches: dict[str, list[dict[str, Any]]] = {}
    if candidate_df.empty or index_lib is None or model is None or not isinstance(vocab_dict, dict):
        return matches

    for _, row in candidate_df.iterrows():
        query = _clean_text(row.get("name", ""))
        definition = _clean_text(row.get("description", ""))
        if not query:
            continue

        emb_word = model.encode(query, normalize_embeddings=True)
        if definition and definition.lower() != "nan":
            emb_def = model.encode(definition, normalize_embeddings=True)
            vec = 0.6 * emb_word + 0.4 * emb_def
        else:
            vec = emb_word
        vec = np.asarray(vec, dtype="float32").reshape(1, -1)

        D_match, I_match = index_lib.search(vec, k=k_nearest)
        indices = I_match.flatten()
        distances = D_match.flatten()

        for idx, dist in zip(indices, distances):
            if idx not in vocab_dict:
                continue
            item = dict(vocab_dict[idx])
            item.setdefault("uri", item.get("concept_uri") or item.get("uri") or "")
            item.setdefault("source", item.get("source_id") or item.get("source") or "")
            item.setdefault("QC_label", item.get("label_type") or item.get("QC_label") or "")
            item.setdefault("definition", item.get("definition_en") or item.get("definition") or "")
            source_value = str(item.get("source", "")).strip().removeprefix("local_vocabularies:")
            item["source"] = source_value
            if source_filter is not None and source_value not in source_filter:
                continue
            item.update({
                "query": query,
                "definition": definition,
                "Distance": float(1.0 - float(dist)),
            })
            matches.setdefault(query, []).append(item)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for query_name, query_hits in matches.items():
        ordered = sorted(query_hits, key=lambda hit: float(hit.get("Distance", 1.0)))
        seen: set[str] = set()
        deduped: list[dict[str, Any]] = []
        for hit in ordered:
            uri = str(hit.get("uri") or "")
            if not uri or uri in seen:
                continue
            seen.add(uri)
            deduped.append(hit)
        grouped[query_name] = deduped
    return grouped


def _generate_keyword_matches_for_metadata(
    metadata_dict: dict[str, pd.DataFrame],
    modelname: str = "models_local/all-MiniLM-L6-v2",
    source_filter: set[str] | None = None,
    k_nearest: int = 50,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Reuse the app’s keyword matcher loader and nearest-neighbor search for batch export."""
    results: dict[str, dict[str, list[dict[str, Any]]]] = {}
    try:
        matcher_module = _load_module("dataannot_keyword_matcher_autopilot", KEYWORD_MATCHER_FILE)
    except Exception:
        matcher_module = None

    index_lib = None
    vocab_dict = None
    model = None
    if matcher_module is not None:
        try:
            index_lib, vocab_dict = matcher_module.load_vocab_indexes(modelname=modelname)
        except Exception:
            index_lib, vocab_dict = None, None

        if index_lib is not None and vocab_dict is not None:
            try:
                if not hasattr(matcher_module, "model") or matcher_module.model is None:
                    if hasattr(matcher_module, "load_sentence_model"):
                        matcher_module.model = matcher_module.load_sentence_model(modelname=modelname)
                model = getattr(matcher_module, "model", None)
            except Exception:
                model = None

    if index_lib is None or vocab_dict is None or model is None:
        try:
            import faiss
            from sentence_transformers import SentenceTransformer

            index_lib = faiss.read_index(f"{ROOT / 'data' / ('vocabulary-' + modelname.replace('/', '--') + '.index')}")
            metadata_lib = np.load(f"{ROOT / 'data' / ('vocabulary-' + modelname.replace('/', '--') + '-meta.npz')}", allow_pickle=True)
            raw_metadata = metadata_lib["id_to_meta"].item()
            vocab_dict = {}
            for identifier, item in raw_metadata.items():
                metadata = dict(item)
                metadata.setdefault("uri", metadata.get("concept_uri"))
                metadata.setdefault("source", metadata.get("source_id"))
                metadata.setdefault("QC_label", metadata.get("label_type"))
                metadata.setdefault("definition", metadata.get("definition_en"))
                metadata["source"] = str(metadata.get("source") or "").removeprefix("local_vocabularies:")
                vocab_dict[int(identifier)] = metadata
            model = SentenceTransformer(modelname)
        except Exception:
            index_lib = None
            vocab_dict = {}
            model = None

    for table_name, df in metadata_dict.items():
        if not isinstance(df, pd.DataFrame):
            continue
        if "name" in df.columns and "description" in df.columns:
            candidate_df = df[["name", "description"]].copy()
        elif "name" in df.columns:
            candidate_df = df[["name"]].copy()
            candidate_df["description"] = ""
        else:
            candidate_df = pd.DataFrame({"name": list(df.columns), "description": [""] * len(df.columns)})

        candidate_df = candidate_df.fillna("")
        candidate_df["name"] = candidate_df["name"].map(lambda value: _clean_text(value))
        candidate_df["description"] = candidate_df["description"].map(lambda value: _clean_text(value))
        candidate_df = candidate_df[candidate_df["name"].astype(bool)]

        if "concept_type" in df.columns:
            excluded = {"geo:Feature", "sosa:phenomenonTime", "sosa:resultTime"}
            concept_type_values = df["concept_type"].fillna("").astype(str).str.strip()
            in_excluded_type = concept_type_values.isin(excluded)
            concept_values = df.get("concept", pd.Series("", index=df.index)).fillna("").astype(str).str.strip()
            concept_uri_values = df.get("concept_uri", pd.Series("", index=df.index)).fillna("").astype(str).str.strip()
            empty_tokens = {"", "none", "null", "nan"}
            has_concept_info = (~concept_values.str.lower().isin(empty_tokens)) | (~concept_uri_values.str.lower().isin(empty_tokens))
            candidate_df = df.loc[~(in_excluded_type & has_concept_info), ["name", "description"]].copy() if len(df) == len(candidate_df) else candidate_df

        try:
            if matcher_module is not None and hasattr(matcher_module, "find_nearest_vocab"):
                raw_hits = matcher_module.find_nearest_vocab(candidate_df, index_lib, vocab_dict, k_nearest=k_nearest)
            else:
                raw_hits = []
        except TypeError:
            try:
                raw_hits = matcher_module.find_nearest_vocab(candidate_df, index_lib, vocab_dict, k_nearest)
            except Exception:
                raw_hits = []
        except Exception:
            raw_hits = []

        if not raw_hits:
            raw_hits = _match_keyword_candidates_direct(candidate_df, index_lib, vocab_dict, model, source_filter=source_filter, k_nearest=k_nearest)

        grouped: dict[str, list[dict[str, Any]]] = {}
        if isinstance(raw_hits, dict):
            if all(isinstance(v, list) for v in raw_hits.values()):
                flattened: list[dict[str, Any]] = []
                for query_name, hit_list in raw_hits.items():
                    for hit in hit_list:
                        if isinstance(hit, dict):
                            hit = dict(hit)
                            hit.setdefault("query", query_name)
                            flattened.append(hit)
                raw_hits = flattened
            else:
                raw_hits = [
                    {**({"query": key} if not isinstance(value, dict) else {}), **(value if isinstance(value, dict) else {})}
                    for key, value in raw_hits.items()
                ]
        for hit in raw_hits:
            if not isinstance(hit, dict):
                continue
            if source_filter is not None and str(hit.get("source", "")).strip() not in source_filter:
                continue
            query_name = str(hit.get("query") or "").strip()
            if not query_name:
                continue
            grouped.setdefault(query_name, []).append(hit)

        if not grouped and not candidate_df.empty:
            grouped = {str(row["name"]): [] for _, row in candidate_df.iterrows() if str(row["name"]).strip()}

        sorted_grouped: dict[str, list[dict[str, Any]]] = {}
        for query_name, hits in grouped.items():
            hits_sorted = sorted(hits, key=lambda hit: float(hit.get("Distance", 1.0)))
            sorted_grouped[query_name] = hits_sorted
        if sorted_grouped:
            results[table_name] = sorted_grouped

    return results


def _run_original_page_reuse(metadata_dict: dict[str, pd.DataFrame]) -> dict[str, Any]:
    reused: dict[str, Any] = {
        "context": {},
        "column_sorting": {},
        "unit_measurement": {},
        "method_review": {},
    }

    # Reuse the original context page helpers for name normalization and alignment logic.
    try:
        context_module = _load_module("dataannot_context_page_autopilot", CONTEXT_PAGE_FILE)
        if hasattr(context_module, "normalize_variable_name"):
            reused["context"]["normalize_variable_name"] = {
                table_name: {
                    str(name): context_module.normalize_variable_name(str(name))
                    for name in (df["name"].tolist() if "name" in df.columns else df.columns)
                    if str(name).strip()
                }
                for table_name, df in metadata_dict.items()
                if isinstance(df, pd.DataFrame)
            }
    except Exception:
        reused["context"]["normalize_variable_name"] = {}

    # Reuse the original sorting-page bucket logic for schema role inference.
    try:
        sorting_module = _load_module("dataannot_sorting_page_autopilot", SORTING_PAGE_FILE)
        per_table: dict[str, dict[str, str]] = {}
        for table_name, df in metadata_dict.items():
            if not isinstance(df, pd.DataFrame):
                continue
            names = df["name"].tolist() if "name" in df.columns else list(df.columns)
            per_table[table_name] = {}
            for name in names:
                text = str(name).strip()
                if not text:
                    continue
                if hasattr(sorting_module, "_guess_spatial_role"):
                    per_table[table_name][text] = sorting_module._guess_spatial_role(text) or "unspecified"
                elif hasattr(sorting_module, "_guess_bucket"):
                    per_table[table_name][text] = sorting_module._guess_bucket({"name": text})
        reused["column_sorting"] = per_table
    except Exception:
        reused["column_sorting"] = {}

    # Reuse the original QUDT unit-search logic and ranking pipeline.
    try:
        qudt_module = _load_module("dataannot_qudt_page_autopilot", QUDT_PAGE_FILE)
        qudt_df = qudt_module.load_qudt_data()
        per_table: dict[str, list[dict[str, str]]] = {}
        for table_name, df in metadata_dict.items():
            if not isinstance(df, pd.DataFrame):
                continue
            names = df["name"].tolist() if "name" in df.columns else list(df.columns)
            suggestions: list[dict[str, str]] = []
            for name in names:
                text = str(name).strip()
                if not text:
                    continue
                try:
                    matches = qudt_module.search_units(qudt_df, text, search_type="both")
                    if matches.empty:
                        continue
                    top = matches.head(5)
                    suggestions.append({
                        "name": text,
                        "count": str(len(top)),
                        "top_matches": [
                            {
                                "label": str(row.get("label", "")),
                                "symbol": str(row.get("symbol", "")),
                                "unit": str(row.get("unit", "")),
                                "quantity_kind": str(row.get("QuantityKind", "")),
                            }
                            for _, row in top.iterrows()
                        ],
                    })
                except Exception:
                    continue
            per_table[table_name] = suggestions
        reused["unit_measurement"] = per_table
    except Exception:
        reused["unit_measurement"] = {}

    # Reuse the original method page helpers for reference extraction and method normalization.
    try:
        method_module = _load_module("dataannot_method_page_autopilot", METHOD_PAGE_FILE)
        per_table: dict[str, dict[str, dict[str, str]]] = {}
        for table_name, df in metadata_dict.items():
            if not isinstance(df, pd.DataFrame):
                continue
            names = df["name"].tolist() if "name" in df.columns else list(df.columns)
            per_table[table_name] = {}
            for name in names:
                text = str(name).strip()
                if not text:
                    continue
                doi = method_module._extract_first_doi(text) if hasattr(method_module, "_extract_first_doi") else ""
                if hasattr(method_module, "_first_match"):
                    iso = method_module._first_match(text, method_module.ISO_PATTERN)
                    cen = method_module._first_match(text, method_module.CEN_PATTERN)
                else:
                    iso = cen = ""
                per_table[table_name][text] = {
                    "doi": doi,
                    "iso": iso,
                    "cen": cen,
                    "normalized": method_module._to_text(name) if hasattr(method_module, "_to_text") else text,
                }
        reused["method_review"] = per_table
    except Exception:
        reused["method_review"] = {}

    return reused


def process_record(url: str, output_dir: Path) -> dict[str, Any]:
    helpers = load_batch_helpers()
    metadata_tables = helpers.collect_metadata_tables_for_zenodo_url(url)
    metadata_utils = load_metadata_utils()
    normalized = metadata_utils.normalize_metadata_columns(metadata_tables)
    inferred_relationships = _infer_relationships(normalized)
    normalized = metadata_utils.sync_relationships_to_metadata(normalized, inferred_relationships)

    qudt_module = None
    try:
        qudt_module = load_qudt_helpers()
    except Exception:
        qudt_module = None

    qudt_df = None
    if qudt_module and hasattr(qudt_module, "load_qudt_data"):
        try:
            qudt_df = qudt_module.load_qudt_data()
        except Exception:
            qudt_df = None

    record_id = helpers.HELPERS.get_record_id_from_Zenodo_url(url)
    record_dir = output_dir / str(record_id)
    record_dir.mkdir(parents=True, exist_ok=True)

    export_metadata = {table_name: df.copy() if isinstance(df, pd.DataFrame) else df for table_name, df in normalized.items()}

    table_summaries: dict[str, Any] = {}
    downstream: dict[str, Any] = {
        "descriptions": {},
        "description_generation_feedback": {},
        "unit_suggestions": {},
        "method_suggestions": {},
        "relationships": [],
        "keyword_matches": {},
    }

    api_key = _resolve_openai_api_key()
    extra_context = _collect_context_text_for_record_url(url)
    context_text = _compose_context_text(normalized, extra_context=extra_context)
    keyword_matches = _generate_keyword_matches_for_metadata(normalized, k_nearest=50)
    downstream["keyword_matches"] = keyword_matches

    llm_descriptions = _generate_llm_descriptions_for_metadata(
        normalized,
        api_key=api_key,
        context_text=context_text,
        debug_feedback=downstream["description_generation_feedback"],
    )
    llm_uom = _generate_llm_uom_for_metadata(normalized, api_key=api_key, context_text=context_text)
    llm_method = _generate_llm_method_for_metadata(normalized, api_key=api_key, context_text=context_text)

    for table_name, df in export_metadata.items():
        if not isinstance(df, pd.DataFrame):
            continue
        safe_name = metadata_utils.build_metadata_export_filename(table_name, fallback="table")
        output_path = record_dir / safe_name
        df.to_csv(output_path, index=False)
        description_rows: list[dict[str, str]] = []
        unit_rows: list[dict[str, str]] = []
        method_rows: list[dict[str, str]] = []
        names = [str(v).strip() for v in (df["name"].tolist() if "name" in df.columns else df.columns) if str(v).strip()]
        for name in names:
            clean_name = _clean_text(name)
            description = _resolve_variable_description(df, clean_name)
            if table_name in llm_descriptions:
                matched = next((item for item in llm_descriptions[table_name] if item.get("name") == clean_name), None)
                if matched and matched.get("description"):
                    description = matched["description"]
            if table_name in llm_uom:
                uom_match = next((item for item in llm_uom[table_name] if item.get("name") == clean_name), None)
                if uom_match and uom_match.get("unit_symbol"):
                    unit_symbol = str(uom_match["unit_symbol"])
                else:
                    unit_symbol = _infer_unit_symbol(clean_name)
            else:
                unit_symbol = _infer_unit_symbol(clean_name)
            if table_name in llm_method:
                method_match = next((item for item in llm_method[table_name] if item.get("name") == clean_name), None)
                if method_match and method_match.get("method"):
                    method_hint = str(method_match["method"])
                else:
                    method_hint = _infer_method_hint(clean_name)
            else:
                method_hint = _infer_method_hint(clean_name)
            description_rows.append({"name": clean_name, "description": description})
            unit_rows.append({"name": clean_name, "unit_symbol": unit_symbol})
            method_rows.append({"name": clean_name, "method": method_hint})

        table_summary = build_record_summary(table_name, df)
        table_summary["bucket_map"] = _build_bucket_map_with_sorting_page(
            table_name,
            df,
            description_rows=description_rows,
            uom_rows=unit_rows,
            method_rows=method_rows,
        )
        table_summary["concept_map"] = _concept_map_from_bucket_map(table_summary["bucket_map"])
        if table_name in keyword_matches:
            table_summary["keyword_matches"] = keyword_matches[table_name]

        downstream["descriptions"][table_name] = description_rows
        downstream["unit_suggestions"][table_name] = unit_rows
        downstream["method_suggestions"][table_name] = method_rows

        if qudt_df is not None and hasattr(qudt_module, "translate_QUDT"):
            try:
                translated = qudt_module.translate_QUDT(df.copy(), qudt_df)
                if isinstance(translated, pd.DataFrame):
                    table_summary["qudt_translation_columns"] = list(translated.columns)
            except Exception:
                pass

        table_summaries[table_name] = table_summary

    downstream["relationships"] = inferred_relationships

    page_reuse = _run_original_page_reuse(normalized)
    report = {
        "record_id": record_id,
        "url": url,
        "table_count": len(table_summaries),
        "tables": table_summaries,
        "downstream": downstream,
        "page_reuse": page_reuse,
        "human_in_the_loop": build_human_in_the_loop_guide(normalized),
        "status": "ok" if table_summaries else "empty",
    }
    (record_dir / "autopilot_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    default_links_file = ROOT / "data" / "Link_List.txt"
    parser = argparse.ArgumentParser(description="Run the app’s metadata and downstream heuristics for a list of Zenodo records.")
    parser.add_argument("--links", nargs="*", default=[], help="One or more Zenodo URLs to process.")
    parser.add_argument("--links-file", type=Path, default=default_links_file, help="Text file with one Zenodo URL per line. Defaults to the repo test list in data/Link_List.txt.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "zenodo_autopilot_exports", help="Directory for exported metadata and summaries.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    urls: list[str] = list(args.links)
    if args.links_file:
        urls.extend(_read_links(args.links_file))
    urls = _normalize_link_list(urls)
    if not urls:
        raise SystemExit("No Zenodo URLs provided. Use --links or --links-file.")

    results: list[dict[str, Any]] = []
    for url in urls:
        try:
            report = process_record(url, output_dir)
            results.append(report)
            print(f"OK: {report['record_id']} -> {report['table_count']} tables")
        except Exception as exc:  # pragma: no cover - hard network failures are kept visible
            results.append({
                "url": url,
                "status": "error",
                "error": str(exc),
            })
            print(f"ERROR: {url} -> {exc}")

    run_summary = output_dir / "run_summary.json"
    run_summary.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved batch summary to: {run_summary}")


if __name__ == "__main__":
    main()
