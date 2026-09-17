import os
import re
from typing import Dict

import pandas as pd

import streamlit as st


METADATA_EXPORT_FILENAME_SUFFIX = "_metadata.csv"


LEGACY_COLUMN_RENAMES = {
    "datatype": "column_type",
    "dateTime format": "column_format",
    "element_uri": "concept_uri",
}


METADATA_COLUMN_ORDER = [
    "name",
    "column_type",
    "column_format",
    "concept_type",
    "concept",
    "concept_uri",
    "unit_symbol",
    "unit_uri",
    "quantity_kind_uri",
    "method",
    "method_uri",
    "description",
]


def safe_filename_component(value: str, fallback: str = "export") -> str:
    """Normalize arbitrary text into a filesystem-safe filename component."""
    text = str(value) if value is not None else ""
    # Windows-invalid filename chars: <>:"/\\|?* plus control chars.
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", text)
    text = text.strip().strip(".")
    return text or fallback


def build_metadata_export_filename(table_key: str, fallback: str = "table") -> str:
    """Build canonical metadata export filename for a table key."""
    return f"{safe_filename_component(table_key, fallback=fallback)}{METADATA_EXPORT_FILENAME_SUFFIX}"


def build_filename_match_tokens(value: str) -> set[str]:
    """Build normalized filename tokens used for table/metadata matching."""
    raw = str(value or "").strip()
    base = os.path.basename(raw)
    variants = {raw, base}
    tokens: set[str] = set()

    for variant in variants:
        candidate = str(variant or "").strip().lower()
        if not candidate:
            continue

        tokens.add(candidate)

        if candidate.endswith(METADATA_EXPORT_FILENAME_SUFFIX):
            candidate = candidate[: -len(METADATA_EXPORT_FILENAME_SUFFIX)]
            if candidate:
                tokens.add(candidate)

        if candidate.endswith(".csv"):
            stem = candidate[:-4]
            if stem:
                tokens.add(stem)

        safe_candidate = safe_filename_component(candidate, fallback="").lower()
        if safe_candidate:
            tokens.add(safe_candidate)

    return tokens


def _reorder_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply a stable canonical column order for metadata tables.

    Known metadata columns are ordered first; any extra columns are preserved
    in their existing order after the canonical set.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df

    ordered_existing = [col for col in METADATA_COLUMN_ORDER if col in df.columns]
    remaining = [col for col in df.columns if col not in ordered_existing]
    return df[ordered_existing + remaining]


def normalize_metadata_columns(metadata_dict: Dict) -> Dict:
    """Migrate legacy metadata column names to the current storage schema.

    Migration is only applied when legacy columns are detected, so already-migrated
    metadata is left untouched.
    """
    if not isinstance(metadata_dict, dict):
        return metadata_dict

    normalized = metadata_dict.copy()
    for table_key, table_df in normalized.items():
        if not isinstance(table_df, pd.DataFrame):
            continue

        md = table_df.copy()
        has_legacy_columns = any(col in md.columns for col in LEGACY_COLUMN_RENAMES)
        if has_legacy_columns:
            for old_col, new_col in LEGACY_COLUMN_RENAMES.items():
                if old_col not in md.columns:
                    continue
                if new_col in md.columns:
                    needs_fill = md[new_col].isna() | (md[new_col].astype(str).str.strip() == "")
                    md.loc[needs_fill, new_col] = md.loc[needs_fill, old_col]
                    md = md.drop(columns=[old_col])
                else:
                    md = md.rename(columns={old_col: new_col})

            # Swap element/concept only when migrating from legacy shape.
            if "element" in md.columns and "concept" in md.columns:
                old_element = md["element"].copy()
                old_concept = md["concept"].copy()
                md["concept_type"] = old_concept
                md["concept"] = old_element
            elif "element" in md.columns and "concept" not in md.columns:
                md["concept"] = md["element"]
                md["concept_type"] = ""
            elif "concept" in md.columns and "element" not in md.columns:
                md["concept_type"] = md["concept"]
                md["concept"] = ""

        md = _reorder_metadata_columns(md)

        normalized[table_key] = md

    return normalized


def _merge_metadata_rows(
    metadata_df: pd.DataFrame,
    current_meta: pd.DataFrame,
    overwrite: str,
) -> pd.DataFrame:
    md = current_meta.copy()
    for _, row in metadata_df.iterrows():
        name = row.get("name")
        if name not in md["name"].values:
            continue

        idx = md.index[md["name"] == name][0]
        
        for col in md.columns:
            # Ensure column exists in the target DataFrame before writing.
            if col not in md.columns:
                md[col] = None

            source_col = col
            # Backward compatibility: allow legacy payloads that still use element_uri.
            if col == "concept_uri" and source_col not in row and "element_uri" in row:
                source_col = "element_uri"

            current_value = md.at[idx, col]
            if overwrite == "no_overwrite" and pd.notna(current_value) and current_value not in [None, ""]:
                continue

            can_write_value = overwrite == "yes" and source_col in row and pd.notna(row[source_col]) and row[source_col] != ""
            can_write_including_blanks = overwrite == "yes_incl_blanks" and source_col in row
            if not (can_write_value or can_write_including_blanks):
                continue

            value = row[source_col]
            md.loc[idx, col] = str(value) if isinstance(value, dict) else value

    return md


def apply_new_metadata_info(
                            new_metadata: Dict,
                            current_meta: Dict,
                            overwrite: str = "no_overwrite",
                        ) -> Dict:
    """Merge metadata updates for either single-table DataFrames or table dictionaries."""
    if new_metadata is None:
        return current_meta

    md_dict = current_meta.copy()
    for key, metadata in new_metadata.items():
        if key not in md_dict:
            continue

        if isinstance(metadata, pd.DataFrame):
            metadata_df = metadata
        elif isinstance(metadata, dict):
            metadata_df = pd.DataFrame(metadata)
        else:
            st.write(f"⚠️ Warning: Metadata for key '{key}' is not in a recognized format (dict or DataFrame). Skipping.")
            continue  # Skip if metadata is neither a dict nor a DataFrame

        md_dict[key] = _merge_metadata_rows(metadata_df, md_dict[key], overwrite)

    return md_dict
