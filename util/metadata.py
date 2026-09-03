from typing import Dict

import pandas as pd

import streamlit as st


LEGACY_COLUMN_RENAMES = {
    "datatype": "resulttype",
    "dateTime format": "resultformat",
    "concept_uri": "element uri",
}


METADATA_COLUMN_ORDER = [
    "name",
    "resulttype",
    "resultformat",
    "concept",
    "element",
    "element uri",
    "unit_symbol",
    "unit_uri",
    "quantity kind_uri",
    "method",
    "description",
]


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
                md["element"] = old_concept
                md["concept"] = old_element
            elif "element" in md.columns and "concept" not in md.columns:
                md["concept"] = md["element"]
                md["element"] = ""
            elif "concept" in md.columns and "element" not in md.columns:
                md["element"] = md["concept"]
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
        
        #TODO: delete fixed column list if no errors occur.
        #for col in ["datatype","dateTime format", "element", "concept", "unit", "method", "description", "concept_uri", "conversionMultiplier", "conversionOffset"]:
        for col in md.columns:
            # Ensure column exists in the target DataFrame before writing.
            if col not in md.columns:
                md[col] = None

            current_value = md.at[idx, col]
            if overwrite == "no_overwrite" and pd.notna(current_value) and current_value not in [None, ""]:
                continue

            can_write_value = overwrite == "yes" and col in row and pd.notna(row[col]) and row[col] != ""
            can_write_including_blanks = overwrite == "yes_incl_blanks" and col in row
            if not (can_write_value or can_write_including_blanks):
                continue

            value = row[col]
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
