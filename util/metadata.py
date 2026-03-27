from typing import Dict

import pandas as pd

import streamlit as st


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
        for col in ["datatype","date format", "element", "unit", "method", "description", "element_uri"]:
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

        if key not in md_dict:
            continue

        md_dict[key] = _merge_metadata_rows(metadata_df, md_dict[key], overwrite)

    return md_dict
