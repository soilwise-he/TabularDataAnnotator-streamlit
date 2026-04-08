"""
Unit of Measurement Annotation Page

Steps:
1. Detect where UOM are required (only int and float)
2. Detect if any matches with ansis. if yes, take UOM from there 
3. Parse results from LLM for UOM 
4. See if earlier matches of ansis brought up some reasonable consensus
5. UI to browse QUDT and search for units
6. Allow selection and combination of QUDT units
"""
import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import os
import re
from ui.blocks import add_Soilwise_contact_sidebar,add_Soilwise_logo,add_clear_cache_button
from pylatexenc.latex2text import LatexNodes2Text
from util.metadata import apply_new_metadata_info

st.set_page_config(page_title="Tabular Soil Data Annotation", layout="wide")
add_Soilwise_logo()
add_clear_cache_button(key_prefix="Keyword_Matcher")


DATA_TYPE_OPTIONS = ['anyURI', 'base64Binary', 'boolean', 'date',
                     'dateTime', 'datetime', 'dateTimeStamp', 'decimal',
                     'integer', 'integer', 'long', 'int', 'short', 'byte',
                     'nonNegativeInteger', 'positiveInteger', 'unsignedLong',
                     'unsignedInt', 'unsignedShort', 'unsignedByte',
                     'nonPositiveInteger', 'negativeInteger', 'double',
                     'number', 'duration', 'dayTimeDuration', 'yearMonthDuration',
                     'float', 'gDay', 'gMonth', 'gMonthDay', 'gYear', 'gYearMonth',
                     'hexBinary', 'QName', 'string', 'normalizedString', 'token',
                     'language', 'Name', 'NMTOKEN', 'time', 'xml', 'html', 'json']

# CSVW types for which a unit of measurement is meaningful.
DATA_TYPE_OPTIONS_UoM = [
    'decimal', 'integer', 'long', 'int', 'short', 'byte',
    'nonNegativeInteger', 'positiveInteger',
    'unsignedLong', 'unsignedInt', 'unsignedShort', 'unsignedByte',
    'nonPositiveInteger', 'negativeInteger',
    'double', 'float', 'number',
]

meta_key = f"metadata_df"

# -------------------- Helper data and functions --------------------

@st.cache_data
def load_qudt_data():
    """Load QUDT data from CSV file"""
    qudt_path = Path(__file__).parent.parent / "code" / "all.csv"
    if not qudt_path.exists():
        st.warning(f"QUDT data not found at {qudt_path}")
        return pd.DataFrame()
    
    df = pd.read_csv(qudt_path, low_memory=False)
    df = df[df['labelLang'] == 'en']
    

    # strip latex formatting from descriptions for better searchability
    if "description" in df.columns:
        latex_converter = LatexNodes2Text()

        def _safe_latex_to_text(value) -> str:
            if pd.isna(value):
                return ""
            try:
                return latex_converter.latex_to_text(value if isinstance(value, str) else str(value)).strip()
            except Exception:
                # Keep original content if parsing fails on malformed LaTeX.
                return value if isinstance(value, str) else str(value)

        df["description"] = df["description"].apply(_safe_latex_to_text)

    duplicate_keys = ["unit", "label", "labelLang", "QuantityKind"]

    if "applicableSystem" in df.columns:
        def _join_applicable_systems(values: pd.Series) -> Optional[str]:
            unique_values: List[str] = []
            seen = set()
            for value in values.dropna():
                text = str(value).strip()
                if not text or text in seen:
                    continue
                seen.add(text)
                unique_values.append(text)
            if not unique_values:
                return None
            return " || ".join(unique_values)

        aggregated_applicable_systems = (
            df.groupby(duplicate_keys, dropna=False)["applicableSystem"]
            .agg(_join_applicable_systems)
            .reset_index(name="applicableSystem")
        )

        df = (
            df.drop(columns=["applicableSystem"])
            .drop_duplicates(subset=duplicate_keys, keep="first")
            .merge(aggregated_applicable_systems, on=duplicate_keys, how="left")
        )
    else:
        df = df.drop_duplicates(subset=duplicate_keys, keep="first")


    return df


def search_units(qudt_df: pd.DataFrame, search_term: str, search_type: str = "both") -> pd.DataFrame:
    """Search for units in QUDT data"""
    if not search_term or search_term.strip() == "":
        return pd.DataFrame()
    
    search_lower = search_term.lower()
    mask = pd.Series(False, index=qudt_df.index)
    lookup = build_qudt_lookup(qudt_df)
    
    if search_type in ["unit", "both"]:
        mask |= qudt_df["label"].fillna("").str.lower().str.contains(search_lower, regex=False)
        mask |= qudt_df["symbol"].fillna("").str.lower().str.contains(search_lower, regex=False)
        mask |= qudt_df["description"].fillna("").str.lower().str.contains(search_lower, regex=False)
        mask |= qudt_df["ucumCode"].fillna("").str.lower().str.contains(search_lower, regex=False)
        search_lower_normalized = _normalize_qudt_text(search_lower)
        if search_lower_normalized in lookup["unit"]:
            matched_uris = lookup["unit"][search_lower_normalized]
            mask |= qudt_df["unit"].isin(matched_uris)


    if search_type in ["quantitykind", "both"]:
        mask |= qudt_df["QuantityKind"].fillna("").str.lower().str.contains(search_lower, regex=False)
        mask |= qudt_df["QuantityKindDescription"].fillna("").str.lower().str.contains(search_lower, regex=False)
        if search_lower_normalized in lookup["quantitykind"]:
            matched_uris = lookup["quantitykind"][search_lower_normalized]
            mask |= qudt_df["QuantityKind"].isin(matched_uris)

    results = qudt_df[mask].copy()
    results = results.drop_duplicates(subset=["unit","QuantityKind"], keep="first")
    return results.sort_values("label")


def _conversion_multiplier_is_one(value) -> bool:
    if _is_missing_value(value):
        return False
    try:
        return float(value) == 1.0
    except (TypeError, ValueError):
        return False


def rank_unit_results(
    results_df: pd.DataFrame,
    prioritized_unit_uris: Optional[List[str]] = None,
) -> pd.DataFrame:
    
    """Rank unit search results by user relevance signals."""
    
    if results_df.empty:
        return results_df

    # prioritezed set are the units sugested by the AI or Ansis geusses
    prioritized_set = set(prioritized_unit_uris or [])
    ranked = results_df.copy()

    ranked["_priority_guess"] = ranked["unit"].isin(prioritized_set)
    ranked["_priority_si"] = ranked["applicableSystem"].fillna("").astype(str).str.contains(
        "http://qudt.org/vocab/sou/SI", regex=False
    )

    # TODO: Check if multiplies one is actualy working. Is if susceptible for string error?
    ranked["_priority_multiplier_one"] = ranked["conversionMultiplier"].apply(_conversion_multiplier_is_one)

    ranked = ranked.sort_values(
        by=["_priority_guess", "_priority_si", "_priority_multiplier_one", "label"],
        ascending=[False, False, False, True],
        kind="stable",
    )
    return ranked #DEBUG: return when everything works; .drop(columns=["_priority_guess", "_priority_si", "_priority_multiplier_one"])


def get_suggested_units_for_variable(guesses_qudt_df: pd.DataFrame, variable_name: str) -> List[str]:
    """Get suggested unit URIs for a specific variable from translated guesses."""
    if guesses_qudt_df.empty or "Variable" not in guesses_qudt_df.columns:
        return []

    variable_rows = guesses_qudt_df[guesses_qudt_df["Variable"] == variable_name]
    if variable_rows.empty:
        return []

    variable_row = variable_rows.iloc[0]
    suggested: List[str] = []
    for column in guesses_qudt_df.columns:
        if "UoM" not in column or "Kind" in column:
            continue
        suggested.extend(
            _extract_uri_list(variable_row.get(column), "http://qudt.org/vocab/unit/")
        )

    return list(dict.fromkeys(suggested))


def search_units_with_kind_filter(
    qudt_df: pd.DataFrame,
    search_term: str,
    selected_kinds: Optional[List[str]] = None,
    prioritized_unit_uris: Optional[List[str]] = None,
    max_results: int = 300,
) -> pd.DataFrame:
    """Search units and optionally constrain to selected quantity kinds."""
    scoped_df = qudt_df
    if selected_kinds:
        scoped_df = scoped_df[scoped_df["QuantityKind"].isin(selected_kinds)]

    if search_term and search_term.strip():
        ranked_results = rank_unit_results(
            search_units(scoped_df, search_term, "unit"),
            prioritized_unit_uris,
        )
        return ranked_results.head(max_results)

    if scoped_df.empty:
        return pd.DataFrame()

    return rank_unit_results(
        scoped_df.drop_duplicates(subset=["unit", "QuantityKind"], keep="first")
        .sort_values("label"),
        prioritized_unit_uris,
    ).head(max_results)


@st.cache_data
def get_quantity_kind_catalog(qudt_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare unique quantity-kind records with searchable metadata."""
    kind_df = (
        qudt_df[["QuantityKind", "QuantityKindDescription"]]
        .dropna(subset=["QuantityKind"])
        .drop_duplicates(subset=["QuantityKind"], keep="first")
        .copy()
    )
    kind_df["kind_short"] = kind_df["QuantityKind"].str.replace(
        "http://qudt.org/vocab/quantitykind/", "", regex=False
    )
    kind_df["QuantityKindDescription"] = kind_df["QuantityKindDescription"].fillna("")
    kind_df["display"] = kind_df.apply(
        lambda row: (
            f"{row['kind_short']} | {row['QuantityKindDescription']}"
            if row["QuantityKindDescription"].strip()
            else row["kind_short"]
        ),
        axis=1,
    )
    kind_df["search_text"] = (
        kind_df["QuantityKind"].str.lower()
        + " "
        + kind_df["kind_short"].str.lower()
        + " "
        + kind_df["QuantityKindDescription"].str.lower()
    )
    return kind_df.sort_values("display")


def _extract_uri_list(value, uri_prefix: str) -> List[str]:
    if _is_missing_value(value):
        return []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str) and item.startswith(uri_prefix)]
    if isinstance(value, str) and value.startswith(uri_prefix):
        return [value]
    return []


def get_suggested_quantity_kinds_for_variable(guesses_qudt_df: pd.DataFrame, variable_name: str) -> List[str]:
    """Get suggested quantity-kind URIs for a specific variable from translated guesses."""
    if guesses_qudt_df.empty or "Variable" not in guesses_qudt_df.columns:
        return []

    variable_rows = guesses_qudt_df[guesses_qudt_df["Variable"] == variable_name]
    if variable_rows.empty:
        return []

    variable_row = variable_rows.iloc[0]
    suggested: List[str] = []
    for column in guesses_qudt_df.columns:
        if "Kind" not in column:
            continue
        suggested.extend(
            _extract_uri_list(variable_row.get(column), "http://qudt.org/vocab/quantitykind/")
        )

    return list(dict.fromkeys(suggested))


def get_numeric_variables(metadata_dict: Dict) -> Dict[str, List[str]]:
    """Get all numeric variables from metadata"""
    numeric_vars = {}
    for table_key, df in metadata_dict.items():
        numeric_cols = df[df["datatype"] == "numeric"]
        numeric_vars[table_key] = numeric_cols["name"].tolist()
    return numeric_vars


def format_unit_display(row: pd.Series) -> str:
    """Format a unit row for display"""
    parts = []
    if pd.notna(row.get("QuantityKind")):
        qk = str(row["QuantityKind"]).replace("http://qudt.org/vocab/quantitykind/", "")
        parts.append(f"[{qk}]")
    if pd.notna(row.get("label")):
        parts.append(str(row["label"]))
    if pd.notna(row.get("symbol")) and row["symbol"]:
        parts.append(f"({row['symbol']})")

    return " ".join(parts)

def _is_missing_value(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if isinstance(value, str) and value.strip() in {"", "N/A", "None", "nan"}:
        return True
    return False


def _normalize_qudt_text(value: str) -> str:
    normalized = str(value).strip().lower()
    normalized = normalized.replace("http://qudt.org/vocab/unit/", "")
    normalized = normalized.replace("http://qudt.org/vocab/quantitykind/", "")

    # Use token-aware replacements of known aliases.
    alias_patterns = {
        r"\b(?:yr|yrs|year|years)\b": "a",
        "²": "2",
        "³": "3",
    }
    for pattern, replacement in alias_patterns.items():
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)

    for character in ["-1","-", "_", "/", "(", ")", "[", "]", ".", ",", "·","^"," "]:
        normalized = normalized.replace(character, "")
    return "".join(normalized.split())


def _coerce_to_list(value):
    if _is_missing_value(value):
        return []
    if isinstance(value, (list, tuple, set)):
        return [item for item in value if not _is_missing_value(item)]
    return [value]


def _kind_column_for(unit_column: str, columns: List[str]) -> Optional[str]:
    direct_candidate = unit_column.replace("UoM", "UoM Kind")
    if direct_candidate in columns:
        return direct_candidate

    if unit_column.endswith("Candidate"):
        candidate_kind = unit_column.replace("Candidate", "Kind Candidate")
        if candidate_kind in columns:
            return candidate_kind

    return None


@st.cache_data
def build_qudt_lookup(qudt_reference: pd.DataFrame) -> Dict[str, Dict[str, List[str]]]:
    unit_lookup: Dict[str, List[str]] = {}
    kind_lookup: Dict[str, List[str]] = {}

    #TODO: explore also use of [conversionMultiplier,baseUnit,UnitOfSystem,applicableSystem]

    for _, row in qudt_reference.iterrows():
        unit_uri = row.get("unit")
        kind_uri = row.get("QuantityKind")

        if pd.notna(unit_uri):
            for candidate in [
                row.get("label"),
                row.get("symbol"),
                row.get("ucumCode"),
                unit_uri,
                str(unit_uri).rsplit("/", 1)[-1],
            ]:
                if _is_missing_value(candidate):
                    continue
                key = _normalize_qudt_text(candidate)
                unit_lookup.setdefault(key, [])
                if unit_uri not in unit_lookup[key]:
                    unit_lookup[key].append(unit_uri)
                    
        if pd.notna(kind_uri):
            for candidate in [
                kind_uri,
                str(kind_uri).rsplit("/", 1)[-1],
                row.get("QuantityKindDescription"), #TODO: Check where used? 
            ]:
                if _is_missing_value(candidate):
                    continue
                key = _normalize_qudt_text(candidate)
                kind_lookup.setdefault(key, [])
                if kind_uri not in kind_lookup[key]:
                    kind_lookup[key].append(kind_uri)


    return {"unit": unit_lookup, "kind": kind_lookup}


def resolve_quantity_kind_uri(kind_value, qudt_reference: pd.DataFrame, lookup: Dict[str, Dict[str, List[str]]]):
    candidates = _coerce_to_list(kind_value)
    resolved = []

    known_kind_uris = set(qudt_reference["QuantityKind"].dropna().unique())

    for candidate in candidates:
        if candidate in known_kind_uris:
            resolved.append(candidate)
            continue

        normalized = _normalize_qudt_text(candidate)
        if normalized in lookup["kind"]:
            resolved.extend(lookup["kind"][normalized])
            continue

        for lookup_key, lookup_values in lookup["kind"].items():
            if normalized and normalized in lookup_key:
                resolved.extend(lookup_values)

    resolved = list(dict.fromkeys(resolved))
    if not resolved:
        return None
    if len(resolved) == 1:
        return resolved[0]
    return resolved


def resolve_unit_string_to_uri(unit_value,
                    kind_value,
                    qudt_reference: pd.DataFrame,
                    lookup: Dict[str, Dict[str, List[str]]]
                    )-> List[str] | None:
    """
    Resolve unit URIs based on the provided unit value and optional quantity kind context, using direct matches and lookup tables. If quantity kind context is provided, it will prioritize units that are associated with the resolved quantity kinds in the QUDT reference data but when no matches are found with the quantity kind filter, it will fallback to returning any units that match the unit value regardless of quantity kind association.
    """
    unit_candidates = _coerce_to_list(unit_value)

    if not unit_candidates:
        return None

    resolved_kinds = resolve_quantity_kind_uri(kind_value, qudt_reference, lookup)
    if isinstance(resolved_kinds, str):
        resolved_kind_list = [resolved_kinds]
    elif isinstance(resolved_kinds, list):
        resolved_kind_list = resolved_kinds
    else:
        resolved_kind_list = []
    
    known_unit_uris = set(qudt_reference["unit"].dropna().unique())
    resolved_units = []

    for candidate in unit_candidates:
        if candidate in known_unit_uris:
            resolved_units.append(candidate)

            # this extra check is useful but ansis doesn't respect this :|
            # if not resolved_kind_list:
            #     resolved_units.append(candidate)
            #     continue

            # matched_rows = qudt_reference[
            #     (qudt_reference["unit"] == candidate)
            #     & (qudt_reference["QuantityKind"].isin(resolved_kind_list))
            # ]
            # if not matched_rows.empty:
            #     resolved_units.append(candidate)
            #     continue

        normalized = _normalize_qudt_text(candidate)

        # Direct lookup
        candidate_unit_uris = lookup["unit"].get(normalized, []) 

        

        if not candidate_unit_uris:
            for lookup_key, lookup_values in lookup["unit"].items():
                if normalized and normalized in lookup_key:
                    candidate_unit_uris.extend(lookup_values)

        candidate_unit_uris = list(dict.fromkeys(candidate_unit_uris))
        unfiltered_candidate_unit_uris = candidate_unit_uris.copy()

        if resolved_kind_list:
            filtered_unit_uris = []
            for unit_uri in candidate_unit_uris:
                matched_rows = qudt_reference[
                    (qudt_reference["unit"] == unit_uri)
                    & (qudt_reference["QuantityKind"].isin(resolved_kind_list))
                ]
                if not matched_rows.empty:
                    filtered_unit_uris.append(unit_uri)
            candidate_unit_uris = filtered_unit_uris

            # Fallback: if kind-constrained resolution yields nothing, retry without kind filtering.
            if not candidate_unit_uris:
                candidate_unit_uris = unfiltered_candidate_unit_uris

        resolved_units.extend(candidate_unit_uris)

    resolved_units = list(dict.fromkeys(resolved_units))
    if not resolved_units:
        return None
    if len(resolved_units) == 1:
        return resolved_units[0]
    return resolved_units


def _append_unit_kinds_to_kind_value(
    current_kind_value,
    resolved_unit_value,
    unit_to_kinds: Dict[str, List[str]],
):
    """Append inferred QuantityKind values from resolved units to the current kind value."""
    existing_kinds = [
        kind
        for kind in _coerce_to_list(current_kind_value)
        if isinstance(kind, str) and kind.startswith("http://qudt.org/vocab/quantitykind/")
    ]

    inferred_kinds: List[str] = []
    for unit_uri in _coerce_to_list(resolved_unit_value):
        if not isinstance(unit_uri, str):
            continue
        inferred_kinds.extend(unit_to_kinds.get(unit_uri, []))
    inferred_kinds = [] if len(inferred_kinds) > 2 else inferred_kinds  # Heuristic to avoid adding too many inferred kinds in case of very generic units.
    
    merged = list(dict.fromkeys(existing_kinds + inferred_kinds))
    if not merged:
        return None
    if len(merged) == 1:
        return merged[0]
    return merged


def translate_QUDT(df: pd.DataFrame, qudt_reference: pd.DataFrame) -> pd.DataFrame:
    translated_df = df.copy()
    lookup = build_qudt_lookup(qudt_reference)
    columns = translated_df.columns.tolist()
    unit_to_kinds = (
        qudt_reference.dropna(subset=["unit", "QuantityKind"])
        .groupby("unit")["QuantityKind"]
        .agg(lambda series: list(dict.fromkeys(series.tolist())))
        .to_dict()
    )

    for column in columns:
        if column == "Variable":
            continue

        if "Kind" in column:
            translated_df[column] = translated_df[column].apply(
                lambda value: resolve_quantity_kind_uri(value, qudt_reference, lookup)
            )
            continue

    for column in columns:
        if "UoM" in column and not "Kind" in column:
            kind_column = _kind_column_for(column, columns)

            if kind_column:
                def _resolve_and_enrich_row(row):
                    resolved_units = resolve_unit_string_to_uri(
                        row[column],
                        row[kind_column],
                        qudt_reference,
                        lookup,
                    )
                    enriched_kinds = _append_unit_kinds_to_kind_value(
                        row[kind_column],
                        resolved_units,
                        unit_to_kinds,
                    )
                    return pd.Series(
                        {
                            column: resolved_units,
                            kind_column: enriched_kinds,
                        }
                    )

                translated_df[[column, kind_column]] = translated_df.apply(
                    _resolve_and_enrich_row,
                    axis=1,
                )
            else:
                translated_df[column] = translated_df.apply(
                    lambda row: resolve_unit_string_to_uri(
                        row[column],
                        None,
                        qudt_reference,
                        lookup,
                    ),
                    axis=1,
                )

    # Consistency healing for Arrow compatibility:
    # if a column mixes strings and lists, promote strings to one-item lists.
    for column in columns:
        if column == "Variable":
            continue

        col_series = translated_df[column]
        non_missing = col_series[~col_series.apply(_is_missing_value)]
        has_list_values = non_missing.apply(lambda value: isinstance(value, list)).any()
        has_string_values = non_missing.apply(lambda value: isinstance(value, str)).any()

        if has_list_values and has_string_values:
            translated_df[column] = col_series.apply(
                lambda value: [value] if isinstance(value, str) else value
            )

    return translated_df
    

# PIN -------------------- UI --------------------

st.title("📐 Unit of Measurement Annotation")

st.markdown("""
This page helps you annotate unit of measurements for numeric variables in your dataset.
The tool provides:
- AI-generated guesses based on variable names and descriptions
- Vocabulary-matched suggestions from domain ontologies (ANSIS)
- Search capability to browse QUDT units and quantity kinds
- Manual selection and confirmation interface
""")

st.markdown("---")

## --- Initialize Session State ---
if "qudt_data" not in st.session_state:
    st.session_state["qudt_data"] = load_qudt_data()
if "uom_selections" not in st.session_state:
    st.session_state["uom_selections"] = {}
if "suggestion_pending" not in st.session_state:
    st.session_state["suggestion_pending"] = {}
if "uom_manual_conversion" not in st.session_state:
    st.session_state["uom_manual_conversion"] = {}


st.markdown("#### 🔍 Search & Select Units from QUDT")

# Verify that required data is available
if meta_key not in st.session_state or st.session_state.get(meta_key) is None:
    st.info("📋 No metadata found. Please upload a data file in step 1 to proceed with unit annotation.")
    st.stop()

qudt_df = st.session_state["qudt_data"]
if qudt_df.empty:
    st.error("❌ QUDT data is not available. Cannot proceed with unit annotation.")
    st.stop()

# Create one tab per key
tab_labels = list(st.session_state[meta_key].keys())
tabs = st.tabs(tab_labels)


for tab, key in zip(tabs, tab_labels):
    with tab:
        # # Show metadata overview
        # st.dataframe(st.session_state[meta_key][key])

        # st.divider()
        
        # Get numeric columns (CSVW types for which a UoM is meaningful)
        meta_df = st.session_state[meta_key][key]
        numeric_cols = meta_df[meta_df["datatype"].isin(DATA_TYPE_OPTIONS_UoM)]
        
        if numeric_cols.empty:
            st.info(f"ℹ️ No numeric variables found in **{key}**. Only numeric variables require unit annotations.")
            continue
        
        # Initialize uom_selections for this table if not already done
        if key not in st.session_state["uom_selections"]:
            st.session_state["uom_selections"][key] = {}
        
       
        # Prepare summary of guesses
        guesses_summary = []
        for _, row in numeric_cols.iterrows():
            var_name = row["name"]
            guess_info = {"Variable": var_name}
            
            # Get AI context guesses
            if "AI_var_UoM" in st.session_state and key in st.session_state["AI_var_UoM"]:
                ai_df = st.session_state["AI_var_UoM"][key]
                ai_row = ai_df[ai_df["name"] == var_name]
                if not ai_row.empty:
                    ai_row = ai_row.iloc[0]
                    guess_info["AI UoM"] = ai_row.get("UoM", "")
                    guess_info["AI UoM Kind"] = ai_row.get("UoM kind", "")
                    guess_info["AI Best Guess UoM"] = ai_row.get("UoM - Best guess", "")
                    guess_info["AI Best Guess UoM Kind"] = ai_row.get("UoM kind - Best guess", "")
            
            # Get ANSIS guesses
            if "ansis_UoM" in st.session_state and key in st.session_state["ansis_UoM"]:
                ansis_data = st.session_state["ansis_UoM"][key]
                if var_name in ansis_data and "selected_UoM" in ansis_data[var_name]:
                    uom_tuple = ansis_data[var_name]["selected_UoM"]
                    guess_info["ANSIS UoM"] = uom_tuple[0] if uom_tuple[0] else "N/A"
                    guess_info["ANSIS UoM Kind"] = uom_tuple[1] if uom_tuple[1] else "N/A"
                if var_name in ansis_data and "Potential_UoM" in ansis_data[var_name]:
                    uom_tuples_candidates = ansis_data[var_name]["Potential_UoM"]
                    guess_info["ANSIS UoM Candidate"] = {uom_candidate[0] for uom_candidate in uom_tuples_candidates if uom_candidate[0]}
                    guess_info["ANSIS UoM Kind Candidate"] = {uom_candidate[1] for uom_candidate in uom_tuples_candidates if uom_candidate[1] }

            guesses_summary.append(guess_info)
        
        if guesses_summary:
            guesses_df = pd.DataFrame(guesses_summary)
            guesses_qudt_df = translate_QUDT(guesses_df, qudt_df)

            with st.expander("🛠️ Debug QUDT guesses", expanded=False):
                st.markdown("#### Raw guesses")
                st.dataframe(guesses_df, width='stretch')

                st.markdown("#### QUDT-translated guesses")
                st.dataframe(guesses_qudt_df, width='stretch')

        else:
            guesses_qudt_df = pd.DataFrame()
 


        kind_catalog = get_quantity_kind_catalog(qudt_df)
        kind_uri_to_display = dict(zip(kind_catalog["QuantityKind"], kind_catalog["display"]))
        display_to_uri = {v: k for k, v in kind_uri_to_display.items()}

        # Create a section for each variable to identify suitable quantity kinds and units
        for _, row in numeric_cols.iterrows():
            col_expander, col_feedback = st.columns([7,1])
            with col_expander:
                var_name = row["name"]

                with st.expander(f"📊 {var_name}", expanded=False):

                    suggested_kind_uris = get_suggested_quantity_kinds_for_variable(guesses_qudt_df, var_name)
                    suggested_unit_uris = get_suggested_units_for_variable(guesses_qudt_df, var_name)
                    
                    merged_kind_uris = list(dict.fromkeys(suggested_kind_uris))

                    
                    default_displays = [
                        kind_uri_to_display[uri]
                        for uri in suggested_kind_uris
                        if uri in kind_uri_to_display
                    ]

                    selected_kind_displays = st.multiselect(
                        "Quantity kinds (autocomplete enabled):",
                        options=kind_catalog["display"],
                        default=default_displays,
                        key=f"selected_kinds_{key}_{var_name}",
                        help="Type to autocomplete. quantity kinds as defined by QUDT; https://www.qudt.org/doc/DOC_VOCAB-QUANTITY-KINDS.html"
                    )
                    
                    selected_kind_uris = [display_to_uri[display] for display in selected_kind_displays]
                    
                    # # Perform search
                    # if search_term or selected_kind_uris:
                    search_term = st.text_input(
                            "Refine search for Unit of Measurement:",
                            key=f"search_{key}_{var_name}",
                            help="Search by unit name, symbol, or ucum symbol"
                        )
                    
                    cap_results =500 
                    search_results = search_units_with_kind_filter(
                        qudt_df,
                        search_term,
                        selected_kind_uris,
                        suggested_unit_uris,
                        cap_results
                    )


                    if not search_results.empty:
                        # Auto-select if there's exactly one _priority_guess result
                        auto_select_idx = None
                        if "_priority_guess" in search_results.columns:
                            guess_rows = search_results[search_results["_priority_guess"] == True]
                            if len(guess_rows) == 1:
                                auto_select_idx = search_results.index.get_loc(guess_rows.index[0])
                        
                        if len(search_results)==cap_results:
                            st.markdown(f"⚠️ Search results capped at {cap_results}. Please refine your search term or quantity kind selection to see more specific results.")
                        else:    
                            st.markdown(f"**Found {len(search_results)} results:**")                   
                        
                        selection = st.dataframe(search_results,
                                                hide_index=True,
                                                column_order=("label", "description", "symbol", "ucumCode", "QuantityKind"),
                                                on_select="rerun",
                                                selection_mode="single-row",
                                                key=f"select_{key}_{var_name}_searchresults")
                        
                        selection_idx = selection.get("selection").get("rows")[0] if selection.get("selection").get("rows") else auto_select_idx
                    

                        if selection_idx!= None:
                            selected = search_results.iloc[selection_idx]
                            st.session_state["uom_selections"][key][var_name] = selected
                            
                            # If this is the auto-selected suggestion, mark as pending approval (only if not already approved)
                            if selection_idx == auto_select_idx:
                                if key not in st.session_state["suggestion_pending"]:
                                    st.session_state["suggestion_pending"][key] = {}
                                # Only mark as pending if not already processed (i.e., not yet approved)
                                if var_name not in st.session_state["suggestion_pending"][key]:
                                    st.session_state["suggestion_pending"][key][var_name] = "pending"
                            else:
                                # User made a manual selection; clear any pending suggestion
                                if key in st.session_state["suggestion_pending"] and var_name in st.session_state["suggestion_pending"][key]:
                                    del st.session_state["suggestion_pending"][key][var_name]
                            
                            st.success(f"✅ {var_name} → {selected['label']}")
                        else:
                            if var_name in st.session_state["uom_selections"][key]:
                                del st.session_state["uom_selections"][key][var_name]
                    else:
                            st.warning("⚠️ No matching units found. Try a different search term. Or deselect quantity kind filters to broaden the search.")
                    
                    # --- Last-resort manual conversion override ---
                    with st.expander("⚙️ Advanced: manual conversion override", expanded=False):
                        st.caption(
                            "Only use this if your unit is not in QUDT and consider contributing to this open repository yourself. "
                            "In the meantime, we are trying to provide a temporary solution using the method described below. \n\n  "
                            "STEPS TO USE: \n\n"
                            "1. Select the unit that is closest to yours (e.g., if you have 'hectares' and find 'square meters', select 'square meters'). \n\n"
                            "2. Define how the column's raw values relate to the selected unit: \n\n"
                            "   `converted = raw × multiplier + offset`."
                        )
                        conv_key_prefix = f"conv_{key}_{var_name}"

                        # Lazy init for this table
                        if key not in st.session_state["uom_manual_conversion"]:
                            st.session_state["uom_manual_conversion"][key] = {}

                        no_unit_selected = var_name not in st.session_state["uom_selections"].get(key, {})
                        conv_enabled = st.toggle(
                            "Enable manual conversion",
                            value=False,
                            key=f"{conv_key_prefix}_enabled",
                            disabled=no_unit_selected,
                            help="Select a unit above before enabling manual conversion." if no_unit_selected else None,
                        )

                        conv_multiplier = st.number_input(
                            "Conversion multiplier",
                            value=1.0,
                            format="%.10g",
                            key=f"{conv_key_prefix}_mult",
                            disabled=not conv_enabled,
                            help="Multiplier applied to raw values before offset.",
                        )
                        conv_offset = st.number_input(
                            "Conversion offset",
                            value=0.0,
                            format="%.10g",
                            key=f"{conv_key_prefix}_offset",
                            disabled=not conv_enabled,
                            help="Offset added after multiplying. Use 0 if not needed.",
                        )

                        if conv_enabled:
                            if conv_multiplier == 0:
                                st.warning("⚠️ A multiplier of 0 will collapse all values to the offset.")
                            st.session_state["uom_manual_conversion"][key][var_name] = {
                                "multiplier": conv_multiplier,
                                "offset": conv_offset,
                            }
                        else:
                            st.session_state["uom_manual_conversion"][key].pop(var_name, None)

                # Show current selection if exists
                if var_name in st.session_state["uom_selections"][key]:
                    selected = st.session_state["uom_selections"][key][var_name]
                    
                    # Check if this is a pending suggestion (not yet approved)
                    is_suggestion_pending = (
                        key in st.session_state["suggestion_pending"]
                        and var_name in st.session_state["suggestion_pending"][key]
                        and st.session_state["suggestion_pending"][key][var_name] == "pending"
                    )
                    
                    if is_suggestion_pending:
                        # Show approval button for suggested selection
                        col1, col2, col3, col4, col5 = st.columns([2,2,3,3,4])
                        with col1:
                            if st.button(
                                "✅ Approve",
                                key=f"approve_{key}_{var_name}",
                                width='stretch',
                            ):
                                # Mark suggestion as approved
                                st.session_state["suggestion_pending"][key][var_name] = "approved"
                                st.rerun()
                        with col2:
                            st.markdown("**🔍 SUGGESTED**")
                        with col3:
                            st.write(f"**Unit:** {selected['label']}")
                        with col4:
                            st.write(f"**Symbol:** {selected.get('symbol', 'N/A')}")
                        with col5:
                            qk = selected["QuantityKind"].replace("http://qudt.org/vocab/quantitykind/", "") if selected["QuantityKind"] else "N/A"
                            st.write(f"**Quantity Kind:** {qk}")
                    else:
                        # Show normal confirmed selection
                        col1, col2, col3, col4, col5 = st.columns([2,2,3,3,4])
                        with col2:
                            st.markdown("**➡️ SELECTED:**   ")
                        with col3:
                            st.write(f"**Unit:** {selected['label']}")
                        with col4:
                            st.write(f"**Symbol:** {selected.get('symbol', 'N/A')}")
                        with col5:
                            qk = selected["QuantityKind"].replace("http://qudt.org/vocab/quantitykind/", "") if selected["QuantityKind"] else "N/A"
                            st.write(f"**Quantity Kind:** {qk}")

                    # Show manual conversion info if active
                    conv_info = st.session_state.get("uom_manual_conversion", {}).get(key, {}).get(var_name)
                    if conv_info:
                        st.caption(
                            f"⚙️ Manual conversion: × {conv_info['multiplier']} + {conv_info['offset']}"
                        )

                    st.space('small')

                    with col_feedback:
                        feedback_key = f"feedback_{key}_{var_name}"
                        if is_suggestion_pending:
                            icon_feedback = "❓"
                        else:
                            icon_feedback = "✅"

                        if var_name in st.session_state["uom_selections"][key]:
                            st.write(icon_feedback) 

        # Approve All button (shown when at least one suggestion is pending approval)
        pending_vars = [
            var_name
            for var_name, status in st.session_state["suggestion_pending"].get(key, {}).items()
            if status == "pending"
        ]
        if pending_vars:
            st.space('medium')
            if st.button(
                f"✅ Approve All ({len(pending_vars)} pending suggestion{'s' if len(pending_vars) != 1 else ''})",
                key=f"approve_all_{key}",
                help=(
                    "We're genuinely flattered by your trust in our robot overlords — truly, it warms our circuits. "
                    "Please always give the suggestions above a quick sanity check first. "
                    "Your data will thank you. 🙏"
                ),
            ):
                for var_name in pending_vars:
                    st.session_state["suggestion_pending"][key][var_name] = "approved"
                st.rerun()


#TODO; How to delete?
# transform the uom_selections into a dataframe for saving it to session state
uom_selections_export = {}
for key, metadata in st.session_state["uom_selections"].items():
    if isinstance(metadata, Dict):
        pending_by_var = st.session_state.get("suggestion_pending", {}).get(key, {})

        approved_selections = {
            var_name: selection
            for var_name, selection in metadata.items()
            if var_name not in pending_by_var or pending_by_var.get(var_name) == "approved"
        }
        if approved_selections:
            metadata_df = pd.DataFrame(approved_selections).T.reset_index().rename(columns={"index": "name"})
            export_cols = ["name", "unit"]

            # Merge manual conversion values for approved variables
            manual_conv = st.session_state.get("uom_manual_conversion", {}).get(key, {})
            multipliers = {}
            offsets = {}
            for vn in approved_selections:
                conv = manual_conv.get(vn)
                if conv:
                    multipliers[vn] = conv["multiplier"]
                    offsets[vn] = conv["offset"]

            if multipliers:
                metadata_df["conversionMultiplier"] = metadata_df["name"].map(multipliers)
                export_cols.append("conversionMultiplier")
            if offsets:
                metadata_df["conversionOffset"] = metadata_df["name"].map(offsets)
                export_cols.append("conversionOffset")

            uom_selections_export[key] = metadata_df[export_cols]

st.session_state['metadata_df'] = apply_new_metadata_info(uom_selections_export, st.session_state['metadata_df'], overwrite='yes')


st.divider()

# Create one tab of selection per key
tabs_selection = st.tabs(tab_labels)


for tab, key in zip(tabs_selection, tab_labels):
    with tab:
        # # Show metadata overview
        # st.dataframe(st.session_state[meta_key][key])
                        

        # Summary of selections
        st.markdown("### 📋 Summary of Selections")
        
        if key in st.session_state["uom_selections"] and st.session_state["uom_selections"][key]:
            summary_data = []
            for var_name, selection in st.session_state["uom_selections"][key].items():
                entry = {
                    "Variable": var_name,
                    "Unit": selection["label"],
                    "Symbol": selection.get("symbol", ""),
                    "Quantity Kind": selection["QuantityKind"].replace("http://qudt.org/vocab/quantitykind/", "") if selection["QuantityKind"] else "N/A",
                }
                conv = st.session_state.get("uom_manual_conversion", {}).get(key, {}).get(var_name)
                if conv:
                    entry["Multiplier"] = conv["multiplier"]
                    entry["Offset"] = conv["offset"]
                summary_data.append(entry)
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, width='stretch')
            
            # Show quick stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Units Selected", len(st.session_state["uom_selections"][key]))
            with col2:
                st.metric("Numeric Variables", len(numeric_cols))
            with col3:
                if len(numeric_cols) > 0:
                    progress = len(st.session_state["uom_selections"][key]) / len(numeric_cols) * 100
                    st.metric("Progress", f"{progress:.0f}%")
        else:
            st.info("ℹ️ No units selected yet. Use the search boxes above to find and select units.")


        st.markdown("#### Current state of your metadata table:")
        st.session_state['metadata_df'][key]

# -------------------- reach us --------------------
add_Soilwise_contact_sidebar()
