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
from ui.blocks import add_Soilwise_contact_sidebar,add_Soilwise_logo,add_clear_cache_button

st.set_page_config(page_title="Tabular Soil Data Annotation", layout="wide")
add_Soilwise_logo()
add_clear_cache_button(key_prefix="Keyword_Matcher")

DATA_TYPE_OPTIONS = ["string", "numeric", "date"]

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

    # TODO delete later if priority is not to filter for SI units only
    # df = df[df['applicableSystem'] == 'http://qudt.org/vocab/sou/SI']

    df = df.drop_duplicates(subset=["unit", "label", "labelLang"], keep="first")

    return df


def search_units(qudt_df: pd.DataFrame, search_term: str, search_type: str = "both") -> pd.DataFrame:
    """Search for units in QUDT data"""
    if not search_term or search_term.strip() == "":
        return pd.DataFrame()
    
    search_lower = search_term.lower()
    mask = pd.Series(False, index=qudt_df.index)
    
    if search_type in ["unit", "both"]:
        mask |= qudt_df["label"].fillna("").str.lower().str.contains(search_lower, regex=False)
        mask |= qudt_df["symbol"].fillna("").str.lower().str.contains(search_lower, regex=False)
        mask |= qudt_df["description"].fillna("").str.lower().str.contains(search_lower, regex=False)
        mask |= qudt_df["ucumCode"].fillna("").str.lower().str.contains(search_lower, regex=False)
    
    if search_type in ["quantitykind", "both"]:
        mask |= qudt_df["QuantityKind"].fillna("").str.lower().str.contains(search_lower, regex=False)
        mask |= qudt_df["QuantityKindDescription"].fillna("").str.lower().str.contains(search_lower, regex=False)
    
    results = qudt_df[mask].copy()
    results = results.drop_duplicates(subset=["unit","QuantityKind"], keep="first")
    return results.sort_values("label")


def search_units_with_kind_filter(
    qudt_df: pd.DataFrame,
    search_term: str,
    selected_kinds: Optional[List[str]] = None,
    max_results: int = 300,
) -> pd.DataFrame:
    """Search units and optionally constrain to selected quantity kinds."""
    scoped_df = qudt_df
    if selected_kinds:
        scoped_df = scoped_df[scoped_df["QuantityKind"].isin(selected_kinds)]

    if search_term and search_term.strip():
        return search_units(scoped_df, search_term, 'unit').head(max_results)

    if scoped_df.empty:
        return pd.DataFrame()

    return (
        scoped_df.drop_duplicates(subset=["unit", "QuantityKind"], keep="first")
        .sort_values("label")
        .head(max_results)
    )


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
    for character in ["-", "_", "/", "(", ")", "[", "]", ".", ","]:
        normalized = normalized.replace(character, " ")
    return " ".join(normalized.split())


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


def resolve_unit_uri(unit_value,
                    kind_value,
                    qudt_reference: pd.DataFrame,
                    lookup: Dict[str, Dict[str, List[str]]]
                    ):
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
        candidate_unit_uris = lookup["unit"].get(normalized, [])

        if not candidate_unit_uris:
            for lookup_key, lookup_values in lookup["unit"].items():
                if normalized and normalized in lookup_key:
                    candidate_unit_uris.extend(lookup_values)

        candidate_unit_uris = list(dict.fromkeys(candidate_unit_uris))

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

        resolved_units.extend(candidate_unit_uris)

    resolved_units = list(dict.fromkeys(resolved_units))
    if not resolved_units:
        return None
    if len(resolved_units) == 1:
        return resolved_units[0]
    return resolved_units


def translate_QUDT(df: pd.DataFrame, qudt_reference: pd.DataFrame) -> pd.DataFrame:
    translated_df = df.copy()
    lookup = build_qudt_lookup(qudt_reference)
    columns = translated_df.columns.tolist()

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
            translated_df[column] = translated_df.apply(
                lambda row: resolve_unit_uri(
                    row[column],
                    row[kind_column] if kind_column else None,
                    qudt_reference,
                    lookup,
                ),
                axis=1,
            )

    return translated_df
    

# -------------------- Initialize Session State --------------------
if "qudt_data" not in st.session_state:
    st.session_state["qudt_data"] = load_qudt_data()

if "uom_selections" not in st.session_state:
    st.session_state["uom_selections"] = {}

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

st.markdown("#### Current state of your metadata table:")

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

# # Get numeric variables
# numeric_vars = get_numeric_variables(st.session_state[meta_key])


for tab, key in zip(tabs, tab_labels):
    with tab:
        # # Show metadata overview
        # st.dataframe(st.session_state[meta_key][key])

        # st.divider()
        
        # Get numeric columns
        meta_df = st.session_state[meta_key][key]
        numeric_cols = meta_df[meta_df["datatype"] == "numeric"]
        
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

            #DEBUG
            st.markdown("#### Raw guesses")
            st.dataframe(guesses_df, width='stretch')
            st.markdown("#### QUDT-translated guesses")
            st.dataframe(guesses_qudt_df, width='stretch')
        else:
            guesses_qudt_df = pd.DataFrame()

        ######
        #[ ]TODO: make informed guess for UoM
        #[x]TODO: Make find a way to use this background information of guesses to pre-populate search results in the next section
        #[x]TODO: Make the search function hierarchical by first searching for quantity kind and then filtering units based on that
        #[ ]TODO: find a sorting methode of the search results -> SI units first, then multiplier factor 1 -> (abs(log(mulplier))), then alphabetically?
        ######        
        
        st.divider()
        st.markdown("### 🔍 Search & Select Units from QUDT")
        
        
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
                        cap_results
                    )


                    if not search_results.empty:
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
                        
                        selection_idx = selection.get("selection").get("rows")[0] if selection.get("selection").get("rows") else None
                    

                        if selection_idx!= None:
                            selected = search_results.iloc[selection_idx]
                            st.session_state["uom_selections"][key][var_name] = selected
                            st.success(f"✅ {var_name} → {selected['label']}")
                        else:
                            if var_name in st.session_state["uom_selections"][key]:
                                del st.session_state["uom_selections"][key][var_name]
                    else:
                            st.warning("⚠️ No matching units found. Try a different search term.")
                    
                    
                    # Show current selection if exists
                    if var_name in st.session_state["uom_selections"][key]:
                        selected = st.session_state["uom_selections"][key][var_name]
                        st.markdown("**✅ Current Selection:**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Unit:** {selected['label']}")
                            st.write(f"**Symbol:** {selected.get('symbol', 'N/A')}")
                        with col2:
                            qk = selected["QuantityKind"].replace("http://qudt.org/vocab/quantitykind/", "") if selected["QuantityKind"] else "N/A"
                            st.write(f"**Quantity Kind:** {qk}")
                with col_feedback:
                    feedback_key = f"feedback_{key}_{var_name}"
                    if var_name in st.session_state["uom_selections"][key]:
                        st.write("✅") 

                        
        #TODO; start new tab section
        # Summary of selections
        st.divider()
        st.markdown("### 📋 Summary of Selections")
        
        if st.session_state["uom_selections"][key]:
            summary_data = []
            for var_name, selection in st.session_state["uom_selections"][key].items():
                summary_data.append({
                    "Variable": var_name,
                    "Unit": selection["label"],
                    "Symbol": selection.get("symbol", ""),
                    "Quantity Kind": selection["QuantityKind"].replace("http://qudt.org/vocab/quantitykind/", "") if selection["QuantityKind"] else "N/A"
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, width='stretch')
            
            # Show quick stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Units Selected", len(st.session_state["uom_selections"][key]))
            with col2:
                st.metric("Numeric Variables", len(numeric_cols))
            with col3:
                progress = len(st.session_state["uom_selections"][key]) / len(numeric_cols) * 100
                st.metric("Progress", f"{progress:.0f}%")
        else:
            st.info("ℹ️ No units selected yet. Use the search boxes above to find and select units.")

# -------------------- reach us --------------------
add_Soilwise_contact_sidebar()
