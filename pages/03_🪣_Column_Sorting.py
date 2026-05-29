from cProfile import label
import re
from datetime import date, datetime, timedelta
import requests
import pandas as pd
import streamlit as st
import hashlib, json
from streamlit_sortables import sort_items
from ui.blocks import add_Soilwise_logo, add_clear_cache_button

add_Soilwise_logo()
add_clear_cache_button(key_prefix="column_sorting_page")

st.set_page_config(page_title="Column Sorting", layout="wide")

st.title("🪣 Sort Columns into Buckets")
st.markdown(
    "Drag and drop the columns from your dataset into the appropriate buckets. "
    "This helps classify each column by its role and align with SOSA standards."
)

BUCKET_NAMES = [
    "Unsorted",
    "Feature of Interest (FOI) - ID",
    "FOI - Spatial Information",
    "FOI - Attribute",
    "Observed Property",
    "Temporal",
]

# --------------- gather column names from all loaded tables ---------------
meta_key = "metadata_df"
meta_dict = st.session_state.get(meta_key)
if not meta_dict:
    st.warning("⚠️ No data loaded yet. Please go to the **Input** page first and upload a dataset.")
    st.stop()

# Build per-table column lists and metadata lookup
_table_columns: dict[str, list[str]] = {}
_col_meta: dict[str, dict[str, dict]] = {}  # table_name -> col_name -> info
for table_name, df in meta_dict.items():
    cols = []
    _col_meta[table_name] = {}
    for _, row in df.iterrows():
        col_name = str(row["name"])
        if col_name not in cols:
            cols.append(col_name)
            _col_meta[table_name][col_name] = {
                "datatype": str(row.get("datatype", "")).lower(),
                "name": col_name.lower(),
                "description": str(row.get("description", "")).lower(),
            }
    _table_columns[table_name] = cols

# Enrich _col_meta with AI-derived signals (has_uom, high_confidence_method).
for _tbl, _col_dict in _col_meta.items():
    _uom_cols: set[str] = set()
    _ai_uom = st.session_state.get("AI_var_UoM", {})
    if _tbl in _ai_uom:
        _uom_df = _ai_uom[_tbl]
        if "name" in _uom_df.columns and "UoM" in _uom_df.columns:
            _uom_cols = set(
                _uom_df.loc[_uom_df["UoM"].astype(str).str.strip() != "", "name"].astype(str)
            )
    _high_conf_cols: set[str] = set()
    _ai_methode = st.session_state.get("AI_var_Methode", {})
    if _tbl in _ai_methode:
        _meth_df = _ai_methode[_tbl]
        if "name" in _meth_df.columns and "confidence" in _meth_df.columns:
            _high_conf_cols = set(
                _meth_df.loc[
                    _meth_df["confidence"].astype(str).str.strip().str.lower() == "high",
                    "name",
                ].astype(str)
            )
    for _col, _info in _col_dict.items():
        _info["has_uom"] = _col in _uom_cols
        _info["high_confidence_method"] = _col in _high_conf_cols

if not _table_columns:
    st.warning("⚠️ No columns found in the loaded data.")
    st.stop()

# --------------- auto-guess bucket for a column ---------------
_TEMPORAL_DTYPES = {"date", "datetime", "time", "datetimestamp", "gday", "gyear", "gmonth", "gyearmonth", "gmonthday"}

_SPATIAL_X_PATTERN = re.compile(
    r"\b(easting|longitude|lon|lng|x_coord|xcoord|x_pos|xpos)\b"
    r"|(?<![a-z])x(?![a-z])",
    re.IGNORECASE,
)
_SPATIAL_Y_PATTERN = re.compile(
    r"\b(northing|latitude|lat|y_coord|ycoord|y_pos|ypos)\b"
    r"|(?<![a-z])y(?![a-z])",
    re.IGNORECASE,
)
_SPATIAL_Z_PATTERN = re.compile(
    r"\b(altitude|elevation|height|z_coord|zcoord|z_pos|zpos)\b"
    r"|(?<![a-z])z(?![a-z])",
    re.IGNORECASE,
)
_SPATIAL_CRS_PATTERN = re.compile(
    r"\b(crs|epsg|srid|proj|projection|reference.?system|coord.?ref|geom|geometry|wkt|wkb)\b",
    re.IGNORECASE,
)

_FOI_ID_PATTERN = re.compile(
    r"\b(id|ids|identifier|identifiers|site[._-]?id|sample[._-]?id|station[._-]?id"
    r"|location[._-]?id|plot[._-]?id|point[._-]?id|obs[._-]?id|observation[._-]?id"
    r"|key|code|uuid|guid|pk|primary[._-]?key|ref|reference)\b",
    re.IGNORECASE,
)

# Combined pattern for bucket auto-detection — union of all role patterns.
_SPATIAL_PATTERN = re.compile(
    "|".join(
        p.pattern for p in [
            _SPATIAL_X_PATTERN,
            _SPATIAL_Y_PATTERN,
            _SPATIAL_Z_PATTERN,
            _SPATIAL_CRS_PATTERN,
        ]
    ),
    re.IGNORECASE,
)


@st.cache_data(show_spinner=False)
def _check_epsg(code: int, api_key: str) -> tuple[bool | None, str]:
    """Validate an EPSG code via the MapTiler Coordinates API.

    Returns (True, crs_name) if valid, (False, msg) if not found,
    or (None, msg) on network/config error.
    """
    if not api_key:
        return None, "No MapTiler API key configured"
    try:
        url = f"https://api.maptiler.com/coordinates/search/EPSG:{int(code)}.json"
        resp = requests.get(url, params={"key": api_key, "limit": 5}, timeout=5)
        resp.raise_for_status()
        for result in resp.json().get("results", []):
            rid = result.get("id", {})
            if rid.get("authority") == "EPSG" and int(rid.get("code", -1)) == int(code):
                return True, result.get("name", f"EPSG:{code}")
        return False, f"EPSG:{code} not found"
    except Exception as exc:
        return None, str(exc)


def _guess_spatial_role(col_name: str) -> str:
    """Best-guess spatial role for a column name: X, Y, Z, or XY reference system."""
    name = col_name.lower()
    if _SPATIAL_Z_PATTERN.search(name):
        return "Z"
    if _SPATIAL_X_PATTERN.search(name):
        return "X"
    if _SPATIAL_Y_PATTERN.search(name):
        return "Y"
    if _SPATIAL_CRS_PATTERN.search(name):
        return "XY reference system"
    return None


def _guess_bucket(info: dict) -> str:
    """Return the best-guess bucket name for a column based on its metadata."""

    if info["datatype"] in _TEMPORAL_DTYPES:
        return "Temporal"

    if _SPATIAL_PATTERN.search(info["name"]) or _SPATIAL_PATTERN.search(info["description"]):
        return "FOI - Spatial Information"

    if _FOI_ID_PATTERN.search(info["name"]) or _FOI_ID_PATTERN.search(info["description"]):
        return "Feature of Interest (FOI) - ID"

    # AI-derived signals: a non-empty UoM or a high-confidence method are indicators of an Observed Property.
    if info.get("has_uom") or info.get("high_confidence_method"):
        return "Observed Property"

    #return "Unsorted"
    return "FOI - Attribute"

# --------------- initialise buckets in session state (per table) ---------------
if "column_buckets" not in st.session_state:
    st.session_state["column_buckets"] = {}

for tbl, cols in _table_columns.items():
    if tbl not in st.session_state["column_buckets"]:
        # First time: auto-guess
        buckets = {b: [] for b in BUCKET_NAMES}
        for col in cols:
            bucket = _guess_bucket(_col_meta[tbl][col])
            buckets[bucket].append(col)

        # Keep only the leftmost FOI-ID match; demote the rest to Unsorted.
        foi_id_key = "Feature of Interest (FOI) - ID"
        if len(buckets[foi_id_key]) > 1:
            # Preserve original left-to-right order (cols list) to find leftmost.
            ordered = [c for c in cols if c in buckets[foi_id_key]]
            buckets["Unsorted"] = ordered[1:] + buckets["Unsorted"]
            buckets[foi_id_key] = [ordered[0]]
        # Fallback: if still no FOI ID, assign the leftmost column that is
        # currently in Unsorted (i.e. not already Temporal or Spatial).
        if not buckets[foi_id_key]:
            for col in cols:
                if col in buckets["Unsorted"]:
                    buckets["Unsorted"].remove(col)
                    buckets[foi_id_key].append(col)
                    break

        st.session_state["column_buckets"][tbl] = buckets
    else:
        # Sync: add new columns, remove stale ones
        existing = {
            item
            for bucket_items in st.session_state["column_buckets"][tbl].values()
            for item in bucket_items
        }
        for col in cols:
            if col not in existing:
                st.session_state["column_buckets"][tbl]["Unsorted"].append(col)
        col_set = set(cols)
        for bucket in st.session_state["column_buckets"][tbl]:
            st.session_state["column_buckets"][tbl][bucket] = [
                c for c in st.session_state["column_buckets"][tbl][bucket] if c in col_set
            ]

# Remove tables that no longer exist
for old_tbl in list(st.session_state["column_buckets"]):
    if old_tbl not in _table_columns:
        del st.session_state["column_buckets"][old_tbl]

# --------------- loop-invariant UI constants ---------------
_TEMPORAL_OPTIONS = ["sosa:phenomenonTime", "sosa:resultTime"]
_SPATIAL_OPTIONS = ["X", "Y", "Z", "XY reference system", "Z reference system", "WKT geometry", "BBOX"]
_TEMPORAL_PRECISION = {
    "Year": "%Y",
    "Year-Month": "%Y-%m",
    "Date": "%Y-%m-%d",
    "DateTime (minute)": "%Y-%m-%dT%H:%M",
    #"DateTime (second)": "%Y-%m-%dT%H:%M:%S",
}
# BUCKET_NAMES order (div:nth-of-type skips the injected <style> element):
# 1=Unsorted, 2=FOI-ID, 3=FOI-Spatial, 4=FOI-Attribute, 5=ObservedProperty, 6=Temporal
_BUCKET_HINTS = {
    2: "Including primary keys columns \A and foreign keys columns",
    4: "Descriptive attribute (e.g. country)",
    5: "Observation, requiring more \A information like unit and/or \A procedure (e.g. temperature)",
}

_HINT_CSS = "\n".join(
    f"""
    .sortable-component > div:nth-of-type({n}) .sortable-container-header::after {{
        content: "{hint}";
        display: block;
        font-size: 0.72em;
        font-weight: normal;
        opacity: 0.65;
        white-space: pre-line;
        line-height: 1.3;
        padding-top: 2px;
    }}"""
    for n, hint in _BUCKET_HINTS.items()
)

# --------------- render per-table tabs ---------------
tab_labels = list(_table_columns.keys())
_tabs = st.tabs(tab_labels)


for tab, tbl in zip(_tabs, tab_labels):
    with tab:
        tbl_buckets = st.session_state["column_buckets"][tbl]

        items = [
            {"header": bucket, "items": tbl_buckets[bucket]}
            for bucket in BUCKET_NAMES
        ]
        

        # Key based on the *set* of columns (not their bucket assignments).
        # This means the component is stable (no remount) while the user
        # drags items between buckets, eliminating flicker. It only remounts
        # when the actual column list changes (i.e. new data was loaded).
        _all_cols = sorted(col for bucket_cols in tbl_buckets.values() for col in bucket_cols)
        _col_hash = hashlib.md5(json.dumps(_all_cols).encode()).hexdigest()[:8]

        st.markdown("#### Drag columns between the buckets below")


        # Data preview for the current table
        with st.expander(f"Table: {tbl}", expanded=False):
            st.dataframe(st.session_state["tabular_data_dict_preview"][tbl], width='stretch')

        sorted_items = sort_items(
            items,
            multi_containers=True,
            direction="vertical",
            custom_style=_HINT_CSS,
            key=f"sortable_{tbl}_{_col_hash}",
        )

        # Persist. st.rerun() keeps the expanders below in sync with the
        # new bucket assignment. The stable key above ensures the sortable
        # component is NOT remounted on that rerun → no flicker.
        if sorted_items:
            new_buckets = {entry["header"]: entry["items"] for entry in sorted_items}
            if new_buckets != tbl_buckets:
                st.session_state["column_buckets"][tbl] = new_buckets
                st.rerun()



        # --- Temporal sub-type selection ---
        temporal_cols = tbl_buckets.get("Temporal", [])
        temporal_formats_for_table = {}
        with st.expander("Deepdive Temporal Column Types", expanded=True):
            
            # initialise per-table dict
            if "temporal_deepdive" not in st.session_state:
                st.session_state["temporal_deepdive"] = {}
            if "temporal_precision" not in st.session_state:
                st.session_state["temporal_precision"] = {}
            if tbl not in st.session_state["temporal_deepdive"]:
                st.session_state["temporal_deepdive"][tbl] = {}
            if tbl not in st.session_state["temporal_precision"]:
                st.session_state["temporal_precision"][tbl] = {}
            # clean up stale entries
            st.session_state["temporal_deepdive"][tbl] = {
                c: v for c, v in st.session_state["temporal_deepdive"][tbl].items()
                if c in temporal_cols or c == "__fit_for_all__"
            }
            st.session_state["temporal_precision"][tbl] = {
                c: v for c, v in st.session_state["temporal_precision"][tbl].items()
                if c in temporal_cols or c == "__fit_for_all__"
            }

            if temporal_cols:
                st.caption(
                    "For each temporal column, choose the appropriate SOSA type: \n "
                    " - [phenomenonTime](https://www.w3.org/TR/vocab-ssn/#SOSAphenomenonTime) - The time that the Result of an Observation applies to the FeatureOfInterest. (e.g. time of sampling) \n "
                    " - [resultTime](https://www.w3.org/TR/vocab-ssn/#SOSAresultTime) - The result time is the instant of time when the Observation was completed. (e.g. time in lab of measurement)"
                )

                
                for tcol in temporal_cols:
                    cur = st.session_state["temporal_deepdive"][tbl].get(tcol, _TEMPORAL_OPTIONS[0])
                    description_tcol = st.session_state["metadata_df"][tbl][st.session_state["metadata_df"][tbl]["name"]==tcol]["description"].iloc[0]
                    desc = "-> " + description_tcol if description_tcol else ""
                    st.markdown(
                        f"**{tcol}**"
                        + (f"<br><span style='font-size:0.8em;opacity:0.6;'>{desc}</span>" if desc else ""),
                        unsafe_allow_html=True,
                    )
                    sel = st.radio(
                        tcol,
                        _TEMPORAL_OPTIONS,
                        label_visibility="collapsed",
                        index=_TEMPORAL_OPTIONS.index(cur),
                        horizontal=True,
                        key=f"temporal_type_{tbl}_{tcol}",
                    )
                    st.session_state["temporal_deepdive"][tbl][tcol] = sel

                st.session_state["temporal_deepdive"][tbl].pop("__fit_for_all__", None)
                st.session_state["temporal_precision"][tbl].pop("__fit_for_all__", None)
            else:
                st.info("No temporal columns selected. Drag and drop temporal columns in the 'Temporal' bucket or indicate a 'fit-for-all' timestamp manually below.", icon="ℹ️")

                # COMMENT: Do we need to also implement dateranges? Of Periodes; e.g. Q1 -> iso notation 2026-01/P3M (periode of 3 months starting Jan 1st, 2026) https://en.wikipedia.org/wiki/ISO_8601#Durations
                
                fit_for_all_type = st.radio(
                    "Temporal type",
                    _TEMPORAL_OPTIONS,
                    horizontal=True,
                    key=f"temporal_fit_for_all_type_{tbl}",
                )
                fit_precision = st.selectbox(
                    "Datetime format input",
                    help = "indicate the format of the temporal column(s) in this table, if you want to apply a fit-for-all format for parsing timestamps. This will be applied to all columns in the 'Temporal' bucket that don't have an individual format specified above.",
                    options=list(_TEMPORAL_PRECISION.keys()),
                    index=list(_TEMPORAL_PRECISION.keys()).index("DateTime (minute)"),
                    key=f"temporal_fit_for_all_precision_{tbl}",
                )

                today = date.today()
                default_year = max(1, min(9999, today.year))
                fit_value = ""

                if fit_precision == "Year":
                    year_val = st.number_input(
                        "Year",
                        min_value=1,
                        max_value=9999,
                        value=default_year,
                        step=1,
                        key=f"temporal_fit_for_all_year_{tbl}",
                    )
                    fit_value = f"{int(year_val):04d}"
                elif fit_precision == "Year-Month":
                    ym_year_col, ym_month_col = st.columns(2)
                    with ym_year_col:
                        year_val = st.number_input(
                            "Year",
                            min_value=1,
                            max_value=9999,
                            value=default_year,
                            step=1,
                            key=f"temporal_fit_for_all_yearmonth_year_{tbl}",
                        )
                    with ym_month_col:
                        month_val = st.selectbox(
                            "Month",
                            options=list(range(1, 13)),
                            index=today.month - 1,
                            key=f"temporal_fit_for_all_yearmonth_month_{tbl}",
                        )
                    fit_value = f"{int(year_val):04d}-{int(month_val):02d}"
                elif fit_precision == "Date":
                    date_val = st.date_input(
                        "Date",
                        value=today,
                        key=f"temporal_fit_for_all_date_{tbl}",
                    )
                    fit_value = date_val.isoformat()
                elif fit_precision == "DateTime (minute)":
                    dt_date_col, dt_time_col = st.columns(2)
                    with dt_date_col:
                        date_val = st.date_input(
                            "Date",
                            value=today,
                            key=f"temporal_fit_for_all_dt_min_date_{tbl}",
                        )
                    with dt_time_col:
                        time_val = st.time_input(
                            "Time",
                            value=datetime.now().replace(second=0, microsecond=0).time(),
                            step=timedelta(minutes=1),
                            key=f"temporal_fit_for_all_dt_min_time_{tbl}",
                        )
                    fit_value = f"{date_val.isoformat()}T{time_val.strftime('%H:%M')}"
                # 
                # BUG: seconds con't be input properly
                # else:
                #     dt_date_col, dt_time_col = st.columns(2)
                #     with dt_date_col:
                #         date_val = st.date_input(
                #             "Date",
                #             value=today,
                #             key=f"temporal_fit_for_all_dt_sec_date_{tbl}",
                #         )
                #     with dt_time_col:
                #         time_val = st.time_input(
                #             "Time",
                #             value=datetime.now().replace(microsecond=0).time(),
                #             step=timedelta(seconds=1),
                #             key=f"temporal_fit_for_all_dt_sec_time_{tbl}",
                #         )
                #     fit_value = f"{date_val.isoformat()}T{time_val.strftime('%H:%M:%S')}"

                fit_format = _TEMPORAL_PRECISION[fit_precision]
                st.caption(f"Check: `{fit_value}` | Format: `{fit_format}`")

                st.session_state["temporal_deepdive"][tbl]["__fit_for_all__"] = {
                    fit_value: fit_for_all_type
                }
                st.session_state["temporal_precision"][tbl]["__fit_for_all__"] = fit_precision

        
        # --- Spatial sub-type selection ---

        spatial_cols = tbl_buckets.get("FOI - Spatial Information", [])


        if "spatial_deepdive" not in st.session_state:
            st.session_state["spatial_deepdive"] = {}
        if tbl not in st.session_state["spatial_deepdive"]:
            st.session_state["spatial_deepdive"][tbl] = {}
        # clean up stale entries
        st.session_state["spatial_deepdive"][tbl] = {
            c: v for c, v in st.session_state["spatial_deepdive"][tbl].items()
            if c in spatial_cols
        }
       
        with st.expander("Deepdive Spatial Column Types", expanded=True):
            if spatial_cols:
                st.caption(
                    "For each spatial column, provide some more context:  \n"
                    #"please provide EPSG information [epsg.io](https://epsg.io/) "
                )



                set_columns = st.columns(max(4, len(spatial_cols)))

                for i, scol in enumerate(spatial_cols):
                    cur = st.session_state["spatial_deepdive"][tbl].get(scol)
                    if cur is None or cur not in _SPATIAL_OPTIONS:
                        cur = _guess_spatial_role(scol)
                    description_scol = st.session_state["metadata_df"][tbl][st.session_state["metadata_df"][tbl]["name"]==scol]["description"].iloc[0]
                    with set_columns[i]:
                        desc = "-> " + description_scol if description_scol else ""
                        st.markdown(
                            f"**{scol}**"
                            + (f"<br><span style='font-size:0.8em;opacity:0.6;'>{desc}</span>" if desc else ""),
                            unsafe_allow_html=True,
                        )
                        sel = st.selectbox(
                            scol,
                            label_visibility="collapsed",
                            options=[None]+_SPATIAL_OPTIONS,
                            index=(_SPATIAL_OPTIONS.index(cur)+1 if cur in _SPATIAL_OPTIONS else 0),
                            key=f"spatial_type_{tbl}_{scol}",
                        )
                    st.session_state["spatial_deepdive"][tbl][scol] = sel

            # Roles that are not yet assigned to any column
            assigned_roles = {
                v for v in st.session_state["spatial_deepdive"][tbl].values()
                if v is not None
            }
            unassigned_roles = [r for r in _SPATIAL_OPTIONS if r not in assigned_roles]

            # Remove fit-for-all entries whose role is now covered by a column
            if "spatial_fit_for_all" not in st.session_state:
                st.session_state["spatial_fit_for_all"] = {}
            if tbl not in st.session_state["spatial_fit_for_all"]:
                st.session_state["spatial_fit_for_all"][tbl] = {}
            st.session_state["spatial_fit_for_all"][tbl] = {
                role: v
                for role, v in st.session_state["spatial_fit_for_all"][tbl].items()
                if role not in assigned_roles
            }

            if unassigned_roles:
                st.markdown("**Alternative: Fit-for-all values for unassigned context**")
                st.caption(
                    "The following spatial context items have no column assigned. "
                    "You can provide a default value for all observations."
                )

                set_unassigned_columns = st.columns(max(4, len(unassigned_roles)))

                for i, role in enumerate(unassigned_roles):
                    # Use the stored value if the key exists; only fall back to
                    # the suggested default when the role has never been visited.
                    if role in st.session_state["spatial_fit_for_all"][tbl]:
                        existing_fit = st.session_state["spatial_fit_for_all"][tbl][role]
                    elif role == "XY reference system":
                        existing_fit = 4326
                    elif role == "Z reference system":
                        existing_fit = 9389
                    else:
                        existing_fit = None

                    help_text = "Please provide an [EPSG code](https://epsg.io/) (e.g. 4326 for WGS 84, 9389 for EVRF2019 height)  \n Enter '0' to delete" if "reference system" in role else None

                    if role not in {"WKT geometry", "BBOX"}:
                        fit = set_unassigned_columns[i].number_input(
                            role,
                            value=existing_fit,
                            help=help_text,
                            key=f"spatial_fit_for_all_{tbl}_{role}"
                            )
                    else:
                        fit = set_unassigned_columns[i].text_input(
                            role,
                            value=existing_fit if existing_fit is not None else "",
                            help=help_text,
                            key=f"spatial_fit_for_all_{tbl}_{role}"
                        )
                    st.session_state["spatial_fit_for_all"][tbl][role] = fit

                    if "reference system" in role and fit:
                        _maptiler_key = st.secrets.get("MAPTILER", {}).get("api_key", "")
                        _epsg_valid, _epsg_name = _check_epsg(int(fit), _maptiler_key)
                        if _epsg_valid is True:
                            set_unassigned_columns[i].caption(f"✅ [{_epsg_name}](https://epsg.io/{int(fit)})")
                        elif _epsg_valid is False:
                            set_unassigned_columns[i].caption(f"❌ EPSG:{int(fit)} not found")

                # Rebuild fit-for-all dict from all accumulated role→value pairs.
                st.session_state["spatial_deepdive"][tbl]["__fit_for_all__"] = {
                    role: fit
                    for role, fit in st.session_state["spatial_fit_for_all"][tbl].items()
                }

            




        # --- getting information in session state ---

        # Write bucket assignments into the 'element' column of metadata_df
        _BUCKET_TO_URI = {
            "Feature of Interest (FOI) - ID": "sosa:FeatureOfInterest",
            "Observed Property": "sosa:observedProperty",
            "FOI - Spatial Information": "geo:Feature",
            "FOI - Attribute": "ssn:Property",
        }
        meta_df = st.session_state[meta_key][tbl]
        col_to_uri = {}
        for bucket, bucket_cols in tbl_buckets.items():
            if bucket == "Temporal":
                # use per-column sub-type selection
                tt = st.session_state.get("temporal_deepdive", {}).get(tbl, {})
                for col in bucket_cols:
                    col_to_uri[col] = tt.get(col, "sosa:phenomenonTime")
                continue
            uri = _BUCKET_TO_URI.get(bucket)
            if not uri:
                continue
            for col in bucket_cols:
                col_to_uri[col] = uri
        meta_df["element"] = meta_df["name"].map(col_to_uri).fillna("")
        st.session_state[meta_key][tbl] = meta_df

        # --- Derive temporal extent ---
        if "temporal_extent" not in st.session_state:
            st.session_state["temporal_extent"] = {}
        _td = st.session_state.get("temporal_deepdive", {}).get(tbl, {})
        _fit_all = _td.get("__fit_for_all__")
        _t_begin = None
        _t_end = None
        if _fit_all:
            # fit_for_all is {fit_value: role}; fixed-point extent
            _fit_value = next(iter(_fit_all))
            _t_begin = _fit_value
            _t_end = _fit_value
        else:
            _temporal_bucket = tbl_buckets.get("Temporal", [])
            _phenom_cols = [c for c in _temporal_bucket if _td.get(c) == "sosa:phenomenonTime"]
            _result_cols = [c for c in _temporal_bucket if _td.get(c) == "sosa:resultTime"]
            _time_col = (_phenom_cols or _result_cols or [None])[0]
            if _time_col:
                _full_df = st.session_state.get("tabular_data_dict", {}).get(tbl)
                if _full_df is not None and _time_col in _full_df.columns:
                    _parsed = pd.to_datetime(_full_df[_time_col], errors="coerce").dropna()
                    if not _parsed.empty:
                        _t_begin = _parsed.min().isoformat()
                        _t_end = _parsed.max().isoformat()
        st.session_state["temporal_extent"][tbl] = {"begin": _t_begin, "end": _t_end}

        st.session_state[meta_key][tbl]