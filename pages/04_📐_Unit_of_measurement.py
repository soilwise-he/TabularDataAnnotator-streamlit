# step1; detect where UOM are required (only int and float)
# step2; detect if any matches with ansis. if yes, take UOM from there 
# step3; parse results from LLM for UOM 
# step4; see in earlier matches of matches with ansis brought up some reasonable consensus
# step5; UI to browse QUDT
# step6; make own combination of QUDT


import streamlit as st
import pandas as pd
from ui.blocks import add_Soilwise_contact_sidebar,add_Soilwise_logo,add_clear_cache_button

st.set_page_config(page_title="Tabular Soil Data Annotation", layout="wide")
add_Soilwise_logo()
add_clear_cache_button(key_prefix="Keyword_Matcher")


DATA_TYPE_OPTIONS = ["string", "numeric", "date"]



meta_key = f"metadata_df"

# -------------------- Helper data and functions --------------------


# PIN -------------------- UI --------------------


st.markdown("#### Current state of your metadata table:")

# Create one tab per key
tab_labels = list(st.session_state[meta_key].keys())
tabs = st.tabs(tab_labels)
error_handling_tabs = st.empty()
error_tabs = []

total_size = 0
descriptions_completed = 0

for tab, key in zip(tabs, tab_labels):
    df = st.session_state[meta_key][key]
    
    total_size=+len(df)
    count_completed = df["description"].notna() & (df["description"] != "")
    descriptions_completed=+ count_completed.sum()
    with tab:
        st.dataframe(st.session_state[meta_key][key])














# -------------------- reach us --------------------
add_Soilwise_contact_sidebar()