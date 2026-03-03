import streamlit as st
from ui.blocks import add_Soilwise_logo

add_Soilwise_logo()


st.set_page_config(page_title="Tabular Soil Data Annotation", layout="wide")


# -------------------- UI --------------------
st.title("🌱 Tabular Soil Data Annotation")

st.markdown("""🧪 Experimental tool developed within the Soilwise project. All feedback is welcome! --> 
            <a href='mailto:max.vercruyssen@vlaanderen.be' target='_blank'>Contact us trough mail</a>
            or
            <a href="https://outlook.office.com/bookwithme/user/ab3771ccfeed4c5c83ca7a43d657865d@vlaanderen.be?" target="_blank">book a meeting with us</a>""",
            unsafe_allow_html=True)


st.markdown("""
### Making FAIR soil data easier — together

FAIR data publication is a shared ambition in the soil science community.  
At the same time, we all know the reality: **making datasets truly Findable, Accessible, Interoperable, and Reusable takes time, expertise, and effort from data providers**.

Metadata standards, variable descriptions, units, methods, vocabularies…  
All essential — and sometimes overwhelming.

**You are not alone in this.**
""")

st.markdown("""
---

### Why this tool exists

Within the **Soilwise project**, we aim to **lower the effort required to annotate and document soil datasets**, without compromising scientific quality or FAIR principles.

This application is an **experimental data‑annotation tool** designed to:

- ✍️ Support data providers in describing their datasets clearly and consistently  
- 🧩 Help translate raw tables into **FAIR‑ready, reusable data assets**  
- 🚀 Reduce friction between data creation and data publication  
- 🤝 Encourage shared practices across the soil community
""")

st.markdown("""
---

### What you can do here

With this tool, you can:

- Annotate variables in your dataset in a guided way  
- Reuse contextual information from documents or protocols  (we also hate to type the same thing twice)
- Improve clarity, consistency, and reusability of your data  
- Prepare your dataset for FAIR publication with **less manual effort**

This is an **experimental tool**: your feedback helps shape how we collectively improve data annotation workflows in Soilwise.
""")

st.markdown("""
---

### 🌍 A community effort

FAIR data is not just a technical requirement — it is a **collective investment** in better science, transparency, and long‑term reuse.

Thank you for contributing your time, expertise, and publishing your data FAIR.  
Let’s make soil data more reusable — **step by step**.
""")

st.markdown("""---""")
st.success("👉 Get started by going to the next page (in the sidebar) and uploading your dataset or selecting an annotation workflow.")




