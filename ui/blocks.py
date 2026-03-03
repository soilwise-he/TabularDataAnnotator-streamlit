import streamlit as st

EMAIL = "max.vercruyssen@vlaanderen.be"
BOOKING_URL = "https://outlook.office.com/bookwithme/user/ab3771ccfeed4c5c83ca7a43d657865d@vlaanderen.be?"

def add_Soilwise_contact_sidebar():
    with st.sidebar:
        st.markdown("""
### 🌱 Let’s improve SoilWise together!

This tool is part of the Horizon Europe **SoilWise** project, and your input is essential for making it truly useful in real-world contexts.  
If you’d like to discuss the functionalities, share your experiences, or explore collaboration, we’d be glad to hear from you.

**💬 Reach out — your insights matter.**
""")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                f"""
                <a href="mailto:{EMAIL}" target="_blank">
                    <button style="padding:10px 18px; border-radius:8px; height: 5rem;
                                   background-color:#58a9f8; color:white; border:none; cursor:pointer;">
                        Email me
                    </button>
                </a>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
                <a href="{BOOKING_URL}" target="_blank">
                    <button style="padding:10px 18px; border-radius:8px; height: 5rem;
                                   background-color:#50c082; color:white; border:none; cursor:pointer;">
                        Book a meeting
                    </button>
                </a>
                """,
                unsafe_allow_html=True,
            )
def add_Soilwise_logo():
    st.logo(
    'Logo/SWlogocolor.svg',
    link="https://soilwise-he.eu/",
    )

def add_clear_cache_button(key_prefix: str = "global"):
    # Clear Cache Button aligned to the right
    col1, col2 = st.columns([5, 1])  # Adjust column proportions as needed

    with col2:
        if st.button("🔄 Clear Cache and Restart", key=f"{key_prefix}-clear-cache"):
            # Clear Streamlit caches (if you use caching)
            st.cache_data.clear()
            st.cache_resource.clear()

            # Clear session state safely
            st.session_state.clear()

            st.rerun()

