import streamlit as st

# Optional but nice to have
st.set_page_config(page_title="nuggt", page_icon="🧠")

# Set the title of the main page
st.title("Welcome to nuggt")

# Add a professional header and description
st.header("Choose an application from the side panel to get started:")

# Links
IDEATION_URL = "https://nuggt-ai.streamlit.app/Ideation"
DUE_DILIGENCE_URL = "https://nuggt-ai.streamlit.app/Due_Diligence"
AETHER_URL = "https://nuggt-ai.streamlit.app/Aether"
BUTA_URL = "https://nuggt-ai.streamlit.app/BUTA"

# Side panel navigation
st.sidebar.title("Applications")
st.sidebar.markdown(
    f"""
- [Ideation (Validate your idea)]({IDEATION_URL})
- [Due Diligence]({DUE_DILIGENCE_URL})
- [Aether Workshop]({AETHER_URL})
- [BUTA]({BUTA_URL})
"""
)

st.subheader("Ideation (Validate your idea)")
st.write("Validate innovative ideas using the market dynamics framework and identify new opportunities.")
st.markdown(f"[Open Ideation]({IDEATION_URL})")

st.subheader("Due Diligence")
st.write("Conduct due diligence on a company to assess its strengths, weaknesses, opportunities, and risks.")
st.markdown(f"[Open Due Diligence]({DUE_DILIGENCE_URL})")

st.subheader("Aether Workshop")
st.write("Run the Aether Workshop.")
st.markdown(f"[Open Aether Workshop]({AETHER_URL})")

st.subheader("BUTA")
st.write("Framework to analyse Beachheads (Budget, Urgency, Top-3 Fit, Access).")
st.markdown(f"[Open BUTA]({BUTA_URL})")
