"""Small, shared presentation settings for the nuggt tools."""

import streamlit as st


def hide_text_field_hints():
    """Hide shortcut/counter footers while keeping labels, help and limits intact.

    Submission behavior is controlled separately by enter_to_submit=False on
    every input form; hiding a hint alone must never change keyboard behavior.
    """
    st.markdown(
        """
<style>
[data-testid="stTextArea"] [data-testid="InputInstructions"],
[data-testid="stTextInput"] [data-testid="InputInstructions"] {
    display: none !important;
}
</style>
""",
        unsafe_allow_html=True,
    )
