"""Small, shared presentation settings for the nuggt tools."""

import base64
from html import escape

import streamlit as st


def download_link_html(data, file_name, mime, label):
    """Carry small prepared reports in the page, avoiding temporary media URLs.

    Downloading is a browser-only action: no rerun, new research, expiring media
    file, or shared server-side report cache is involved.
    """
    if mime not in ("application/pdf", "text/markdown"):
        raise ValueError("Unsupported report format.")
    if not isinstance(data, bytes) or not data:
        raise ValueError("The report download is empty.")
    if mime == "application/pdf" and not data.startswith(b"%PDF-"):
        raise ValueError("The report is not a valid PDF file.")
    encoded = base64.b64encode(data).decode("ascii")
    return (
        f'<a class="dt-download-link" target="_self" download="{escape(file_name, quote=True)}" '
        f'href="data:{mime};base64,{encoded}">{escape(label)}</a>'
    )


def render_download_link(data, file_name, mime, label):
    st.markdown(download_link_html(data, file_name, mime, label), unsafe_allow_html=True)


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
