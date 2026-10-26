"""Browser-carried downloads for small prepared reports."""

import base64
from html import escape

import streamlit as st


def download_link_html(data, file_name, mime, label):
    """Embed the file, avoiding temporary media URLs and script reruns."""
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
