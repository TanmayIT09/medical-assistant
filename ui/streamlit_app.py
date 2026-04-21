"""Streamlit frontend for the local medical assistant."""

from __future__ import annotations

import os
from pathlib import Path

import requests
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError


try:
    API_BASE_URL = st.secrets["api_base_url"]
except (KeyError, AttributeError, StreamlitSecretNotFoundError):
    API_BASE_URL = os.getenv("STREAMLIT_API_URL", "http://localhost:8000")


def read_error_message(response: requests.Response) -> str:
    """Normalize API error messages for display in Streamlit."""
    try:
        return response.json().get("detail", "Request failed.")
    except ValueError:
        return response.text or "Request failed."


st.set_page_config(page_title="Medical Assistant", page_icon=":hospital:", layout="centered")
st.title("Medical Assistant")
st.caption("Upload a medical PDF or image, then get safe report-based guidance or ask questions.")


with st.sidebar:
    st.subheader("Safety Notice")
    st.write(
        "This tool is for informational summarization of uploaded medical documents only. "
        "It does not provide diagnoses, prescriptions, or emergency advice."
    )


uploaded_file = st.file_uploader(
    "Upload a PDF or image",
    type=["pdf", "png", "jpg", "jpeg", "bmp", "tiff", "tif"],
)

if st.button("Upload Document", disabled=uploaded_file is None):
    with st.spinner("Uploading document..."):
        try:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            response = requests.post(f"{API_BASE_URL}/upload", files=files, timeout=180)
            if response.ok:
                payload = response.json()
                st.success(
                    f"Indexed {payload['filename']} with {payload['chunks_indexed']} chunks."
                )
                st.json(payload)
                if payload.get("preview"):
                    with st.expander("Extracted Text Preview"):
                        st.text(payload["preview"])
            else:
                st.error(read_error_message(response))
        except requests.RequestException as exc:
            st.error(f"Could not reach the backend: {exc}")


if st.button("Generate Report Guidance"):
    with st.spinner("Creating report-based guidance..."):
        try:
            response = requests.get(f"{API_BASE_URL}/report-guidance", timeout=180)
            if response.ok:
                payload = response.json()
                st.subheader("Suggested Guidance")
                st.write(payload["guidance"])
                if payload.get("sources"):
                    st.caption(f"Sources: {', '.join(payload['sources'])}")
            else:
                st.error(read_error_message(response))
        except requests.RequestException as exc:
            st.error(f"Could not reach the backend: {exc}")


question = st.text_input("Ask a question about the uploaded document")

if st.button("Get Answer", disabled=not question.strip()):
    with st.spinner("Generating answer..."):
        try:
            response = requests.get(
                f"{API_BASE_URL}/ask",
                params={"question": question.strip()},
                timeout=180,
            )
            if response.ok:
                payload = response.json()
                st.subheader("Answer")
                st.write(payload["answer"])
                if payload.get("sources"):
                    st.caption(f"Sources: {', '.join(payload['sources'])}")
            else:
                st.error(read_error_message(response))
        except requests.RequestException as exc:
            st.error(f"Could not reach the backend: {exc}")


st.divider()
st.caption(f"Backend URL: {API_BASE_URL}")
st.caption(f"Frontend file: {Path(__file__).resolve()}")
