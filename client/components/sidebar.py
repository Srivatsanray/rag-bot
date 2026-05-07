import streamlit as st
from types import SimpleNamespace

from state.session import is_chat_ready
from utils.helpers import process_uploaded_pdfs


def render_upload_files_button():
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"uploaded_files_{st.session_state.get('uploader_key')}"
    )

    uploaded_filenames = [f.name for f in uploaded_files] if uploaded_files else []
    session_filenames = st.session_state.get("pdf_files", [])

    if uploaded_files and uploaded_filenames != session_filenames:
        st.session_state.update(unsubmitted_files=True)

    submitted = st.button("Submit")
    return uploaded_files, submitted


def render_view_selector():
    with st.sidebar.expander("View Options", expanded=False):
        view_option = st.selectbox(
            "Select View",
            options=["Chat", "Inspector"],
            index=0,
            disabled=not is_chat_ready(),
            key="view"
        )
        return view_option


def sidebar_file_upload():
    uploaded_files, submitted = render_upload_files_button()

    if submitted:
        if uploaded_files:
            file_objs = [
                SimpleNamespace(name=f.name, type=f.type, data=f.read())
                for f in uploaded_files
            ]

            with st.spinner("Processing PDFs..."):
                try:
                    process_uploaded_pdfs(file_objs)
                    st.session_state.update(
                        chat_ready=True,
                        pdf_files=[f.name for f in file_objs],
                        unsubmitted_files=False
                    )
                    st.toast("PDFs processed successfully!")
                except Exception as e:
                    st.error(f"Error processing PDFs: {str(e)}")
        else:
            st.warning("Please upload at least one PDF before submitting.")

    return uploaded_files, submitted


def sidebar_utilities():
    with st.expander("Utilities", expanded=False):
        col1, col2, col3 = st.columns(3)

        if col1.button("Reset"):
            st.session_state.clear()
            st.toast("Session reset.")
            st.rerun()

        if col2.button("Clear Chat"):
            st.session_state.chat_history = []
            st.session_state.pdf_files = []
            st.session_state.uploader_key += 1
            st.toast("Chat and PDFs cleared.")
            st.rerun()

        if col3.button("Undo") and st.session_state.get("chat_history"):
            st.session_state.chat_history.pop()
            st.toast("Last message removed.")
            st.rerun()
