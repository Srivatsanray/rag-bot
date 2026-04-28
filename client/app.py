import streamlit as st

from state.session import setup_session_state, is_chat_ready
from components.chat import (
    render_chat_history,
    render_download_chat_history,
    render_uploaded_files_expander,
    render_user_input
)
from components.sidebar import (
    render_view_selector,
    sidebar_file_upload,
    sidebar_utilities
)
from components.inspector import render_inspect_query


def main():
    st.set_page_config(page_title="RAG PDFBot", layout="centered")
    st.title("RAG PDFBot")
    st.caption("Chat with multiple PDFs")

    setup_session_state()

    with st.sidebar:
        with st.expander("Configuration", expanded=True):
            sidebar_file_upload()

        view_option = render_view_selector()
        sidebar_utilities()

    if st.session_state.get("chat_history"):
        render_download_chat_history()

    if not st.session_state.get("pdf_files"):
        st.info("Please upload and submit PDFs to start chatting.")

    if st.session_state.get("unsubmitted_files"):
        st.warning("New PDFs uploaded. Please submit before chatting.")

    if st.session_state.get("chat_ready") and st.session_state.get("pdf_files"):
        render_uploaded_files_expander()

    if view_option == "Chat":
        if st.session_state.get("chat_history"):
            render_chat_history()
        render_user_input()

    elif view_option == "Inspector":
        if is_chat_ready():
            render_inspect_query()


if __name__ == "__main__":
    main()
