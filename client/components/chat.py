from datetime import datetime

import pandas as pd
import streamlit as st
from utils.helpers import process_user_input


def render_citation_expanders(chunks: list[dict]):
    for i, chunk in enumerate(chunks, start=1):
        with st.expander(f"[{i}] {chunk['doc_name']} — Page {chunk['page_number']}"):
            st.caption(chunk["chunk_text"])


def render_user_input():
    disable_input = (
        st.session_state.get("unsubmitted_files", False)
        or not st.session_state.get("pdf_files")
        or not st.session_state.get("chat_ready")
    )

    question = st.chat_input(
        "Ask a question about the uploaded PDFs", disabled=disable_input
    )

    if not question:
        return

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("ai"):
        with st.spinner("Thinking..."):
            try:
                result = process_user_input(question)
                answer = result["answer"]
                chunks = result["chunks"]
                low_confidence = result["low_confidence"]

                if low_confidence:
                    st.warning(
                        "Low confidence: the documents may not contain a direct answer to this question."
                    )

                st.markdown(answer)
                render_citation_expanders(chunks)

                st.session_state.chat_history.append(
                    (question, answer, chunks, low_confidence, datetime.now())
                )
            except Exception as e:
                st.error(f"Error: {str(e)}")


def render_uploaded_files_expander():
    pdf_files = st.session_state.get("pdf_files", [])
    if pdf_files and not st.session_state.get("unsubmitted_files"):
        with st.expander("Uploaded Files"):
            for name in pdf_files:
                st.markdown(f"- {name}")


def render_chat_history():
    for q, a, chunks, low_confidence, *_ in st.session_state.get("chat_history", []):
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("ai"):
            if low_confidence:
                st.warning(
                    "Low confidence: the documents may not contain a direct answer to this question."
                )
            st.markdown(a)
            render_citation_expanders(chunks)


def render_download_chat_history():
    history = st.session_state.get("chat_history", [])
    if not history:
        return

    df = pd.DataFrame(
        [
            {
                "Question": q,
                "Answer": a,
                "Sources": ", ".join(
                    f"{c['doc_name']} p.{c['page_number']}" for c in chunks
                ),
                "Timestamp": ts,
            }
            for q, a, chunks, _, ts in history
        ]
    )

    with st.expander("Download Chat History"):
        st.download_button(
            "Download as CSV",
            data=df.to_csv(index=False),
            file_name="chat_history.csv",
            mime="text/csv",
        )
