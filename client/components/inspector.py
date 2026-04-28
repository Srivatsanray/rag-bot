import streamlit as st
from state.session import is_chat_ready
from utils.helpers import get_documents_count, get_similar_chunks


def render_inspect_query():
    st.caption("Look under the hood: test raw retrieval without the LLM")

    try:
        doc_count = get_documents_count()
        st.success(f"{doc_count} chunks stored in vectorstore.")
    except Exception as e:
        st.error("Could not fetch chunk count.")
        st.code(str(e))

    query = st.chat_input(
        "Test a query against the vectorstore directly", disabled=not is_chat_ready()
    )

    if not query:
        return

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("ai"):
        with st.spinner("Searching..."):
            try:
                results = get_similar_chunks(query)
                if results:
                    st.markdown(f"Top {len(results)} matching chunks:")
                    for i, chunk in enumerate(results, start=1):
                        with st.expander(
                            f"[{i}] {chunk['doc_name']} — Page {chunk['page_number']} "
                            f"(score: {chunk['score']})"
                        ):
                            st.caption(chunk["chunk_text"])
                else:
                    st.info("No matching chunks found.")
            except Exception as e:
                st.error("Error querying vectorstore.")
                st.code(str(e))
