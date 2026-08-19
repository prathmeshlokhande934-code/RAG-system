import os
import tempfile

import streamlit as st

from rag import (
    load_pdf,
    split_documents,
    create_vectorstore,
    search_documents,
    generate_answer,
)

st.set_page_config(
    page_title="RAG PDF Assistant",
    page_icon="📄",
    layout="wide",
)

# ----------------------------
# Session state initialization
# ----------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []  # list of filenames already embedded

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of dicts: {role, content, sources?}


def process_uploaded_pdfs(uploaded_files):
    """Save uploaded PDFs to disk, chunk them, and add them to the vectorstore."""
    new_files = [
        f for f in uploaded_files
        if f.name not in st.session_state.processed_files
    ]

    if not new_files:
        return

    with st.spinner(f"Processing {len(new_files)} PDF(s)..."):
        for uploaded_file in new_files:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".pdf"
            ) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            try:
                documents = load_pdf(tmp_path, uploaded_file.name)
                chunks = split_documents(documents)
                st.session_state.vectorstore = create_vectorstore(chunks)
                st.session_state.processed_files.append(uploaded_file.name)
            finally:
                os.unlink(tmp_path)

    st.success(f"Added {len(new_files)} PDF(s) to the knowledge base.")


# ----------------------------
# Sidebar - upload & manage PDFs
# ----------------------------
with st.sidebar:
    st.title("📄 RAG PDF Assistant")
    st.caption("Upload PDFs, then ask questions about their content.")

    st.divider()

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button("Process PDFs", type="primary", use_container_width=True):
            if not os.getenv("OPENROUTER_API_KEY"):
                st.error(
                    "OPENROUTER_API_KEY is not set. Please add it to your "
                    ".env file before processing PDFs."
                )
            else:
                process_uploaded_pdfs(uploaded_files)

    st.divider()

    if st.session_state.processed_files:
        st.subheader("Knowledge base")
        st.caption(f"{len(st.session_state.processed_files)} PDF(s) indexed")
        selected_pdfs = st.multiselect(
            "Search within",
            options=st.session_state.processed_files,
            default=st.session_state.processed_files,
            help="Only these PDFs will be used to answer your questions.",
        )
        for name in st.session_state.processed_files:
            st.write(f"• {name}")
    else:
        selected_pdfs = []
        st.info("No PDFs indexed yet. Upload and process files to get started.")

    st.divider()

    if st.button("Clear chat history", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# ----------------------------
# Main - chat interface
# ----------------------------
st.header("Chat with your PDFs")

if not st.session_state.processed_files:
    st.info("👈 Upload and process at least one PDF from the sidebar to start chatting.")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for i, source in enumerate(message["sources"], start=1):
                    st.markdown(f"**{i}. {source['name']}**")
                    st.caption(source["snippet"])

question = st.chat_input(
    "Ask a question about your PDFs...",
    disabled=not st.session_state.processed_files,
)

if question:
    if not selected_pdfs:
        st.warning("Select at least one PDF from the sidebar to search within.")
    else:
        st.session_state.chat_history.append(
            {"role": "user", "content": question}
        )
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching PDFs and generating answer..."):
                documents = search_documents(
                    st.session_state.vectorstore,
                    question,
                    selected_pdfs,
                )
                answer = generate_answer(question, documents)

            st.markdown(answer)

            sources = []
            for doc in documents:
                sources.append(
                    {
                        "name": doc.metadata.get("source", "Unknown"),
                        "snippet": doc.page_content[:300] + "...",
                    }
                )

            if sources:
                with st.expander("Sources"):
                    for i, source in enumerate(sources, start=1):
                        st.markdown(f"**{i}. {source['name']}**")
                        st.caption(source["snippet"])

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
            }
        )