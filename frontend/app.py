
import os

import requests
import streamlit as st

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000"
)

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI Document Assistant")
st.caption(
    "Chat with your documents using RAG, "
    "ChromaDB, and Qwen 2.5"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "document" not in st.session_state:
    st.session_state.document = None


# Sidebar: PDF upload
with st.sidebar:
    st.header("Your Documents")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        max_upload_size=10
    )

    if uploaded_file is not None:
        if st.button(
            "Process Document",
            use_container_width=True
        ):
            with st.spinner("Processing PDF and creating embeddings..."):
                try:
                    response = requests.post(
                        f"{API_URL}/upload",
                        files={
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                "application/pdf",
                            )
                        },
                        timeout=180,
                    )
                    response.raise_for_status()

                    data = response.json()
                    st.session_state.document = data
                    st.session_state.messages = []

                    st.success("Document indexed successfully!")

                except requests.RequestException as exc:
                    st.error(f"Upload failed: {exc}")

    if st.session_state.document:
        doc = st.session_state.document

        st.divider()
        st.subheader("Active Document")
        st.write(doc["filename"])
        st.metric("Pages", doc["pages"])
        st.metric("Chunks", doc["total_chunks"])

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# Chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("View Sources"):
                for source in message["sources"]:
                    st.markdown(
                        f"**Source {source['source']}** "
                        f"— Similarity: {source['similarity']:.3f}"
                    )
                    st.write(source["text"])


# Ask questions
question = st.chat_input(
    "Ask a question about your document...",
    disabled=st.session_state.document is None
)

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            try:
                response = requests.post(
                    f"{API_URL}/ask",
                    json={
                        "question": question,
                        "top_k": 3,"document_id": st.session_state.document["document_id"],
                        },
                    timeout=180,
                )
                response.raise_for_status()

                data = response.json()
                answer = data["answer"]
                sources = data.get("sources", [])

                st.markdown(answer)

                if sources:
                    with st.expander("View Sources"):
                        for source in sources:
                            st.markdown(
                                f"**Source {source['source']}** "
                                f"— Similarity: "
                                f"{source['similarity']:.3f}"
                            )
                            st.write(source["text"])

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

            except requests.RequestException as exc:
                st.error(f"Unable to get an answer: {exc}")
