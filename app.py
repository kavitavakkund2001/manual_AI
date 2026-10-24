
import requests
import numpy as np
import pymupdf
import streamlit as st

OLLAMA_URL = "http://localhost:11434"
CHAT_MODEL = "qwen3:4b"
EMBED_MODEL = "embeddinggemma"

st.set_page_config(page_title="Manual AI Assistant", page_icon="📘")
st.title("📘 Local Manual Troubleshooting Assistant")
st.caption("Your PDF is processed locally on your Mac.")

def split_into_chunks(text, chunk_size=450, overlap=80):
    """Split text into overlapping word groups."""
    words = text.split()
    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + chunk_size])
        if chunk.strip():
            chunks.append(chunk)

    return chunks


def extract_pdf(pdf_bytes):
    """Extract text page by page, retaining PDF page numbers."""
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    records = []

    for page_index, page in enumerate(doc):
        text = page.get_text("text").strip()

        if not text:
            continue

        for chunk in split_into_chunks(text):
            records.append({
                "page": page_index + 1,
                "text": chunk
            })

    doc.close()
    return records


def get_embeddings(texts):
    """Generate embeddings locally using Ollama."""
    all_embeddings = []

    # Process small batches to limit memory usage.
    for start in range(0, len(texts), 8):
        batch = texts[start:start + 8]

        response = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={
                "model": EMBED_MODEL,
                "input": batch
            },
            timeout=300
        )
        response.raise_for_status()

        all_embeddings.extend(response.json()["embeddings"])

    vectors = np.array(all_embeddings, dtype=np.float32)

    # Normalize vectors for cosine similarity.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-12)

    return vectors


def answer_question(question, records, embeddings):
    """Retrieve relevant chunks and ask the local model."""
    query_vector = get_embeddings([question])[0]

    scores = embeddings @ query_vector
    top_indices = np.argsort(scores)[::-1][:4]

    context_parts = []

    for index in top_indices:
        record = records[index]
        context_parts.append(
            f"[PDF page {record['page']}]\n{record['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""
You are a technical manual troubleshooting assistant.

Answer using the supplied manual excerpts only.

Rules:
- Explain the answer in clear, step-by-step language.
- Cite the PDF page number for each important claim.
- Separate documented facts from possible explanations.
- Never invent procedures, measurements, wiring connections,
  or fixes that are not supported by the excerpts.
- If the excerpts do not contain enough information, say so.
- For troubleshooting, explain what to check and what each
  possible result means, when the manual provides that information.

MANUAL EXCERPTS:
{context}

USER QUESTION:
{question}
"""

    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": CHAT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.1}
        },
        timeout=300
    )
    response.raise_for_status()

    answer = response.json()["message"]["content"]
    sources = [
        {
            "page": records[i]["page"],
            "text": records[i]["text"],
            "score": float(scores[i])
        }
        for i in top_indices
    ]

    return answer, sources


uploaded_file = st.file_uploader(
    "Upload a PDF manual",
    type=["pdf"]
)

if uploaded_file:
    pdf_bytes = uploaded_file.getvalue()

    if st.button("Process manual"):
        with st.spinner("Extracting text and indexing the manual..."):
            try:
                records = extract_pdf(pdf_bytes)

                if not records:
                    st.error(
                        "No selectable text was found. "
                        "This PDF may be scanned. OCR support "
                        "will be added in the next stage."
                    )
                else:
                    texts = [record["text"] for record in records]
                    embeddings = get_embeddings(texts)

                    st.session_state["records"] = records
                    st.session_state["embeddings"] = embeddings
                    st.session_state["manual_name"] = uploaded_file.name

                    st.success(
                        f"Indexed {len(records)} sections from "
                        f"{uploaded_file.name}."
                    )

            except requests.RequestException as error:
                st.error(
                    "Could not reach Ollama or generate embeddings. "
                    f"Check that Ollama is running and both models "
                    f"are installed. Details: {error}"
                )
            except Exception as error:
                st.error(f"Could not process this PDF: {error}")


if "records" in st.session_state:
    st.subheader("Ask your manual")

    st.caption(f"Loaded manual: {st.session_state['manual_name']}")

    question = st.text_area(
        "Describe your question or problem",
        placeholder=(
            "Example: Explain the startup workflow and "
            "what to check if the process fails."
        )
    )

    if st.button("Ask AI") and question.strip():
        with st.spinner("Searching the manual and generating an answer..."):
            try:
                answer, sources = answer_question(
                    question,
                    st.session_state["records"],
                    st.session_state["embeddings"]
                )

                st.markdown("### Answer")
                st.write(answer)

                st.markdown("### Retrieved manual sections")

                for source in sources:
                    with st.expander(
                        f"PDF page {source['page']}"
                    ):
                        st.write(source["text"])

            except requests.RequestException as error:
                st.error(f"Ollama request failed: {error}")
