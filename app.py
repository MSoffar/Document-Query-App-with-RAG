import streamlit as st
import requests
import PyPDF2
from io import BytesIO
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import DensePassageRetriever
from haystack.pipelines import DocumentSearchPipeline
import openai
from docx import Document

# Set your OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]
# Initialize FAISS document store (using in-memory SQLite by default)
document_store = FAISSDocumentStore(faiss_index_factory_str="Flat")

# Create the retriever
retriever = DensePassageRetriever(
    document_store=document_store,
    query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
    passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base"
)

# Create the document search pipeline
pipeline = DocumentSearchPipeline(retriever=retriever)


def read_pdf(file):
    """Read a PDF file and convert it to text."""
    reader = PyPDF2.PdfReader(file)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text


def download_and_convert_pdf_to_text(pdf_url):
    """Download a PDF from a URL and convert it to text."""
    response = requests.get(pdf_url, timeout=50)
    response.raise_for_status()
    pdf_file = BytesIO(response.content)
    return read_pdf(pdf_file)


def convert_docx_to_text(file):
    """Convert a DOCX file to text."""
    doc = Document(file)
    return '\n'.join([para.text for para in doc.paragraphs])


def chunk_text_with_overlap(text, chunk_size=500, overlap=50):
    """Chunk text into overlapping segments."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


# Streamlit app setup
st.title("Document Query App with RAG")

# File uploader for PDF/DOCX files
uploaded_files = st.file_uploader("Upload PDF/DOCX files", type=["pdf", "docx"], accept_multiple_files=True)

# Text area for URL input
url_input = st.text_area("Enter PDF URLs (one per line)")

# Text input for the user's query
query = st.text_input("Enter your query")

if st.button("Run Query"):
    documents = []

    # Process uploaded files
    for file in uploaded_files:
        if file.type == "application/pdf":
            text = read_pdf(file)
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = convert_docx_to_text(file)
        else:
            continue
        chunks = chunk_text_with_overlap(text)
        documents.extend([{"content": chunk, "meta": {"source": file.name}} for chunk in chunks])

    # Process URLs
    if url_input:
        urls = url_input.splitlines()
        for url in urls:
            try:
                text = download_and_convert_pdf_to_text(url)
                chunks = chunk_text_with_overlap(text)
                documents.extend([{"content": chunk, "meta": {"source": url}} for chunk in chunks])
            except Exception as e:
                st.error(f"Failed to process URL {url}: {e}")

    # Update FAISS index with new documents
    if documents:
        document_store.write_documents(documents)
        document_store.update_embeddings(retriever)

    # Run the retrieval pipeline
    result = pipeline.run(query=query, params={"Retriever": {"top_k": 10}})
    top_chunks = [(doc.content, doc.meta.get('source', 'Unknown')) for doc in result['documents']]

    # Generate response using OpenAI
    if top_chunks:
        system_prompt = (
            "You are a helpful assistant. You are given a set of text chunks from documents. "
            "Please find the most relevant information based on the question below, "
            "using only the provided chunks. Don't use your own knowledge!"
        )
        user_prompt = query + "\n\n" + "\n\n".join(
            f"Chunk {i + 1}: {chunk[:200]}..." for i, chunk in enumerate(top_chunks)
        )
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        st.write("Refined Response:", response.choices[0].message.content.strip())
    else:
        st.write("No relevant chunks retrieved.")
