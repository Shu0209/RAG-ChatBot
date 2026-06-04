from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, CSVLoader, TextLoader, UnstructuredExcelLoader
)
import tempfile
import os
from io import BytesIO

SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".csv", ".txt", ".xlsx"]

def load_document_from_bytes(file_bytes,filename):
    ext=os.path.splitext(filename)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    try:
        if ext == ".pdf":
            loader = PyPDFLoader(tmp_path)
        elif ext == ".docx":
            loader = Docx2txtLoader(tmp_path)
        elif ext == ".csv":
            loader = CSVLoader(tmp_path)
        elif ext == ".txt":
            loader = TextLoader(tmp_path)
        elif ext == ".xlsx":
            loader = UnstructuredExcelLoader(tmp_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        
        documents = loader.load()

    finally:
        os.unlink(tmp_path)
    
    return documents


def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    return text_splitter.split_documents(extracted_data)


def download_hugging_face_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )