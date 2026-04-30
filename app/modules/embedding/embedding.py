"""
One-time setup — run this before starting the app for the first time.
Loads the job description PDF, chunks it, embeds with text-embedding-3-small,
and saves everything to chroma_db/ so the Info Advisor can do RAG.
"""
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

PDF_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "Python Developer Job Description.pdf"
)
CHROMA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "chroma_db"
)


def build_vectorstore() -> Chroma:
    """Load the PDF, split into chunks, embed, and persist to Chroma."""
    pdf_path = os.path.abspath(PDF_PATH)
    chroma_dir = os.path.abspath(CHROMA_DIR)

    print(f"Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"  → {len(pages)} page(s) loaded")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(pages)
    print(f"  → {len(chunks)} chunks created")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    print(f"Embedding and storing in Chroma: {chroma_dir}")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=chroma_dir,
    )
    print(f"Done — vectorstore ready with {len(chunks)} chunks.")
    return vectorstore


def load_vectorstore() -> Chroma:
    """Load an already-built Chroma vectorstore from disk."""
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    return Chroma(
        persist_directory=os.path.abspath(CHROMA_DIR),
        embedding_function=embeddings,
    )


if __name__ == "__main__":
    build_vectorstore()
