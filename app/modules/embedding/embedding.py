"""
One-time setup — run this before starting the app for the first time.
Loads the job description PDF, chunks it, embeds with text-embedding-3-small,
and saves everything to chroma_db/ so the Info Advisor can do RAG.

To add a new role later:
    build_vectorstore(
        pdf_path="path/to/ML Engineer Job Description.pdf",
        role="ML",
        force_rebuild=False,   # False = only adds if not already present
    )
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

# Single collection holds all roles — chunks are tagged with role metadata.
# Must stay in sync with COLLECTION_NAME in info_advisor.py.
COLLECTION_NAME = "job_descriptions"


def build_vectorstore(
    pdf_path: str = PDF_PATH,
    role: str = "Python Dev",
    collection_name: str = COLLECTION_NAME,
    force_rebuild: bool = False,
) -> Chroma:
    """
    Load a PDF, split into chunks, embed, and persist to Chroma.

    Args:
        pdf_path:        Path to the job description PDF.
        role:            Role label stored as chunk metadata (e.g. "Python Dev",
                         "ML", "Sql Dev", "Analyst"). Must match the position
                         strings used by MainAgent._detect_position().
        collection_name: Chroma collection to write into (shared across roles).
        force_rebuild:   If False (default), skip if this role is already in the
                         collection. Pass True to delete existing role chunks and
                         re-embed from scratch.
    """
    pdf_path = os.path.abspath(pdf_path)
    chroma_dir = os.path.abspath(CHROMA_DIR)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Check whether this role already exists in the collection.
    existing = Chroma(
        persist_directory=chroma_dir,
        embedding_function=embeddings,
        collection_name=collection_name,
    )
    existing_ids = existing.get(where={"role": role})["ids"]
    if existing_ids and not force_rebuild:
        print(
            f"Role '{role}' already has {len(existing_ids)} chunks in collection "
            f"'{collection_name}'. Skipping. Pass force_rebuild=True to overwrite."
        )
        return existing

    if existing_ids and force_rebuild:
        print(f"force_rebuild=True — deleting {len(existing_ids)} existing '{role}' chunks.")
        existing.delete(ids=existing_ids)

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

    # Tag every chunk with its role so the Info Advisor can filter by position.
    for chunk in chunks:
        chunk.metadata["role"] = role

    print(f"Embedding and storing in Chroma collection '{collection_name}': {chroma_dir}")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=chroma_dir,
        collection_name=collection_name,
    )
    print(f"Done — {len(chunks)} chunks stored for role '{role}'.")
    return vectorstore


def load_vectorstore(collection_name: str = COLLECTION_NAME) -> Chroma:
    """Load an already-built Chroma vectorstore from disk."""
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    return Chroma(
        persist_directory=os.path.abspath(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name=collection_name,
    )


if __name__ == "__main__":
    build_vectorstore()
