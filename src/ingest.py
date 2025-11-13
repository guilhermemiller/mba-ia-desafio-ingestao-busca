from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
import os
from dotenv import load_dotenv

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH")

# Connection string for your PostgreSQL database from docker-compose.yml
CONNECTION_STRING = os.getenv("DATABASE_URL")
COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME")

def ingest_pdf():
    if not PDF_PATH or not os.path.exists(PDF_PATH):
        print(f"Error: PDF_PATH is not valid. Please check your .env file. Current value: {PDF_PATH}")
        return

    print(f"Loading PDF from: {PDF_PATH}")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    print("Splitting documents into chunks... chunk_size=1000, chunk_overlap=150")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    print(f"Split into {len(docs)} chunks.")

    print(f"Initializing embeddings model...{os.getenv("GOOGLE_EMBEDDING_MODEL")}")
    # Correctly instantiate the embeddings model
    embeddings_model = GoogleGenerativeAIEmbeddings(model=os.getenv("GOOGLE_EMBEDDING_MODEL"),api_key=os.getenv("GOOGLE_API_KEY"))

    print("Ingesting documents and embeddings into the vector store...")
    # This will create embeddings and store them in PGVector.
    # It handles the embedding process internally.
    db = PGVector.from_documents(
        documents=docs,
        embedding=embeddings_model,
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
    )

    print(f"Successfully ingested {len(docs)} documents into the '{COLLECTION_NAME}' collection.")

if __name__ == "__main__":
    ingest_pdf()