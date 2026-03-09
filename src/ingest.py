from langchain_community.document_loaders import PyPDFLoader
import psycopg
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVectorStore, PGEngine
import os
from langchain.schema import Document
from typing import List
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

# Custom sync-only wrapper for embeddings
class SyncEmbeddingsWrapper:
    """Wraps GoogleGenerativeAIEmbeddings to work synchronously without event loop conflicts."""
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key
        print(f"Inicializando modelo de embeddings...{model_name}")
        # Pre-create embeddings in main thread to cache it
        self._embeddings = GoogleGenerativeAIEmbeddings(model=model_name, api_key=api_key)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents synchronously."""
        return self._embeddings.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a query synchronously."""
        return self._embeddings.embed_query(text)
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async version - wraps sync call in executor."""
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            return await loop.run_in_executor(
                executor,
                self.embed_documents,
                texts
            )
        finally:
            executor.shutdown(wait=False)
    
    async def aembed_query(self, text: str) -> List[float]:
        """Async version of embed_query."""
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            return await loop.run_in_executor(
                executor,
                self.embed_query,
                text
            )
        finally:
            executor.shutdown(wait=False)

class DocumentLoader:
    """Gerencia o carregamento de documentos de um caminho especificado."""
    def __init__(self, path: str):
        if not path or not os.path.exists(path):
            raise ValueError(f"Erro: PDF_PATH não é válido. Caminho: {path}")
        self.path = path

    def load(self) -> List[Document]:
        print(f"Carregando PDF de: {self.path}")
        loader = PyPDFLoader(self.path)
        return loader.load()

class TextChunker:
    """Divide documentos em pedaços menores."""
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, documents: List[Document]) -> List[Document]:
        print(f"Dividindo documentos em pedaços... chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}")
        docs = self.splitter.split_documents(documents)
        print(f"Dividido em {len(docs)} pedaços.")
        return docs

class EmbeddingProvider:
    """Fornece o modelo de embeddings."""
    def __init__(self, model_name: str, api_key: str):
        self.wrapper = SyncEmbeddingsWrapper(model_name, api_key)

    def get_embeddings_model(self):
        """Return the sync wrapper."""
        return self.wrapper

class VectorStoreManager:
    """Gerencia a ingestão de documentos no armazenamento vetorial."""
    def __init__(self, connection_string: str, collection_name: str, embedding_provider, vector_size: int):
        self.connection_string = connection_string
        self.table_name = collection_name
        self.embedding_provider = embedding_provider
        self.vector_size = vector_size
        
        # Initialize table synchronously
        engine = PGEngine.from_connection_string(connection_string)
        try:
            engine.init_vectorstore_table(table_name=self.table_name, vector_size=vector_size)
        except Exception as e:
            # Ignore if the table already exists (concurrent or previous run)
            if isinstance(e, psycopg.errors.DuplicateTable) or "already exists" in str(e):
                print(f"Tabela '{self.table_name}' já existe, seguindo adiante.")
            else:
                raise

    def ingest(self, docs: List[Document]):
        """Synchronous ingestion using embedding wrapper."""
        print("Ingerindo documentos e embeddings no banco de vetores...")
        
        engine = PGEngine.from_connection_string(self.connection_string)
        
        # Use the sync wrapper directly
        embeddings = self.embedding_provider.get_embeddings_model()
        
        store = PGVectorStore.from_documents(
            documents=docs,
            embedding=embeddings,
            table_name=self.table_name,
            engine=engine
        )
        
        print(f"Ingeridos {len(docs)} documentos com sucesso na tabela '{self.table_name}'.")

class IngestionPipeline:
    """Orquestra o processo de ingestão de documentos."""
    def __init__(self, config):
        self.config = config
        self.loader = DocumentLoader(config.pdf_path)
        self.splitter = TextChunker()

    def run(self):
        documents = self.loader.load()
        docs = self.splitter.split(documents)
        
        # Create embeddings provider
        embedding_provider = EmbeddingProvider(self.config.google_embedding_model, self.config.google_api_key)
        
        vector_store = VectorStoreManager(
            self.config.database_url,
            self.config.collection_name,
            embedding_provider,
            self.config.vector_size
        )
        vector_store.ingest(docs)

def main():
    """Função principal para executar o pipeline de ingestão."""
    try:
        from config import config
        pipeline = IngestionPipeline(config)
        pipeline.run()
    except ValueError as e:
        print(e)
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")

if __name__ == "__main__":
    main()
