from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from typing import List

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

    def split(self, documents: List[Document]) -> List[Document]:
        print(f"Dividindo documentos em pedaços... tamanho_pedaço={self.splitter._chunk_size}, sobreposição_pedaço={self.splitter._chunk_overlap}")
        docs = self.splitter.split_documents(documents)
        print(f"Dividido em {len(docs)} pedaços.")
        return docs

class EmbeddingProvider:
    """Fornece o modelo de embeddings."""
    def __init__(self, model_name: str, api_key: str):
        print(f"Inicializando modelo de embeddings...{model_name}")
        self.embeddings_model = GoogleGenerativeAIEmbeddings(model=model_name, api_key=api_key)

    def get_embeddings(self):
        return self.embeddings_model

class VectorStoreManager:
    """Gerencia a ingestão de documentos no armazenamento vetorial."""
    def __init__(self, connection_string: str, collection_name: str, embeddings):
        self.connection_string = connection_string
        self.collection_name = collection_name
        self.embeddings = embeddings

    def ingest(self, docs: List[Document]):
        print("Ingerindo documentos e embeddings no banco de vetores...")
        PGVector.from_documents(
            documents=docs,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            connection_string=self.connection_string,
            use_jsonb=True
        )
        print(f"Ingeridos {len(docs)} documentos com sucesso na coleção '{self.collection_name}'.")

class IngestionPipeline:
    """Orquestra o processo de ingestão de documentos."""
    def __init__(self, config):
        self.config = config
        self.loader = DocumentLoader(config['pdf_path'])
        self.splitter = TextChunker()
        self.embedding_provider = EmbeddingProvider(config['embedding_model'], config['api_key'])
        self.vector_store = VectorStoreManager(
            config['db_url'], 
            config['collection_name'], 
            self.embedding_provider.get_embeddings()
        )

    def run(self):
        documents = self.loader.load()
        docs = self.splitter.split(documents)
        self.vector_store.ingest(docs)

class ChatInterface:
    """Gerencia a interface de chat na linha de comando."""
    def __init__(self, llm):
        self.chain = search_prompt(llm)

    def start(self):
        """Inicia a sessão de chat."""
        if not self.chain:
            print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
            return

        while True:
            question = input("\nSua pergunta: ")
            if question.lower() == 'sair':
                print("Até logo!")
                break

            response = self.chain.invoke(question)
            print("\nResposta:\n", response)

def main():
    """Função principal para executar o pipeline de ingestão."""
    load_dotenv()
    config = {
        'pdf_path': os.getenv("PDF_PATH"),
        'db_url': os.getenv("DATABASE_URL"),
        'collection_name': os.getenv("PG_VECTOR_COLLECTION_NAME"),
        'embedding_model': os.getenv("GOOGLE_EMBEDDING_MODEL"),
        'api_key': os.getenv("GOOGLE_API_KEY")
    }
    
    try:
        pipeline = IngestionPipeline(config)
        pipeline.run()
    except ValueError as e:
        print(e)
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")

if __name__ == "__main__":
    main()
