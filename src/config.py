import os
from dotenv import load_dotenv

class Config:
    """Carrega e fornece acesso à configuração da aplicação."""
    def __init__(self):
        load_dotenv()
        self.pdf_path = os.getenv("PDF_PATH")
        self.database_url = os.getenv("DATABASE_URL")
        self.collection_name = os.getenv("PG_VECTOR_COLLECTION_NAME")
        self.google_embedding_model = os.getenv("GOOGLE_EMBEDDING_MODEL")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.google_chat_model = os.getenv("GOOGLE_MODEL_RESPONSE")

        if self.database_url and "psycopg" not in self.database_url:
            self.database_url = self.database_url.replace("postgresql://", "postgresql+psycopg://")

    def get_db_connection_string(self):
        return self.database_url

    def get_collection_name(self):
        return self.collection_name

    def get_embedding_model(self):
        return self.google_embedding_model

    def get_api_key(self):
        return self.google_api_key
    
    def get_chat_model(self):
        return self.google_chat_model

# Instância singleton
config = Config()
