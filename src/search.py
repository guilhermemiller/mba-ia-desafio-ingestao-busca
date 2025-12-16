from langchain_community.vectorstores.pgvector import PGVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from typing import List, Tuple
from langchain_core.documents import Document
from config import config

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

class VectorDBRetriever:
    """Gerencia a recuperação de documentos do banco de dados vetorial."""
    def __init__(self, config):
        self.config = config
        self.embeddings_model = GoogleGenerativeAIEmbeddings(
            model=config.google_embedding_model,
            api_key=config.google_api_key
        )
        self.db = PGVector(
            connection_string=config.database_url,
            collection_name=config.collection_name,
            embedding_function=self.embeddings_model
        )

    def get_docs_with_score(self, query: str) -> List[Tuple[Document, float]]:
        """Busca documentos e seus scores de similaridade."""
        return self.db.similarity_search_with_score(query, k=10)

    def format_context(self, docs_with_scores: List[Tuple[Document, float]]) -> str:
        """Formata os documentos para o prompt do LLM."""
        return "\n\n".join(doc.page_content for doc, score in docs_with_scores)

class RAGChainBuilder:
    """Constrói a cadeia RAG."""
    def __init__(self, llm, retriever: VectorDBRetriever):
        self.llm = llm
        self.retriever = retriever
        self.prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    def build(self):
        """Constrói e retorna a cadeia RAG."""
        rag_chain_from_prompt = self.prompt | self.llm | StrOutputParser()

        rag_chain = (
            {
                "contexto": RunnableLambda(self.retriever.get_docs_with_score) | RunnableLambda(self.retriever.format_context),
                "pergunta": RunnablePassthrough()
            }
            | rag_chain_from_prompt
        )
        return rag_chain

def search_prompt(llm):
    """
    Configura e retorna uma cadeia de RAG (Retrieval-Augmented Generation).
    """
    retriever = VectorDBRetriever(config)
    chain_builder = RAGChainBuilder(llm, retriever)
    return chain_builder.build()