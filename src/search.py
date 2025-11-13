from langchain_community.vectorstores.pgvector import PGVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
import os
from langchain_core.documents import Document

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

def search_prompt(llm):
    """
    Configura e retorna uma cadeia de RAG (Retrieval-Augmented Generation).
    """
    CONNECTION_STRING = os.getenv("DATABASE_URL")
    COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME")

    embeddings_model = GoogleGenerativeAIEmbeddings(
        model=os.getenv("GOOGLE_EMBEDDING_MODEL"),
        api_key=os.getenv("GOOGLE_API_KEY")
    )

    # Conecta ao banco de dados vetorial e cria um retriever
    db = PGVector(
        connection_string=CONNECTION_STRING,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings_model
    )

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    def _get_docs_with_score(query: str) -> list[tuple[Document, float]]:
        """Busca documentos e seus scores de similaridade."""
        return db.similarity_search_with_score(query, k=10)

    def _format_context(docs_with_scores: list[tuple[Document, float]]) -> str:
        """Formata os documentos para o prompt do LLM."""
        return "\n\n".join(doc.page_content for doc, score in docs_with_scores)

    # Define a sub-cadeia que formata o prompt, chama o LLM e parseia a saída.
    # Ela espera um dicionário com 'contexto' e 'pergunta'.
    rag_chain_from_prompt = prompt | llm | StrOutputParser()

    # Cria a cadeia RAG
    rag_chain = (
        {
            "docs_com_score": RunnableLambda(_get_docs_with_score),
            "pergunta": RunnablePassthrough()
        } | RunnableParallel(
            docs_com_score=lambda x: x["docs_com_score"],
            resposta={
                "contexto": lambda x: _format_context(x["docs_com_score"]),
                "pergunta": lambda x: x["pergunta"]
            } | rag_chain_from_prompt
        ) 
    )
    return rag_chain