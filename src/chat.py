from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import json
import os
from search import search_prompt

load_dotenv()

def get_llm():
    """Inicializa e retorna o modelo de linguagem do Google."""
    return ChatGoogleGenerativeAI(
        model=os.getenv("GOOGLE_CHAT_MODEL", "gemini-2.5-flash-lite"),
        api_key=os.getenv("GOOGLE_API_KEY"),
    )

def main():
    llm = get_llm()
    chain = search_prompt(llm)
    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    while True:
        question = input("\nSua pergunta: ")
        if question.lower() == 'sair':
            print("Até logo!")
            break

        # Invoca a cadeia e obtém um dicionário com a resposta e o contexto
        result = chain.invoke(question)
        response = result.get("resposta", "Não tenho informações necessárias para responder sua pergunta.")
        docs_com_score = result.get("docs_com_score", [])

        print("\n--- Documentos recuperados (Contexto) ---")
        for doc, score in docs_com_score:
            print(f"Score (distância): {score:.4f}")
            print(f"Metadata: {json.dumps(doc.metadata, indent=2)}")
            print("--------------------------------------")

        print("\nResposta:", response)

if __name__ == "__main__":
    main()