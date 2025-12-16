from langchain_google_genai import ChatGoogleGenerativeAI
from search import search_prompt
from config import config

class LLMProvider:
    """Fornece a instância do modelo de linguagem."""
    def __init__(self, config):
        self.config = config

    def get_llm(self):
        """Inicializa e retorna o modelo de linguagem do Google."""
        return ChatGoogleGenerativeAI(
            model=self.config.get_chat_model(),
            api_key=self.config.get_api_key(),
        )

class ChatInterface:
    """Gerencia a interface de chat na linha de comando."""
    def __init__(self, llm):
        self.chain = search_prompt(llm)

    def start(self):
        """Starts the chat session."""
        if not self.chain:
            print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
            return

        while True:
            question = input("\nSua pergunta: ")
            if question.lower() == 'sair':
                print("Até logo!")
                break

            response = self.chain.invoke(question)
            print("=" * 50)
            print("\nResposta:\n", response)
            print("=" * 50)

def main():
    """Função principal para executar a interface de chat."""
    llm_provider = LLMProvider(config)
    llm = llm_provider.get_llm()
    
    chat_interface = ChatInterface(llm)
    chat_interface.start()

if __name__ == "__main__":
    main()
