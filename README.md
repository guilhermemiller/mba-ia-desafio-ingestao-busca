# Sistema RAG de Ingestão e Busca Semântica com IA

Sistema completo de **Retrieval-Augmented Generation (RAG)** que combina ingestão de documentos PDF, armazenamento vetorial e interface de chat inteligente usando Google Generative AI.

## 📋 Visão Geral do Sistema

O sistema é composto por três componentes principais:

1. **Pipeline de Ingestão** (`ingest.py`): Carrega PDFs, divide em chunks e gera embeddings
2. **Armazenamento Vetorial**: PostgreSQL com pgvector para busca semântica
3. **Interface de Chat** (`chat.py`): Chat interativo com RAG para responder perguntas baseadas nos documentos

## 🏗️ Arquitetura

```
┌─────────────────────┐
│   PDF Documents     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Document Loader    │ (DocumentLoader)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Text Chunking     │ (TextChunker)
│ (1000 char chunks)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Embeddings Gen     │ (GoogleGenerativeAI)
│  (SyncWrapper)      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ PostgreSQL pgvector │
│   (Vector Store)    │
└─────────────────────┘
           ▲
           │
    ┌──────┴──────┐
    │             │
┌───▼─────┐  ┌────▼────┐
│ Retriever│  │   LLM   │
└───┬─────┘  └────┬────┘
    │             │
    └──────┬──────┘
           ▼
    ┌─────────────────┐
    │  Chat Interface │
    │   (RAG Chain)   │
    └─────────────────┘
```

## 🚀 Instalação e Setup

### Pré-requisitos

- Python 3.10+
- Docker e Docker Compose
- Chave de API do Google Generative AI

### 1. Clonar o Repositório

```bash
git clone <repository-url>
cd mba-ia-desafio-ingestao-busca
```

### 2. Criar Ambiente Virtual

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

### 3. Instalar Dependências

```bash
pip install -r requirements.txt
```

### 4. Configurar Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto:

```env
# Google Generative AI
GOOGLE_API_KEY=sua_chave_api_google
GOOGLE_EMBEDDING_MODEL=models/embedding-001
GOOGLE_MODEL_RESPONSE=gemini-pro

# PostgreSQL com pgvector
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag

# Vector Store
PG_VECTOR_COLLECTION_NAME=documents

# Caminho do PDF a ser ingerido
PDF_PATH=/caminho/para/seu/documento.pdf
```

### 5. Iniciar o PostgreSQL com pgvector

```bash
docker-compose up -d
```

Isso inicia:
- **PostgreSQL 17** com extensão pgvector
- **Serviço de inicialização** que cria a extensão vector automaticamente

Verificar se está rodando:
```bash
docker-compose ps
```

## 📚 Fluxo de Uso

### Passo 1: Ingerir Documentos

```bash
python src/ingest.py
```

**O que acontece:**
1. Carrega o PDF especificado em `PDF_PATH`
2. Divide o documento em chunks de 1000 caracteres com sobreposição de 150 caracteres
3. Gera embeddings usando Google Generative AI
4. Armazena embeddings no PostgreSQL (tabela: `documents`)
5. Imprime progresso em cada etapa

**Saída esperada:**
```
Carregando PDF de: /caminho/para/documento.pdf
Inicializando modelo de embeddings...models/embedding-001
Dividindo documentos em pedaços... chunk_size=1000, chunk_overlap=150
Dividido em 45 pedaços.
Ingerindo documentos e embeddings no banco de vetores...
Ingeridos 45 documentos com sucesso na tabela 'documents'.
```

### Passo 2: Iniciar o Chat RAG

```bash
python src/chat.py
```

**O que acontece:**
1. Inicializa a interface de chat
2. Carrega o modelo de linguagem (Gemini Pro)
3. Conecta ao banco de vetores
4. Aguarda perguntas do usuário

**Interação:**
```
Sua pergunta: Qual é o tema principal do documento?
==================================================

Resposta:
[Resposta baseada no contexto dos documentos ingeridos]

==================================================
Sua pergunta: sair
Até logo!
```

## 📁 Estrutura de Arquivos

```
mba-ia-desafio-ingestao-busca/
├── src/
│   ├── chat.py              # Interface de chat com RAG
│   ├── search.py            # Configuração da cadeia RAG
│   ├── ingest.py            # Pipeline de ingestão
│   ├── config.py            # Gerenciamento de configuração
│   └── __pycache__/
├── docker-compose.yml       # Configuração PostgreSQL
├── requirements.txt         # Dependências Python
├── .env.example            # Template de variáveis de ambiente
├── .env                    # Variáveis de ambiente (não versionado)
└── README.md               # Este arquivo
```

## 🔧 Componentes Principais

### `config.py`
Gerencia todas as configurações da aplicação:
- Carregamento de variáveis de ambiente
- Inicialização do engine PostgreSQL
- Validação de configurações

### `ingest.py`
Pipeline de ingestão com classes:
- **DocumentLoader**: Carrega PDFs usando PyPDFLoader
- **TextChunker**: Divide documentos em chunks
- **SyncEmbeddingsWrapper**: Wrapper síncrono para embeddings (evita conflitos de event loop)
- **EmbeddingProvider**: Fornece modelo de embeddings
- **VectorStoreManager**: Gerencia armazenamento vetorial
- **IngestionPipeline**: Orquestra todo o processo

**Características:**
- Tratamento robusto de erros
- Detecção de tabelas existentes
- Processamento síncrono sem conflitos de async

### `search.py`
Configuração da cadeia RAG:
- **VectorDBRetriever**: Busca documentos similares no banco vetorial
- **RAGChainBuilder**: Constrói a cadeia RAG
- **search_prompt()**: Função que retorna a cadeia RAG pronta

**Prompt Template:**
```
Responda apenas baseado no CONTEXTO fornecido.
Se a informação não está no CONTEXTO, responda:
"Não tenho informações necessárias para responder sua pergunta."
Nunca invente ou use conhecimento externo.
```

### `chat.py`
Interface interativa:
- **LLMProvider**: Inicializa o modelo Gemini Pro
- **ChatInterface**: Gerencia sessão de chat
- Loop interativo com tratamento de erros

## 🗄️ Banco de Dados

### PostgreSQL + pgvector
- **Container**: `postgres_rag`
- **Usuário**: postgres
- **Senha**: postgres
- **Database**: rag
- **Porta**: 5432
- **Extensão**: pgvector (para busca semântica)

### Tabela de Documentos
```sql
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(768),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops);
```

## 🤖 Modelos IA Utilizados

### Google Generative AI
- **Embeddings**: `models/embedding-001`
  - Dimensionalidade: 768
  - Usado para: Vetorização de documentos e queries
  
- **Chat**: `gemini-pro`
  - Usado para: Geração de respostas conversacionais

## ⚙️ Configurações Ajustáveis

No arquivo `ingest.py`, você pode ajustar:

```python
# TextChunker
chunk_size=1000        # Tamanho dos chunks em caracteres
chunk_overlap=150      # Sobreposição entre chunks

# Embedding
vector_size=768        # Dimensionalidade dos embeddings
```

Na busca vetorial (`search.py`):
```python
k=10                   # Número de documentos retornados na busca
```

## 🐛 Tratamento de Erros

O sistema inclui:
- Validação de paths de arquivos
- Detecção de tabelas duplicadas no banco
- Tratamento de erros na interface de chat
- Wrapper síncrono para embeddings (evita event loop conflicts)

## 📋 Troubleshooting

### Erro: "Event loop attached to different loop"
**Solução**: O sistema usa `SyncEmbeddingsWrapper` para evitar isso. Se ocorrer, verifique se há múltiplas chamadas async simultâneas.

### Erro: "Table already exists"
**Solução**: Normal! O sistema detecta tabelas existentes e continua adiante.

### Erro: "PDF_PATH não é válido"
**Solução**: Verifique se a variável `PDF_PATH` no `.env` aponta para um arquivo PDF existente.

### Erro: "Não consegue conectar ao PostgreSQL"
**Solução**: Certifique-se de que:
1. Docker está rodando
2. `docker-compose up -d` foi executado
3. O container está saudável: `docker-compose ps`

## 📊 Fluxo de Dados

1. **PDF → Loader** → Extrai texto e metadados
2. **Texto → Chunker** → Divide em pedaços menores
3. **Chunks → Embeddings** → Converte para vetores (768 dims)
4. **Vetores → PostgreSQL** → Armazena com índice IVFFLAT
5. **Query → Search** → Busca K documentos similares
6. **Documentos + Query → LLM** → Gera resposta contextualizada

## 🔐 Segurança

- Variáveis sensíveis em `.env` (não versionado)
- Validação de entrada na interface de chat
- Uso de prepared statements via ORM

## 🚦 Status e Logs

Para ver logs do PostgreSQL:
```bash
docker-compose logs postgres
```

Para monitorar a ingestão:
```bash
python src/ingest.py 2>&1 | tee ingest.log
```

## 📦 Dependências Principais

- **langchain**: Framework RAG
- **langchain-google-genai**: Integração Google AI
- **langchain-postgres**: Store vetorial PostgreSQL
- **pgvector**: Extensão PostgreSQL para vetores
- **psycopg**: Driver PostgreSQL
- **python-dotenv**: Gerenciamento de env vars

## 📝 Autor e Versão

- **Desafio**: MBA Engenharia de Software com IA
- **Versão**: 1.0
- **Data**: 2025

## 📞 Suporte

Para problemas, verifique:
1. Variáveis de ambiente (`.env`)
2. Conexão PostgreSQL (`docker-compose ps`)
3. Chave API Google
4. Path do PDF