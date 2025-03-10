import os
import bs4
import numpy as np
from langchain import hub
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader, TextLoader, DirectoryLoader, PyPDFLoader, CSVLoader
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama

import dotenv

dotenv.load_dotenv()

# Set default OpenAI API key from environment
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# Optional user agent setting
os.environ["USER_AGENT"] = "RAG-Testing/1.0"

# Configuration options - adjust these based on your setup
CONFIG = {
    "use_local_model": True,  # Set to True to use local Llama model, False for OpenAI
    "local_model_type": "ollama",  # "llamacpp" or "ollama"
    "local_model_path": "deepseek-r1:8b",  # For llamacpp: path to model, for ollama: model name
    "openai_model": "gpt-4o",  # OpenAI model to use
    "use_local_embeddings": True,  # Set to True to use local embeddings, False for OpenAI
    "local_embeddings_model": "all-MiniLM-L6-v2",  # HuggingFace model for embeddings
    "chunk_size": 1000,
    "chunk_overlap": 200,
}

# Function to load web content
def load_web_content(web_paths):
    loader = WebBaseLoader(web_paths=web_paths)
    # Enable SSL verification for security
    return loader.load()

# Function to load different file types
def load_local_files(directory_path):
    documents = []
    
    # Walk through all files in the directory
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                if file.endswith('.txt'):
                    loader = TextLoader(file_path)
                    documents.extend(loader.load())
                elif file.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                    documents.extend(loader.load())
                elif file.endswith('.csv'):
                    loader = CSVLoader(file_path)
                    documents.extend(loader.load())
                elif file.endswith(('.docx', '.doc', '.pptx', '.xlsx')):
                    # Requires 'unstructured' package for these formats
                    loader = UnstructuredFileLoader(file_path)
                    documents.extend(loader.load())
                # Add more file types as needed
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    return documents

# Function to create LLM based on configuration
def get_llm():
    if CONFIG["use_local_model"]:
        if CONFIG["local_model_type"] == "llamacpp":
            # Use LlamaCpp for local models loaded directly from files
            return LlamaCpp(
                model_path=CONFIG["local_model_path"],
                temperature=0.1,
                max_tokens=2000,
                n_ctx=4096,  # Context window size
                verbose=True,  # Optional for debugging
            )
        elif CONFIG["local_model_type"] == "ollama":
            # Use Ollama for local models running as a service
            return ChatOllama(
                model=CONFIG["local_model_path"],
                temperature=0.1,
            )
        else:
            raise ValueError(f"Unsupported local model type: {CONFIG['local_model_type']}")
    else:
        # Use OpenAI
        return ChatOpenAI(
            model_name=CONFIG["openai_model"],
            temperature=0,
        )

# Function to get embeddings based on configuration
def get_embeddings():
    if CONFIG["use_local_embeddings"]:
        # Use HuggingFace embeddings
        return HuggingFaceEmbeddings(model_name=CONFIG["local_embeddings_model"])
    else:
        # Use OpenAI embeddings
        return OpenAIEmbeddings()

# Main function to set up and run the RAG pipeline
def setup_rag_pipeline(web_paths=None, local_dir=None):
    documents = []
    
    # Load web content if provided
    if web_paths:
        web_docs = load_web_content(web_paths)
        print(f"Loaded {len(web_docs)} web documents")
        documents.extend(web_docs)
    
    # Load local files if directory is provided
    if local_dir and os.path.exists(local_dir):
        local_docs = load_local_files(local_dir)
        print(f"Loaded {len(local_docs)} local documents")
        documents.extend(local_docs)
    
    if not documents:
        print("No documents loaded. Please provide valid web paths or local directory.")
        return None
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CONFIG["chunk_size"], 
        chunk_overlap=CONFIG["chunk_overlap"]
    )
    splits = text_splitter.split_documents(documents)
    print(f"Created {len(splits)} document chunks")
    
    # Create vector store and retriever
    embedding_model = get_embeddings()
    
    # Check if we need to use a fresh DB to avoid dimension issues
    persist_directory = "./chroma_db"
    if os.path.exists(persist_directory):
        print("Found existing vector database. Using a fresh DB to avoid dimension conflicts.")
        # Use a different directory based on the embedding model
        if CONFIG["use_local_embeddings"]:
            persist_directory = f"./chroma_db_local_{CONFIG['local_embeddings_model'].replace('-', '_')}"
        else:
            persist_directory = "./chroma_db_openai"
        print(f"Using database directory: {persist_directory}")
    
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embedding_model, 
        persist_directory=persist_directory
    )
    retriever = vectorstore.as_retriever()
    
    # Set up the RAG pipeline
    prompt = hub.pull("rlm/rag-prompt")
    llm = get_llm()
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# Function to ask questions
def ask_question(rag_chain, question):
    if rag_chain is None:
        print("RAG pipeline not set up properly.")
        return None
    
    print(f"Question: {question}")
    print("Retrieving information...")
    result = rag_chain.invoke(question)
    print(f"Answer: {result}")
    return result

if __name__ == "__main__":
    # Example usage
    web_paths = [
        "",
    ]
    local_docs_dir = "./data"  # Change this to your local data directory
    
    # Create the data directory if it doesn't exist
    if not os.path.exists(local_docs_dir):
        os.makedirs(local_docs_dir)
        print(f"Created directory: {local_docs_dir}")
        print(f"Please add your local documents to this directory and run the script again.")
    
    # Set up the RAG pipeline
    rag_chain = setup_rag_pipeline(web_paths, local_docs_dir)
    
    if rag_chain:
        # Example questions
        ask_question(rag_chain, input("What can I help you with today?"))
        
        # Interactive mode
        while True:
            user_question = input("\nWhat else can I help you with today? (or 'q' to quit): ")
            if user_question.lower() in ['q', 'quit', 'exit']:
                break
            ask_question(rag_chain, user_question)