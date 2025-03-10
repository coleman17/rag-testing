import os
import bs4
from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader, TextLoader, DirectoryLoader, PyPDFLoader, CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain.document_loaders.unstructured import UnstructuredFileLoader

import dotenv

dotenv.load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Function to load web content
def load_web_content(web_paths):
    loader = WebBaseLoader(web_paths=web_paths)
    loader.requests_kwargs = {"verify": False}
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

# Load web content
web_docs = load_web_content([
    "https://devlog.tublian.com/tublian-open-source-internship-cohort2-a-path-to-software-development-mastery",
])

# Load local files from a specified directory
local_docs_dir = "./data"  # Change this to your local data directory
local_docs = load_local_files(local_docs_dir)

# Combine web and local documents
all_docs = web_docs + local_docs
print(f"Loaded {len(web_docs)} web documents and {len(local_docs)} local documents")

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(all_docs)
print(f"Created {len(splits)} document chunks")

# Create vector store and retriever
vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=OpenAIEmbeddings(), 
    persist_directory="./chroma_db"
)
retriever = vectorstore.as_retriever()

# Set up the RAG pipeline
prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Example usage
def ask_question(question):
    print(f"Question: {question}")
    print("Retrieving information...")
    result = rag_chain.invoke(question)
    print(f"Answer: {result}")
    return result

if __name__ == "__main__":
    # Create the data directory if it doesn't exist
    if not os.path.exists(local_docs_dir):
        os.makedirs(local_docs_dir)
        print(f"Created directory: {local_docs_dir}")
        print(f"Please add your local documents to this directory and run the script again.")
    
    # Example questions
    ask_question("How long is the Open Source internship?")
    
    # You can add more questions here
    # ask_question("Your question here")