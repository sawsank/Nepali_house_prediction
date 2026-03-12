import pandas as pd
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.config import CHROMA_PATH, DATA_PATH, EMBEDDING_MODEL_NAME
import chromadb
import os

def get_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    # Ensure the directory exists
    if not os.path.exists(CHROMA_PATH):
        try:
            os.makedirs(CHROMA_PATH, exist_ok=True)
            print(f"--- Created Directory: {CHROMA_PATH} ---")
        except Exception as e:
            print(f"Error creating directory: {e}")
            return None

    # Use persistent client for more control
    try:
        if os.path.exists(os.path.join(CHROMA_PATH, "chroma.sqlite3")):
            print("--- Loading existing Vector DB ---")
            return Chroma(
                persist_directory=CHROMA_PATH, 
                embedding_function=embeddings
            )
        
        print("--- Creating new Vector DB ---")
        if not os.path.exists(DATA_PATH):
            print(f"Error: {DATA_PATH} not found.")
            return None

        df = pd.read_csv(DATA_PATH)
        text_data = []
        for _, row in df.iterrows():
            content = " | ".join([f"{col}: {val}" for col, val in row.items()])
            text_data.append(Document(page_content=content, metadata=row.to_dict()))

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(text_data)

        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_PATH
        )
        return vector_db
    except Exception as e:
        print(f"ChromaDB Initialization Error: {e}")
        return None

def add_document_to_db(content, category):
    """Manually add a piece of knowledge to the vector store."""
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        vector_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
        
        doc = Document(
            page_content=f"Category: {category} | Content: {content}",
            metadata={"category": category, "source": "manual_entry"}
        )
        
        vector_db.add_documents([doc])
        return True, "Knowledge added and indexed successfully!"
    except Exception as e:
        return False, f"Failed to add knowledge: {str(e)}"

if __name__ == "__main__":
    db = get_vector_db()
    if db:
        print("Vector DB initialized successfully.")
    else:
        print("Failed to initialize Vector DB.")
