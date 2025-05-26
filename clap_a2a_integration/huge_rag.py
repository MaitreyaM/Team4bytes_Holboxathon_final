import asyncio
import os
import shutil
import time
from dotenv import load_dotenv
from typing import Optional

# CLAP Imports
from clap import Agent
from clap.vector_stores.chroma_store import ChromaStore # Assuming Chroma for this example
from clap.utils.rag_utils import (
    load_pdf_file,
    chunk_text_by_fixed_size
)
# Choose your LLM Service for the CLAP RAG Agent
# from clap.llm_services.groq_service import GroqService
from clap.llm_services.google_openai_compat_service import GoogleOpenAICompatService
# from clap.llm_services.ollama_service import OllamaService

# Embedding function for ChromaDB
# Ensure sentence-transformers is installed: pip install sentence-transformers
try:
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    DEFAULT_EF = SentenceTransformerEmbeddingFunction() # Uses a default model like all-MiniLM-L6-v2
except ImportError:
    print("ERROR: sentence-transformers library not found. ChromaDB default EF might be used, or RAG will fail.")
    print("Please install with: pip install sentence-transformers")
    # Fallback or exit if SentenceTransformer is critical
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    DEFAULT_EF = DefaultEmbeddingFunction()


load_dotenv() # Load .env file from the current directory or parent

# --- Configuration ---
# IMPORTANT: Place your PDF in the same directory as this script,
# or provide the full absolute path.
PDF_FILENAME = "handsonml.pdf" # Name of your PDF file
PDF_PATH = os.path.join(os.path.dirname(__file__), PDF_FILENAME) # Path relative to this script

CHROMA_DB_PATH = "./interactive_rag_chroma_db" # Path to store the ChromaDB
COLLECTION_NAME = "ml_book_interactive_rag"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Configure the LLM service for your CLAP RAG Agent
# Ensure API keys are in your .env file (e.g., GOOGLE_API_KEY or GROQ_API_KEY)
LLM_SERVICE_CLASS = GoogleOpenAICompatService # Example: Google
LLM_MODEL = "gemini-1.5-flash-latest"         # Example: Gemini Flash

# If using Groq:
# LLM_SERVICE_CLASS = GroqService
# LLM_MODEL = "llama3-70b-8192"

# If using Ollama (ensure Ollama server is running):
# LLM_SERVICE_CLASS = OllamaService
# OLLAMA_LLM_MODEL_FOR_RAG = "llama3" # Or your preferred Ollama model
# LLM_MODEL = OLLAMA_LLM_MODEL_FOR_RAG
# # If OllamaService takes specific args like base_url:
# # llm_service = LLM_SERVICE_CLASS(default_model=LLM_MODEL, base_url="http://localhost:11434/v1")
# # else:
# # llm_service = LLM_SERVICE_CLASS()


async def initialize_rag_system() -> Optional[Agent]:
    """
    Initializes the vector store (ingesting PDF if needed) and the RAG agent.
    Returns the RAG agent instance or None if initialization fails.
    """
    print(f"--- RAG System Initialization ---")
    print(f"Vector DB Path: {CHROMA_DB_PATH}")
    print(f"Collection Name: {COLLECTION_NAME}")

    # Check if DB already exists and has data to potentially skip ingestion
    db_exists_and_has_data = False
    if os.path.exists(CHROMA_DB_PATH):
        try:
            # Try to connect and get collection count
            temp_vector_store = ChromaStore(
                path=CHROMA_DB_PATH,
                collection_name=COLLECTION_NAME,
                embedding_function=DEFAULT_EF # EF needed to connect even for count
            )
            # Chroma's count is on the collection object itself
            # We need to ensure the collection is actually loaded/created by this.
            # The get_or_create_collection happens in ChromaStore.__init__
            # A simple way to check if it has data is to try a dummy query or get count
            # This part might need adjustment based on ChromaStore's API for "count"
            # For now, we'll re-ingest if unsure or simplify to always ingest if DB dir is present but empty.
            # A more robust check would involve querying the collection count.
            # Let's assume for simplicity, if the path exists, we try to use it without re-ingesting.
            # A truly robust way is to check collection.count(). For this script, we can be simpler.
            print(f"Found existing DB path at {CHROMA_DB_PATH}. Attempting to use existing collection.")
            # A more direct way to check count if ChromaStore doesn't expose it:
            # client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
            # try:
            #   collection = client.get_collection(name=COLLECTION_NAME, embedding_function=DEFAULT_EF)
            #   if collection.count() > 0:
            #     db_exists_and_has_data = True
            #     print(f"Collection '{COLLECTION_NAME}' exists with {collection.count()} items. Skipping ingestion.")
            # except: # Collection does not exist or other error
            #   pass
            # For simplicity, we'll just use the existence of the directory for now.
            # More advanced: check collection.count()
            # To keep it simple and ensure data for demo: always re-ingest if PDF changed or force.
            # For this interactive demo, let's make it easy to re-ingest if desired or if db is new.
            if not os.listdir(CHROMA_DB_PATH): # If directory is empty
                 print("DB path exists but is empty. Will ingest.")
            else:
                 print("DB path exists and is not empty. Assuming data is present. Skipping ingestion.")
                 db_exists_and_has_data = True # Assume data if directory is not empty
        except Exception as e:
            print(f"Could not verify existing ChromaDB, will attempt to re-ingest: {e}")
            db_exists_and_has_data = False # Force re-ingestion on error

    if not db_exists_and_has_data:
        if os.path.exists(CHROMA_DB_PATH):
            print(f"Cleaning up existing DB at {CHROMA_DB_PATH} for re-ingestion...")
            shutil.rmtree(CHROMA_DB_PATH)
        
        print(f"Loading PDF from: {PDF_PATH}")
        if not os.path.exists(PDF_PATH):
            print(f"ERROR: PDF file not found at '{PDF_PATH}'. Please place it there or update path.")
            return None
        
        pdf_content = load_pdf_file(PDF_PATH)
        if not pdf_content:
            print("Failed to load content from PDF.")
            return None

        print(f"Chunking PDF content (Size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP})...")
        chunks = chunk_text_by_fixed_size(pdf_content, CHUNK_SIZE, CHUNK_OVERLAP)
        print(f"Generated {len(chunks)} chunks.")

        if not chunks:
            print("No chunks generated from PDF. Cannot proceed with RAG.")
            return None

        # Initialize ChromaStore for ingestion
        vector_store = ChromaStore(
            path=CHROMA_DB_PATH,
            collection_name=COLLECTION_NAME,
            embedding_function=DEFAULT_EF # SentenceTransformer EF
        )
        print("ChromaStore initialized for ingestion.")

        ids = [f"chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": os.path.basename(PDF_PATH), "chunk_index": i} for i in range(len(chunks))]

        print("Adding chunks to vector store (this may take some time)...")
        ingestion_start_time = time.time()
        try:
            await vector_store.add_documents(documents=chunks, ids=ids, metadatas=metadatas)
            ingestion_time = time.time() - ingestion_start_time
            print(f"Ingestion complete. Took {ingestion_time:.2f} seconds.")
        except Exception as e:
            print(f"Error during document ingestion: {e}")
            return None
    else:
        # If DB exists and has data, just connect to it
        vector_store = ChromaStore(
            path=CHROMA_DB_PATH,
            collection_name=COLLECTION_NAME,
            embedding_function=DEFAULT_EF
        )
        print(f"Connected to existing ChromaDB collection '{COLLECTION_NAME}'.")


    # Initialize LLM Service
    print(f"Initializing LLM Service: {LLM_SERVICE_CLASS.__name__} with model {LLM_MODEL}")
    try:
        # Specific initialization if OllamaService needs it
        if LLM_SERVICE_CLASS.__name__ == "OllamaService" or LLM_SERVICE_CLASS.__name__ == "OllamaOpenAICompatService":
            # Assuming OLLAMA_HOST is defined if OllamaService is chosen
            ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            llm_service = LLM_SERVICE_CLASS(default_model=LLM_MODEL, base_url=f"{ollama_host}/v1")
        else:
            llm_service = LLM_SERVICE_CLASS()
    except Exception as e:
        print(f"ERROR: Failed to initialize LLM Service: {e}")
        return None

    # Initialize CLAP RAG Agent
    rag_agent = Agent(
        name="Interactive_RAG_Expert",
        backstory="I am an expert assistant. I will answer your questions based on the content of the 'Hands-On Machine Learning' book. I will retrieve relevant sections from the book and use them to formulate my answer.",
        task_description="Answer user queries based on the ML book.", # Generic, will be replaced by user input
        task_expected_output="A concise and accurate answer derived solely from the retrieved book context.",
        llm_service=llm_service,
        model=LLM_MODEL,
        vector_store=vector_store
    )
    print("CLAP RAG Agent initialized and ready for queries.")
    print("--- RAG System Initialization Complete ---")
    return rag_agent

async def interactive_query_loop(rag_agent: Agent):
    """
    Runs an interactive loop prompting the user for queries and getting answers.
    """
    print("\n--- Interactive RAG Query Mode ---")
    print("Type your query about the ML book, or type 'exit' or 'quit' to end.")

    while True:
        try:
            user_query = await asyncio.to_thread(input, "\nYour Query: ") # Run input in a thread to not block loop
        except RuntimeError: # Fallback for environments where asyncio.to_thread might not work with input()
            user_query = input("\nYour Query: ")


        if user_query.lower() in ["exit", "quit"]:
            print("Exiting interactive query mode.")
            break
        if not user_query.strip():
            continue

        print(f"Processing query: '{user_query}'...")
        query_start_time = time.time()

        # Set the current query as the agent's task description for this run
        rag_agent.task_description = user_query
        
        try:
            result = await rag_agent.run() # CLAP Agent's run method
            answer = result.get("output", "Agent failed to produce an answer for this query.")
        except Exception as e:
            print(f"Error during agent run: {e}")
            answer = "An error occurred while processing your query with the agent."

        query_time = time.time() - query_start_time
        
        print("\n--- Agent's Answer ---")
        print(answer)
        print(f"----------------------\n(Query processed in {query_time:.2f} seconds)")

async def main():
    rag_agent = await initialize_rag_system()

    if rag_agent:
        await interactive_query_loop(rag_agent)
        # Clean up LLM service if it has a close method
        if hasattr(rag_agent.react_agent.llm_service, 'close') and \
           asyncio.iscoroutinefunction(rag_agent.react_agent.llm_service.close):
            print("Closing LLM service...")
            await rag_agent.react_agent.llm_service.close()
    else:
        print("Failed to initialize RAG system. Exiting.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting.")
    finally:
        # Optional: cleanup DB on exit, or leave it for persistence
        # if os.path.exists(CHROMA_DB_PATH):
        #     print(f"Cleaning up test database: {CHROMA_DB_PATH}")
        #     shutil.rmtree(CHROMA_DB_PATH)
        print("Script finished.")