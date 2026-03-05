import chromadb
from chromadb.utils import embedding_functions

class NalamRetriever:
    def __init__(self, db_path="./nalam_chroma_db", collection_name="nalam_knowledge"):
        """
        Initializes the connection to the ChromaDB vector store.
        """
        print(f"🔌 [Retriever] Connecting to DB at {db_path}...")
        
        # 1. Connect to persistent DB
        self.client = chromadb.PersistentClient(path=db_path)
        
        # 2. Define the Embedding Function 
        self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # 3. Get the Collection
        self.collection = self.client.get_collection(
            name=collection_name,
            embedding_function=self.embedding_func
        )
        print(f"✅ [Retriever] Connected to collection: {collection_name}")

    def get_relevant_context(self, query, top_k=5):
        """
        Retrieves the top_k most relevant text chunks for a given query.
        Returns a single string of combined context.
        """
        print(f"🔍 [Retriever] Searching for: '{query}'")
        
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        # Extract documents (list of lists, so we take index 0)
        documents = results['documents'][0]
        
        if not documents:
            return ""
            
        # Join chunks with a separator for clarity
        combined_context = "\n\n".join(documents)
        print(f"📄 [Retriever] Found {len(documents)} relevant documents.")
        return combined_context