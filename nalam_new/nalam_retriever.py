import chromadb
from chromadb.utils import embedding_functions


class NalamRetriever:
    def __init__(self, db_path="./nalam_chroma_db", collection_name="nalam_knowledge"):
        """
        Initializes the connection to the ChromaDB vector store.
        """
        print(f"🔌 [Retriever] Connecting to DB at {db_path}...")

        # 1️⃣ Connect to persistent DB
        self.client = chromadb.PersistentClient(path=db_path)

        # 2️⃣ Define embedding model
        self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        # 3️⃣ Get collection
        self.collection = self.client.get_collection(
            name=collection_name,
            embedding_function=self.embedding_func
        )

        print(f"✅ [Retriever] Connected to collection: {collection_name}")

    def get_relevant_context(self, query: str, top_k: int = 5) -> str:
        """
        Retrieves the top_k most relevant unique text chunks for a given query.
        Returns a single combined context string.
        """

        print(f"🔍 [Retriever] Searching for: '{query}'")

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )

            documents = results.get("documents", [[]])[0]

            if not documents:
                print("⚠️ [Retriever] No documents found.")
                return ""

            # 1️⃣ Remove duplicates while preserving order
            seen = set()
            unique_documents = []

            for doc in documents:
                clean_doc = doc.strip()

                if clean_doc not in seen:
                    seen.add(clean_doc)
                    unique_documents.append(clean_doc)

            print(f"📄 [Retriever] Retrieved {len(documents)} chunks.")
            print(f"✅ [Retriever] {len(unique_documents)} unique chunks after filtering.")

            # 2️⃣ Combine context
            combined_context = "\n\n".join(unique_documents)

            return combined_context

        except Exception as e:
            print(f"❌ [Retriever] Error during retrieval: {e}")
            return ""