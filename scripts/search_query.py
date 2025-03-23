from chromadb import PersistentClient

# ChromaDB client initialize karo; ensure the path is correct relative to where you run this script.
chroma_client = PersistentClient(path="chroma_db")

# Get list of collection names (in v0.6.0, list_collections returns a list of strings)
collection_names = chroma_client.list_collections()
print("✅ Existing collections in ChromaDB:", collection_names)
