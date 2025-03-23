import chromadb  

# Ensure this matches your actual ChromaDB directory
persist_directory = "chroma_db"  

# Connect to the database
client = chromadb.PersistentClient(path=persist_directory)  

# ✅ List existing collections
collections = client.list_collections()
if collections:
    print("✅ Existing Collections:")
    for col in collections:
        print(f"- {col}")  # Now, col is just a name string
else:
    print("❌ No collections found!")
    exit()  # Stop execution if no collections exist

# ✅ Retrieve the subtitle_chunks collection
collection_name = "subtitle_chunks"
if collection_name in [col for col in collections]:
    collection = client.get_collection(collection_name)
    print(f"✅ Total embeddings stored: {collection.count()}")
else:
    print(f"❌ Collection '{collection_name}' does not exist!")
