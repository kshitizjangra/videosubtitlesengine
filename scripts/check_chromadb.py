import chromadb

# --------------------------
# Connect to ChromaDB
# --------------------------
client = chromadb.PersistentClient(path="chroma_db")  # Ensure path is correct
collection = client.get_collection("subtitle_chunks")  # Ensure correct collection name

# --------------------------
# Test Search Query
# --------------------------
query_text = "Titanic ship sinking scene"  # Use something you expect in subtitles

from sentence_transformers import SentenceTransformer

# Define the function to get query embedding
def get_query_embedding(query_text):
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Load a pre-trained model
    return model.encode(query_text)

query_embedding = get_query_embedding(query_text)
search_results = collection.query(query_embeddings=[query_embedding], n_results=5)

print("🔎 **Search Results from ChromaDB:**")
for idx, (doc, score) in enumerate(zip(search_results["documents"][0], search_results["distances"][0])):
    print(f"\n📌 **Result {idx+1}** (Score: {score:.4f})\n📝 Subtitle Chunk: {doc}\n")
