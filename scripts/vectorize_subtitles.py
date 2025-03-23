import os
import json
from sentence_transformers import SentenceTransformer
import chromadb

# Step 1: Initialize ChromaDB Client


# Ensure the persistence directory exists
PERSIST_DIR = "chroma_db"
os.makedirs(PERSIST_DIR, exist_ok=True)

# Initialize ChromaDB client using PersistentClient (local, self-contained)
chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
collection = chroma_client.get_or_create_collection(name="subtitle_chunks")
print("✅ ChromaDB initialized and collection loaded.")

# Step 2: Clear Existing Data (for a Fresh Start)

existing = collection.get()
if existing["ids"]:
    # Delete existing records in batches to avoid batch size limits
    batch_size_delete = 10000
    total_ids = existing["ids"]
    for i in range(0, len(total_ids), batch_size_delete):
        batch = total_ids[i:i + batch_size_delete]
        collection.delete(ids=batch)
        print(f"✅ Deleted batch of IDs from {i} to {i+len(batch)}")
else:
    print("✅ No existing data found in collection.")

# Step 3: Load the SentenceTransformer Model

model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)
print(f"✅ Loaded model: {model_name}")

# Step 4: Process Chunk Files and Generate Embeddings

CHUNKS_FOLDER = "chunked_subtitles"       # Folder with chunked subtitle files
EMBEDDINGS_OUTPUT = "embeddings.json"      # Backup JSON file

# Gather all chunk files (assumed to be .txt files)
chunk_files = [os.path.join(CHUNKS_FOLDER, f) for f in os.listdir(CHUNKS_FOLDER) if f.endswith(".txt")]
print(f"📂 Found {len(chunk_files)} chunk files.")

# Prepare for batching and backup storage
embedding_data = []   # For optional JSON backup
batch_docs = []
batch_embeddings = []
batch_ids = []
batch_size = 1000     # Insert in batches of 1000
chunk_id = 0

def flush_batch():
    global batch_docs, batch_embeddings, batch_ids
    if batch_ids:
        collection.add(
            documents=batch_docs,
            embeddings=batch_embeddings,
            ids=batch_ids
        )
        print(f"✅ Batch inserted with {len(batch_ids)} items.")
        batch_docs = []
        batch_embeddings = []
        batch_ids = []

# Process each chunk file
for file_path in chunk_files:
    with open(file_path, "r", encoding="utf-8") as f:
        # Split file into chunks using the marker "--- CHUNK"
        file_content = f.read().split("--- CHUNK")
        for i, chunk in enumerate(file_content):
            chunk = chunk.strip()
            if not chunk:
                continue

            # Remove header line if present; get pure text
            lines = chunk.splitlines()
            text = " ".join(lines[1:]).strip() if len(lines) > 1 else lines[0].strip()

            if text:
                # Generate embedding for the text chunk
                embedding = model.encode(text).tolist()

                # Create a unique id for this chunk
                chunk_id += 1
                doc_id = f"chunk_{chunk_id}"

                # Append data to the batch lists
                batch_docs.append(text)
                batch_embeddings.append(embedding)
                batch_ids.append(doc_id)

                # Also save to backup list
                embedding_data.append({
                    "id": doc_id,
                    "text": text,
                    "embedding": embedding
                })

                if len(batch_ids) >= batch_size:
                    flush_batch()
                    print(f"🔥 Processed {chunk_id} chunks...")

# Flush any remaining items in the batch
flush_batch()

print(f"\n🎯 Total chunks processed: {chunk_id}")

# Step 5: Save Backup Embedding Data to a JSON File

with open(EMBEDDINGS_OUTPUT, "w", encoding="utf-8") as f_out:
    json.dump(embedding_data, f_out)
print("✅ Embedding generation and storage complete!")
