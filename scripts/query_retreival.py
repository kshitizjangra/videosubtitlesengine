import chromadb
import os
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import re

# 1. Initialize Chroma and load your collection

client = chromadb.PersistentClient(path="chroma_db")  # Update path if needed
collection_name = "subtitle_chunks"  # Your collection name

collections = client.list_collections()
if collection_name not in collections:
    print(f"❌ Collection '{collection_name}' not found in ChromaDB! Check your database.")
    exit()

collection = client.get_collection(collection_name)

# 2. Load your models (Whisper + SentenceTransformer)

whisper_model = pipeline("automatic-speech-recognition", model="openai/whisper-small")
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def get_query_embedding(query_text):
    """Convert query text into an embedding."""
    return embedding_model.encode(query_text).tolist()

# 3. Cleanup function to remove unwanted parts

def cleanup_subtitle_text(text):
    """
    Example: Remove .org, 'www.opensubtitles.org', 'score', etc.
    Then trim or handle lines as you like.
    """
    if not text:
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove occurrences of domain
    text = re.sub(r'www\.opensubtitles\.org', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\.org', '', text, flags=re.IGNORECASE)

    # Remove 'score' references, 'download', etc.
    skip_phrases = ["download", "score", "uploader", "subtitles", "the benchmark", "we set the standards"]
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        lower_line = line.lower()
        # skip if any phrase is found
        if any(skip in lower_line for skip in skip_phrases):
            continue
        # remove extra spaces
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            cleaned_lines.append(line)

    cleaned_text = " ".join(cleaned_lines)

    # Optionally limit to e.g. 300 chars:
    max_len = 300
    if len(cleaned_text) > max_len:
        cleaned_text = cleaned_text[:max_len] + "..."
    
    return cleaned_text

# 4. Searching in ChromaDB

def search_in_chromadb(query_text, top_k=5):
    query_embedding = get_query_embedding(query_text)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    if not results["documents"]:
        print("❌ No matching results found.")
        return

    # We get documents & distances
    docs = results["documents"][0]
    dists = results["distances"][0]

    # Sort them by distance in DESCENDING order (largest distance first)
    # If you want best match first, you'd sort ascending instead!
    combined = list(zip(docs, dists))
    combined.sort(key=lambda x: x[1], reverse=True)  # reverse=True => descending

    print("\n🔍 **Search Results (descending distance)**:")
    for idx, (subtitle, dist) in enumerate(combined, start=1):
        # Clean up the text
        cleaned = cleanup_subtitle_text(subtitle)
        print(f"{idx}. {cleaned} (Score: {dist:.4f})")

# 5. Handling user input (text or audio)
def get_user_query(input_type="text"):
    if input_type == "text":
        query = input("🔹 Enter your search query: ")
        return query
    elif input_type == "audio":
        audio_path = input("🎤 Enter path of your audio file (.wav): ").strip()
        if not os.path.exists(audio_path):
            print("❌ File not found!")
            return None
        print("🎧 Transcribing audio with Whisper...")
        result = whisper_model(audio_path, return_timestamps=False)
        return result["text"]

# 6. Run
if __name__ == "__main__":
    user_query = get_user_query(input_type="text")  # or "audio"
    if user_query:
        search_in_chromadb(user_query, top_k=5)
