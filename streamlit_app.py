import os
import streamlit as st
import chromadb
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import re
import numpy as np

# Initializing ChromaDB Client
client = chromadb.PersistentClient(path="chroma_db")  # Adjust path to match your folder
collection = client.get_collection("subtitle_chunks")  # Must match your stored collection name

# Load Models for Query Embeddings and Audio Transcription
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
whisper_model = pipeline("automatic-speech-recognition", model="openai/whisper-small")

def get_query_embedding(query_text: str):
    """
    Generate an embedding for the query using SentenceTransformer,
    then ensure it matches the dimension used in your ChromaDB collection.
    """
    query_embedding = embedding_model.encode(query_text)
    target_dimension = 384  # Adjust if your stored embeddings have a different dimension
    if len(query_embedding) != target_dimension:
        st.warning(f"⚠️ Embedding dimension mismatch. Resizing from {len(query_embedding)} → {target_dimension}.")
        query_embedding = np.resize(query_embedding, target_dimension)
    return query_embedding.tolist()

def cleanup_document(doc: str) -> str:
    """
    Clean up the retrieved subtitle text for display:
      - Remove URLs.
      - Remove extraneous lines containing skip-phrases.
      - Truncate to 300 characters.
      If cleaning removes all content, fall back to the original text (with URLs removed).
    """
    if not doc:
        return ""
    
    # Remove URLs
    doc_no_urls = re.sub(r'http\S+', '', doc)
    lines = doc_no_urls.splitlines()
    
    skip_phrases = [
        "movie information", "imdb link", "uploader", "download",
        "filename", "nfo created", "md5", "fps", "language",
        "format", "subtitles", "www.opensubtitles.org",
        "we set the standards", "the benchmark", "score", "org"
    ]
    
    cleaned_lines = []
    for line in lines:
        lower_line = line.lower()
        if any(skip in lower_line for skip in skip_phrases):
            continue
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            cleaned_lines.append(line)
    
    cleaned_doc = " ".join(cleaned_lines).strip()
    if not cleaned_doc:
        cleaned_doc = doc_no_urls.strip()
    
    max_len = 300
    if len(cleaned_doc) > max_len:
        cleaned_doc = cleaned_doc[:max_len] + "..."
    return cleaned_doc

def extract_movie_name(doc: str) -> str:
    """
    Attempt to extract a 'Release Name' from the document metadata.
    E.g., "Release Name - X Men Evolution" returns "X Men Evolution".
    Returns an empty string if not found.
    """
    if not doc:
        return ""
    match = re.search(r"Release Name\s*[:\-]\s*(.+)", doc, re.IGNORECASE)
    return match.group(1).strip() if match else ""

def search_subtitles(query_text: str, top_k=5):
    """
    Search for the top_k relevant subtitles in the ChromaDB collection
    using the query_text embedding. Sort results in descending order (higher scores first).
    """
    query_vec = get_query_embedding(query_text)
    try:
        results = collection.query(query_embeddings=[query_vec], n_results=top_k)
        if not results["documents"]:
            return [], []
        # Zip documents with distances
        zipped_results = list(zip(results["documents"][0], results["distances"][0]))
        # Sort descending by score (assuming higher score means better match)
        sorted_results = sorted(zipped_results, key=lambda x: x[1], reverse=True)
        # Unzip back to two lists
        docs_sorted, distances_sorted = zip(*sorted_results)
        return list(docs_sorted), list(distances_sorted)
    except Exception as e:
        st.error(f"Error during ChromaDB query: {e}")
        return [], []
    
# Streamlit Interface
st.title("🎬 Subtitles Search Engine")

mode = st.radio("Select Input Mode", ("Text", "Audio"))

if mode == "Text":
    user_input = st.text_input("🔹 Enter your search query:")
else:
    uploaded_file = st.file_uploader("🎤 Upload your audio (.wav) file", type=["wav"])
    user_input = None
    if uploaded_file is not None:
        temp_audio_path = "temp_audio.wav"
        with open(temp_audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.write("🎧 Processing audio with Whisper...")
        try:
            result = whisper_model(
                temp_audio_path,
                return_timestamps=True,
                generate_kwargs={"task": "transcribe", "language": "en"}
            )
            user_input = result["text"]
            st.write("✅ Transcribed Text:", user_input)
        except Exception as e:
            st.error(f"Whisper transcription failed: {str(e)}")
        finally:
            os.remove(temp_audio_path)

if st.button("Submit Query") and user_input:
    st.write("### User Query:")
    st.write(user_input)
    
    docs, distances = search_subtitles(user_input, top_k=5)
    
    if not docs:
        st.warning("❌ No relevant subtitles found.")
    else:
        st.write("### Top Matching Subtitles:")
        for i, (doc, score) in enumerate(zip(docs, distances), start=1):
            movie_name = extract_movie_name(doc)
            if movie_name:
                display_text = f"**Movie:** {movie_name}"
            else:
                display_text = cleanup_document(doc)
            st.write(f"**{i}.** {display_text} (Score: {score:.4f})")

st.write("---")
st.info("**Note**: If audio & text results differ, it may reflect differences in the actual transcribed text vs. typed text. This is normal for semantic search.")
