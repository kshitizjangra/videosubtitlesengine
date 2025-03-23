import os
import math

# Define paths
INPUT_FOLDER = "cleaned_subtitles"  # Folder with cleaned subtitles
OUTPUT_FOLDER = "chunked_subtitles"  # Folder where chunked files will be saved
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Chunking parameters
CHUNK_SIZE = 500      # number of tokens per chunk
OVERLAP_SIZE = 50     # number of tokens to overlap between chunks

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP_SIZE):
    """
    Splits text into chunks of `chunk_size` tokens with `overlap` tokens overlap.
    Returns a list of chunked texts.
    """
    tokens = text.split()  # simple whitespace tokenizer
    if not tokens:
        return []
    
    chunks = []
    start = 0
    total_tokens = len(tokens)
    
    while start < total_tokens:
        end = start + chunk_size
        chunk = tokens[start:end]
        chunks.append(" ".join(chunk))
        if end >= total_tokens:
            break
        start = end - overlap  # move back for overlap
    return chunks

# Process each cleaned subtitle file
chunk_count = 0
for filename in os.listdir(INPUT_FOLDER):
    if filename.endswith((".srt", ".vtt", ".ass", ".nfo")):
        file_path = os.path.join(INPUT_FOLDER, filename)
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        
        # Generate chunks for this file
        chunks = chunk_text(text)
        
        # Save chunks to a new file; here we can save each file's chunks in a separate text file.
        # Alternatively, you could save each chunk as a separate file.
        output_file_path = os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(filename)[0]}_chunks.txt")
        with open(output_file_path, "w", encoding="utf-8") as out_f:
            for i, chunk in enumerate(chunks):
                out_f.write(f"--- CHUNK {i+1} ---\n")
                out_f.write(chunk + "\n\n")
                chunk_count += 1
        
        print(f"✅ Processed: {filename} -> {len(chunks)} chunks")

print(f"\n🎯 Chunking complete! Total chunks created: {chunk_count}")
