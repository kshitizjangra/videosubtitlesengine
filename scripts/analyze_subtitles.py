import os

# ✅ Define Input Folder
INPUT_FOLDER = "cleaned_subtitles"

# ✅ Store token counts
token_counts = {}

# ✅ Process each subtitle file
for filename in os.listdir(INPUT_FOLDER):
    file_path = os.path.join(INPUT_FOLDER, filename)

    if filename.endswith((".srt", ".vtt", ".ass", ".nfo")):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        # Tokenize (split by spaces)
        tokens = text.split()
        token_counts[filename] = len(tokens)

# ✅ Sorting by file size (largest first)
sorted_files = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)

# ✅ Print top 10 largest subtitle files
print("\n📌 Top 10 Largest Subtitle Files by Token Count:")
for i, (file, count) in enumerate(sorted_files[:10]):
    print(f"{i+1}. {file} - {count} tokens")

# ✅ Print summary
total_files = len(token_counts)
average_tokens = sum(token_counts.values()) / total_files if total_files > 0 else 0

print(f"\n✅ Total Files Analyzed: {total_files}")
print(f"📊 Average Tokens per File: {int(average_tokens)}")
