import os
import re

# ✅ Define Input & Output Folders
INPUT_FOLDER = "subtitles_extracted_data"   # Folder with extracted subtitles
OUTPUT_FOLDER = "cleaned_subtitles"         # Folder where cleaned files will be stored

# ✅ Ensure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ✅ Function to clean subtitle text
def clean_subtitle(text):
    # Remove timestamps (Format: 00:00:06,000 --> 00:00:12,074)
    text = re.sub(r"\d{2}:\d{2}:\d{2}[.,]\d{1,3} --> \d{2}:\d{2}:\d{2}[.,]\d{1,3}", "", text)
    text = re.sub(r"\d{2}:\d{2}:\d{2}", "", text)  # Remove standalone timestamps

    # Remove special characters & extra whitespace
    text = re.sub(r"[^a-zA-Z0-9\s.,!?']", " ", text)  # Keep only letters, numbers, and some punctuation
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces

    return text

# ✅ Process each subtitle file
for filename in os.listdir(INPUT_FOLDER):
    file_path = os.path.join(INPUT_FOLDER, filename)

    if filename.endswith((".srt", ".vtt", ".ass", ".nfo")):  # Process only subtitle files
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        # Clean text
        cleaned_text = clean_subtitle(text)

        # Save cleaned file
        cleaned_file_path = os.path.join(OUTPUT_FOLDER, filename)
        with open(cleaned_file_path, "w", encoding="utf-8") as f:
            f.write(cleaned_text)

        print(f"✅ Cleaned: {filename}")

print("\n🎯 Subtitle Cleaning Complete! Check the 'cleaned_subtitles' folder.")