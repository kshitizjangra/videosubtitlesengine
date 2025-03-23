from transformers import pipeline
import os

def get_user_query(input_type="text", force_language=None):
    """
    Accepts user input: either a text query or a path to an audio (.wav) file.
    If the input is audio, it uses the Hugging Face Whisper pipeline to transcribe the audio.
    
    Parameters:
      input_type (str): "text" or "audio"
      force_language (str or None): If provided (e.g., "en"), transcription is forced in that language.
                                    Otherwise, automatic language detection is used.
    
    Returns:
      str: The user's query as text.
    """
    if input_type == "text":
        user_query = input("🔹 Enter your search query: ")
        return user_query
    
    elif input_type == "audio":
        audio_path = input("🎤 Enter path of your audio file (.wav): ").strip()
        
        if not os.path.exists(audio_path):
            print("❌ File not found!")
            return None

        print("🎧 Processing audio with Whisper...")
        # Load Whisper model via Hugging Face pipeline
        whisper_model = pipeline("automatic-speech-recognition", model="openai/whisper-small")
        
        # If force_language is provided, pass it in generate_kwargs to force transcription in that language.
        if force_language:
            result = whisper_model(audio_path, 
                                   return_timestamps=True, 
                                   generate_kwargs={"task": "transcribe", "language": force_language})
        else:
            result = whisper_model(audio_path, return_timestamps=True)
        
        print(f"✅ Transcribed Text: {result['text']}\n")
        return result["text"]

# Test the function.
# For maximum accuracy on your English dataset, we force language to English.
query = get_user_query(input_type="audio", force_language="en")
print("Final Query:", query)
