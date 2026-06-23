import os
import subprocess
import whisper

from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.chroma_store import add_documents


# --------------------------------------------------
# Load Whisper Model (Once)
# --------------------------------------------------
model = whisper.load_model("medium")


# --------------------------------------------------
# Normalize Audio using FFmpeg
# --------------------------------------------------
def normalize_audio(input_audio, output_audio):

    command = [
        "ffmpeg",
        "-i", input_audio,
        "-ar", "16000",
        "-ac", "1",
        output_audio,
        "-y"
    ]

    subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )


# --------------------------------------------------
# Process Audio: Audio → Text → Chunks → Store
# --------------------------------------------------
def process_audio(audio_path):
    """
    Converts audio → text → chunks → stores in ChromaDB
    """

    temp_audio = "temp.wav"

    # Normalize audio
    normalize_audio(audio_path, temp_audio)

    print("🎧 Transcribing audio...")


    # Transcribe
    result = model.transcribe(temp_audio)

    text = result["text"]
    language = result["language"]


    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)


    # Prepare documents
    documents = []

    for i, chunk in enumerate(chunks):

        documents.append({
            "text": chunk,
            "metadata": {
                "source": "audio",
                "filename": os.path.basename(audio_path),
                "language": language,
                "chunk_id": i
            }
        })


    # Store in ChromaDB
    add_documents(documents)


    # Clean temp file
    os.remove(temp_audio)


    return language, text


