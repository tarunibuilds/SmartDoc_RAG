from faster_whisper import WhisperModel
import os

print("Loading local model...")

model = WhisperModel(
    "medium",
    device="cpu",
    compute_type="int8"
)

def transcribe_audio(audio_path):
    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        vad_filter=True
    )

    text = ""
    for segment in segments:
        text += segment.text + " "

    return info.language, text.strip()

