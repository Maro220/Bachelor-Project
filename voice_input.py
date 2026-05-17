import sounddevice as sd
import soundfile as sf
import whisper
import numpy as np
import tempfile, os

model = whisper.load_model("small")  #

def record_and_transcribe(duration=10, samplerate=16000):
    print("Recording...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate,
                   channels=1, dtype='float32')
    sd.wait()
    print("Transcribing...")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, samplerate)
        result = model.transcribe(f.name)
        os.unlink(f.name)

    return result["text"]