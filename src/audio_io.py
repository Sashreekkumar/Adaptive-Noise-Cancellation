import soundfile as sf


# Load audio file as fp32
def load_audio(path) -> tuple:
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return x, sr


# Save audio file as fp32 .wav
def save_audio(path, x, sr) -> None:
    sf.write(str(path), x, sr, subtype="FLOAT")