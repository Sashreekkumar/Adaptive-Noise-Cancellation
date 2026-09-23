import subprocess
import imageio_ffmpeg

INPUT_PATH = "dataset/common_voice_hi_23796049.mp3"
OUTPUT_PATH = "dataset/common_voice_hi_23796049.wav"

TARGET_SR = 48000
MONO = True

ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

cmd = [
    ffmpeg,
    "-y",
    "-i", INPUT_PATH,
    "-ar", str(TARGET_SR),
]

if MONO:
    cmd += ["-ac", "1"]

cmd += [OUTPUT_PATH]

subprocess.run(cmd, check=True)

print(f"Saved: {OUTPUT_PATH}")
print(f"Sample rate: {TARGET_SR} Hz")
print(f"Channels: {'mono' if MONO else 'stereo'}")