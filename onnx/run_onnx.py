"""
run_onnx.py

Entry point for the standalone ONNX pipeline. Usage:
    python run_onnx.py input.mp3 -o clean.mp3 --onnx_path deepfilternet3.onnx
    python run_onnx.py ./noisy_dir -o ./clean_dir --onnx_path deepfilternet3.onnx

Place this next to the onnx/ folder (as a sibling of it and of
audio_cleaner/), not inside it.
"""

from onnx.cli import main

if __name__ == "__main__":
    main()
