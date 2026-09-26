"""
run.py

Entry point. Usage:
    python run.py input.mp3 -o clean.mp3
    python run.py ./noisy_dir -o ./clean_dir
    python run.py --serve
"""

from audio_cleaner.cli import main

if __name__ == "__main__":
    main()
