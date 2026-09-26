from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import os

# =========================
# CONFIG
# =========================
INPUT_DIR = Path(r"C:\Users\sashr\Documents\Automatic-Noise-Cancellation\data\VCTK-Corpus\wav48")
OUTPUT_DIR = Path(r"C:\Users\sashr\Documents\Automatic-Noise-Cancellation\data\clean")

WORKERS = min(32, (os.cpu_count() or 4) * 2)

AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".opus"
}


def copy_audio(src: Path):
    """
    Copy one audio file to the output directory.
    Automatically avoids filename collisions.
    """
    dst = OUTPUT_DIR / src.name

    # Prevent overwriting files with the same name
    if dst.exists():
        stem = src.stem
        suffix = src.suffix
        counter = 1

        while True:
            dst = OUTPUT_DIR / f"{stem}_{counter}{suffix}"
            if not dst.exists():
                break
            counter += 1

    shutil.copy2(src, dst)
    return src, dst


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Scanning for audio files...")

    files = [
        p
        for p in INPUT_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    ]

    print(f"Found {len(files):,} audio files")
    print(f"Using {WORKERS} workers")
    print(f"Output: {OUTPUT_DIR}")
    print()

    completed = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {
            executor.submit(copy_audio, src): src
            for src in files
        }

        for future in as_completed(futures):
            src = futures[future]

            try:
                future.result()
                completed += 1

                # Print only every 100 files
                if completed % 100 == 0:
                    print(
                        f"Copied {completed:,}/{len(files):,}"
                    )

            except Exception as e:
                failed += 1
                print(f"FAILED: {src} -> {e}")

    print("\nDone.")
    print(f"Successfully copied: {completed:,}")
    print(f"Failed: {failed:,}")


if __name__ == "__main__":
    main()