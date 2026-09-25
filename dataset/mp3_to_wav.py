"""
Parallel MP3 -> WAV converter

- 8 parallel FFmpeg workers
- Processes only the first 50,000 MP3 files
- Saves WAV files to a "wav" folder in the project root
- Deletes MP3 only after successful WAV conversion
- Skips already-converted files
- Prints progress every 500 files
- Bounded task queue for memory efficiency

Requirements:
    sudo apt install ffmpeg
"""

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import subprocess
import time


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent


# ============================================================
# INPUT DIRECTORY
# ============================================================

INPUT_DIR = PROJECT_ROOT / "clips"


# ============================================================
# OUTPUT DIRECTORY
# ============================================================
# WAV files will be saved to:
#
#     PROJECT_ROOT/
#         wav/
#
# The folder will be created automatically if it does not exist.
# ============================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "wav"
)


# ============================================================
# SETTINGS
# ============================================================

MAX_FILES = 165340

WORKERS = 8

SAMPLE_RATE = 48000
CHANNELS = 1

QUEUE_SIZE = WORKERS * 4

PROGRESS_EVERY = 500


# ============================================================
# SINGLE FILE CONVERSION
# ============================================================

def convert_file(mp3_path_str):

    mp3_path = Path(mp3_path_str)

    wav_path = OUTPUT_DIR / f"{mp3_path.stem}.wav"


    # --------------------------------------------------------
    # Already converted
    # --------------------------------------------------------

    if wav_path.exists() and wav_path.stat().st_size > 0:

        try:
            mp3_path.unlink()

            return (
                "skipped",
                mp3_path.name,
            )

        except Exception as e:

            return (
                "failed",
                mp3_path.name,
                f"Could not delete MP3: {e}",
            )


    # --------------------------------------------------------
    # Convert MP3 -> WAV
    # --------------------------------------------------------

    try:

        result = subprocess.run(
            [
                "ffmpeg",

                "-hide_banner",
                "-loglevel", "error",
                "-y",

                "-i",
                str(mp3_path),

                "-ar",
                str(SAMPLE_RATE),

                "-ac",
                str(CHANNELS),

                "-c:a",
                "pcm_s16le",

                str(wav_path),
            ],

            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )


        # ----------------------------------------------------
        # FFmpeg failed
        # ----------------------------------------------------

        if result.returncode != 0:

            if wav_path.exists():

                try:
                    wav_path.unlink()
                except Exception:
                    pass

            error = result.stderr.strip()

            if not error:
                error = "FFmpeg returned a non-zero exit code."

            return (
                "failed",
                mp3_path.name,
                error,
            )


        # ----------------------------------------------------
        # Verify WAV
        # ----------------------------------------------------

        if (
            not wav_path.exists()
            or wav_path.stat().st_size == 0
        ):

            if wav_path.exists():

                try:
                    wav_path.unlink()
                except Exception:
                    pass

            return (
                "failed",
                mp3_path.name,
                "WAV was not created or is empty.",
            )


        # ----------------------------------------------------
        # SUCCESS — delete MP3
        # ----------------------------------------------------

        mp3_path.unlink()

        return (
            "converted",
            mp3_path.name,
        )


    except Exception as e:

        return (
            "failed",
            mp3_path.name,
            str(e),
        )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Check input directory
    # --------------------------------------------------------

    if not INPUT_DIR.exists():

        print("ERROR: Input directory does not exist:")
        print(INPUT_DIR)
        return


    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print("=" * 75)
    print("Parallel MP3 -> WAV conversion")
    print("=" * 75)

    print(f"Input directory  : {INPUT_DIR}")
    print(f"Output directory : {OUTPUT_DIR}")
    print(f"Maximum files    : {MAX_FILES:,}")
    print(f"Workers          : {WORKERS}")
    print(f"Queue size       : {QUEUE_SIZE}")
    print(
        f"WAV format       : "
        f"{SAMPLE_RATE} Hz / "
        f"{CHANNELS} channel / "
        f"16-bit PCM"
    )

    print()


    # --------------------------------------------------------
    # Find MP3 files
    # --------------------------------------------------------

    print("Scanning for MP3 files...")

    mp3_files = sorted(
        INPUT_DIR.glob("*.mp3")
    )

    total_available = len(mp3_files)


    if total_available == 0:

        print("No MP3 files found.")
        return


    # --------------------------------------------------------
    # Limit to first 50,000
    # --------------------------------------------------------

    mp3_files = mp3_files[:MAX_FILES]

    total = len(mp3_files)


    print(
        f"Found {total_available:,} MP3 files."
    )

    print(
        f"Processing first {total:,} files."
    )

    print()


    # --------------------------------------------------------
    # Start conversion
    # --------------------------------------------------------

    print("Starting conversion...")
    print()

    start_time = time.time()

    converted = 0
    skipped = 0
    failed = 0
    processed = 0


    # --------------------------------------------------------
    # Failed-file log
    # --------------------------------------------------------

    failed_log = (
        OUTPUT_DIR
        / "conversion_failed.txt"
    )


    with open(
        failed_log,
        "a",
        encoding="utf-8",
    ) as log:

        with ProcessPoolExecutor(
            max_workers=WORKERS
        ) as executor:

            file_iterator = iter(mp3_files)

            futures = {}


            # =================================================
            # Initial batch
            # =================================================

            for _ in range(
                min(QUEUE_SIZE, total)
            ):

                try:
                    mp3_path = next(file_iterator)

                except StopIteration:
                    break

                future = executor.submit(
                    convert_file,
                    str(mp3_path),
                )

                futures[future] = mp3_path


            # =================================================
            # Process completed tasks
            # =================================================

            while futures:

                done, _ = wait(
                    futures,
                    return_when=FIRST_COMPLETED,
                )


                for future in done:

                    mp3_path = futures.pop(
                        future
                    )


                    # -----------------------------------------
                    # Get result
                    # -----------------------------------------

                    try:

                        result = future.result()

                    except Exception as e:

                        result = (
                            "failed",
                            mp3_path.name,
                            str(e),
                        )


                    status = result[0]


                    # -----------------------------------------
                    # Update counters
                    # -----------------------------------------

                    if status == "converted":

                        converted += 1

                    elif status == "skipped":

                        skipped += 1

                    elif status == "failed":

                        failed += 1

                        filename = result[1]
                        error = result[2]

                        log.write(
                            f"{filename}\n"
                            f"{error}\n"
                            f"{'-' * 60}\n"
                        )

                        log.flush()


                    processed += 1


                    # -----------------------------------------
                    # Submit next file
                    # -----------------------------------------

                    try:

                        next_mp3 = next(
                            file_iterator
                        )

                        new_future = (
                            executor.submit(
                                convert_file,
                                str(next_mp3),
                            )
                        )

                        futures[new_future] = (
                            next_mp3
                        )

                    except StopIteration:

                        pass


                    # -----------------------------------------
                    # Progress
                    # -----------------------------------------

                    if (
                        processed % PROGRESS_EVERY == 0
                        or processed == total
                    ):

                        elapsed = (
                            time.time()
                            - start_time
                        )

                        speed = (
                            processed / elapsed
                            if elapsed > 0
                            else 0
                        )

                        remaining = (
                            total - processed
                        )

                        eta_seconds = (
                            remaining / speed
                            if speed > 0
                            else 0
                        )

                        eta_hours = (
                            eta_seconds / 3600
                        )

                        percent = (
                            processed
                            / total
                            * 100
                        )

                        print(
                            f"{processed:,}/{total:,} "
                            f"({percent:.2f}%) | "
                            f"Converted: {converted:,} | "
                            f"Skipped: {skipped:,} | "
                            f"Failed: {failed:,} | "
                            f"Speed: {speed:.2f} files/s | "
                            f"ETA: {eta_hours:.2f} h"
                        )


    # ========================================================
    # Final statistics
    # ========================================================

    elapsed = time.time() - start_time


    print()
    print("=" * 75)
    print("CONVERSION COMPLETE")
    print("=" * 75)

    print(
        f"Total processed : {processed:,}"
    )

    print(
        f"Converted       : {converted:,}"
    )

    print(
        f"Skipped         : {skipped:,}"
    )

    print(
        f"Failed          : {failed:,}"
    )

    print(
        f"Time            : "
        f"{elapsed / 3600:.2f} hours"
    )


    if elapsed > 0:

        print(
            f"Average speed   : "
            f"{processed / elapsed:.2f} files/sec"
        )


    print()
    print(
        f"WAV directory   : {OUTPUT_DIR}"
    )

    print(
        f"Failed log      : {failed_log}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()