"""
Parallel WAV -> MP3 converter

- 8 parallel FFmpeg workers
- Processes only the first 165,340 WAV files
- Saves MP3 files to an "mp3" folder in the project root
- Deletes WAV only after successful MP3 conversion
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

INPUT_DIR = Path(r"C:\Users\sashr\Documents\Automatic-Noise-Cancellation\data\output\train\clean")


# ============================================================
# OUTPUT DIRECTORY
# ============================================================
# MP3 files will be saved to:
#
#     PROJECT_ROOT/
#         mp3/
#
# The folder will be created automatically if it does not exist.
# ============================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "cleanmp3"
)


# ============================================================
# SETTINGS
# ============================================================

MAX_FILES = 165340

WORKERS = 32

SAMPLE_RATE = 48000
CHANNELS = 1

# MP3 encoding quality
# Use a constant bitrate, or switch to VBR by setting MP3_VBR_QUALITY
MP3_BITRATE = "192k"

# If you'd rather use VBR instead of a fixed bitrate, set this to a value
# 0 (best) - 9 (worst) and set USE_VBR = True below. Otherwise leave alone.
MP3_VBR_QUALITY = 2
USE_VBR = False

QUEUE_SIZE = WORKERS * 4

PROGRESS_EVERY = 500


# ============================================================
# SINGLE FILE CONVERSION
# ============================================================

def convert_file(wav_path_str):

    wav_path = Path(wav_path_str)

    mp3_path = OUTPUT_DIR / f"{wav_path.stem}.mp3"


    # --------------------------------------------------------
    # Already converted
    # --------------------------------------------------------

    if mp3_path.exists() and mp3_path.stat().st_size > 0:

        try:
            wav_path.unlink()

            return (
                "skipped",
                wav_path.name,
            )

        except Exception as e:

            return (
                "failed",
                wav_path.name,
                f"Could not delete WAV: {e}",
            )


    # --------------------------------------------------------
    # Convert WAV -> MP3
    # --------------------------------------------------------

    try:

        cmd = [
            "ffmpeg",

            "-hide_banner",
            "-loglevel", "error",
            "-y",

            "-i",
            str(wav_path),

            "-ar",
            str(SAMPLE_RATE),

            "-ac",
            str(CHANNELS),

            "-c:a",
            "libmp3lame",
        ]

        if USE_VBR:
            cmd += ["-q:a", str(MP3_VBR_QUALITY)]
        else:
            cmd += ["-b:a", MP3_BITRATE]

        cmd += [str(mp3_path)]

        result = subprocess.run(
            cmd,

            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )


        # ----------------------------------------------------
        # FFmpeg failed
        # ----------------------------------------------------

        if result.returncode != 0:

            if mp3_path.exists():

                try:
                    mp3_path.unlink()
                except Exception:
                    pass

            error = result.stderr.strip()

            if not error:
                error = "FFmpeg returned a non-zero exit code."

            return (
                "failed",
                wav_path.name,
                error,
            )


        # ----------------------------------------------------
        # Verify MP3
        # ----------------------------------------------------

        if (
            not mp3_path.exists()
            or mp3_path.stat().st_size == 0
        ):

            if mp3_path.exists():

                try:
                    mp3_path.unlink()
                except Exception:
                    pass

            return (
                "failed",
                wav_path.name,
                "MP3 was not created or is empty.",
            )


        # ----------------------------------------------------
        # SUCCESS — delete WAV
        # ----------------------------------------------------

        wav_path.unlink()

        return (
            "converted",
            wav_path.name,
        )


    except Exception as e:

        return (
            "failed",
            wav_path.name,
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
    print("Parallel WAV -> MP3 conversion")
    print("=" * 75)

    print(f"Input directory  : {INPUT_DIR}")
    print(f"Output directory : {OUTPUT_DIR}")
    print(f"Maximum files    : {MAX_FILES:,}")
    print(f"Workers          : {WORKERS}")
    print(f"Queue size       : {QUEUE_SIZE}")

    if USE_VBR:
        print(
            f"MP3 format       : "
            f"{SAMPLE_RATE} Hz / "
            f"{CHANNELS} channel / "
            f"VBR quality {MP3_VBR_QUALITY}"
        )
    else:
        print(
            f"MP3 format       : "
            f"{SAMPLE_RATE} Hz / "
            f"{CHANNELS} channel / "
            f"{MP3_BITRATE} CBR"
        )

    print()


    # --------------------------------------------------------
    # Find WAV files
    # --------------------------------------------------------

    print("Scanning for WAV files...")

    wav_files = sorted(
        INPUT_DIR.glob("*.wav")
    )

    total_available = len(wav_files)


    if total_available == 0:

        print("No WAV files found.")
        return


    # --------------------------------------------------------
    # Limit to first MAX_FILES
    # --------------------------------------------------------

    wav_files = wav_files[:MAX_FILES]

    total = len(wav_files)


    print(
        f"Found {total_available:,} WAV files."
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

            file_iterator = iter(wav_files)

            futures = {}


            # =================================================
            # Initial batch
            # =================================================

            for _ in range(
                min(QUEUE_SIZE, total)
            ):

                try:
                    wav_path = next(file_iterator)

                except StopIteration:
                    break

                future = executor.submit(
                    convert_file,
                    str(wav_path),
                )

                futures[future] = wav_path


            # =================================================
            # Process completed tasks
            # =================================================

            while futures:

                done, _ = wait(
                    futures,
                    return_when=FIRST_COMPLETED,
                )


                for future in done:

                    wav_path = futures.pop(
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
                            wav_path.name,
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

                        next_wav = next(
                            file_iterator
                        )

                        new_future = (
                            executor.submit(
                                convert_file,
                                str(next_wav),
                            )
                        )

                        futures[new_future] = (
                            next_wav
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
        f"MP3 directory   : {OUTPUT_DIR}"
    )

    print(
        f"Failed log      : {failed_log}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()