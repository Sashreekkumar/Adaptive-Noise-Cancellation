
import numpy as np
import soundfile as sf
import librosa


# ============================================================
# CONFIG
# ============================================================

CLEAN_PATH = "dataset/common_voice_hi_23796049.wav"
NOISE_PATH = "dataset/helicoper.wav"
OUTPUT_PATH = "dataset/noisy.wav"

TARGET_SNR_DB = 5.0


# ============================================================
# MATCH NOISE LENGTH TO CLEAN AUDIO
# ============================================================

def match_length(signal: np.ndarray, target_len: int) -> np.ndarray:
    """
    Match the noise length to the clean speech length.

    If noise is longer:
        Randomly crop a section.

    If noise is shorter:
        Repeat/loop the noise until it is long enough.
    """

    if len(signal) >= target_len:
        start = np.random.randint(
            0,
            len(signal) - target_len + 1
        )

        return signal[start:start + target_len]

    repetitions = int(np.ceil(target_len / len(signal)))

    return np.tile(signal, repetitions)[:target_len]


# ============================================================
# MIX AT TARGET SNR
# ============================================================

def mix_at_snr(
    clean: np.ndarray,
    noise: np.ndarray,
    snr_db: float
) -> np.ndarray:

    # Match noise duration to clean speech
    noise = match_length(noise, len(clean))

    # Calculate signal powers
    clean_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)

    # Avoid division by zero
    if noise_power == 0:
        print("Warning: noise file contains no usable signal.")
        return clean.copy()

    if clean_power == 0:
        raise ValueError("Clean audio contains no usable signal.")

    # Calculate required noise power for target SNR
    target_noise_power = (
        clean_power / (10 ** (snr_db / 10))
    )

    # Calculate noise scaling factor
    scale = np.sqrt(
        target_noise_power / noise_power
    )

    # Mix
    noisy = clean + noise * scale

    # Prevent clipping
    peak = np.max(np.abs(noisy))

    if peak > 1.0:
        noisy = noisy / peak

    return noisy


# ============================================================
# LOAD AUDIO
# ============================================================

print("Loading audio...")

clean, sr_clean = sf.read(CLEAN_PATH)
noise, sr_noise = sf.read(NOISE_PATH)


# ============================================================
# CONVERT TO MONO
# ============================================================

if clean.ndim > 1:
    print("Converting clean audio to mono...")
    clean = clean.mean(axis=1)

if noise.ndim > 1:
    print("Converting noise audio to mono...")
    noise = noise.mean(axis=1)


# ============================================================
# RESAMPLE NOISE
# ============================================================

if sr_noise != sr_clean:

    print(
        f"Resampling noise: "
        f"{sr_noise} Hz → {sr_clean} Hz"
    )

    noise = librosa.resample(
        noise.astype(np.float32),
        orig_sr=sr_noise,
        target_sr=sr_clean
    )

    sr_noise = sr_clean


# ============================================================
# MIX
# ============================================================

print(
    f"Mixing at {TARGET_SNR_DB} dB SNR..."
)

noisy = mix_at_snr(
    clean,
    noise,
    TARGET_SNR_DB
)


# ============================================================
# SAVE
# ============================================================

sf.write(
    OUTPUT_PATH,
    noisy,
    sr_clean
)


# ============================================================
# DONE
# ============================================================

print()
print("========================================")
print("Mixing complete")
print("========================================")
print(f"Clean:       {CLEAN_PATH}")
print(f"Noise:       {NOISE_PATH}")
print(f"Output:      {OUTPUT_PATH}")
print(f"SNR:         {TARGET_SNR_DB} dB")
print(f"Sample rate: {sr_clean} Hz")
print("Channels:    mono")
print("========================================")
