"""
config.py

Shared defaults for every stage of the pipeline. Import from here
instead of hardcoding numbers in individual stage modules, so tuning
a value (e.g. the band-pass edges) only happens in one place.
"""

MODEL_SR = 48000          # DeepFilterNet3's native sample rate
HPF_CUTOFF_HZ = 70        # pre-enhancement high-pass cutoff
TARGET_DBFS = -26.0       # RMS loudness normalization target
BAND_LOW_HZ = 80          # post-enhancement band-pass, low edge
BAND_HIGH_HZ = 8000       # post-enhancement band-pass, high edge
MP3_BITRATE = "192k"
