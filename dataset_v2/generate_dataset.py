#!/usr/bin/env python3
"""Entry point: `python generate_dataset.py --root . --num-samples 1000` (same as `python -m noisygen`)."""
import sys

from noisygen.cli import main

if __name__ == "__main__":
    sys.exit(main())
