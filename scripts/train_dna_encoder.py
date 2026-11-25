#!/usr/bin/env python3
"""Server entry point; runnable from any directory without installing this repo."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dna_training.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
