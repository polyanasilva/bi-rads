"""Caminhos do projeto."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
RAW_DATA = DATA / "raw"
PROCESSED_DATA = DATA / "processed"

NOTEBOOKS = ROOT / "notebooks"
RESULTS = ROOT / "results"
