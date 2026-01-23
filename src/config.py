"""
Centralized repository paths.

Why this exists:
- Ensures all scripts use the same, repo-relative paths
- Avoids hard-coded absolute paths (Google Drive, user home, etc.)
- Improves reproducibility across machines and judges' environments
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    """
    Repository path registry.

    ROOT is resolved relative to this file location:
      repo_root/
        src/config.py  <-- this file
    """
    ROOT: Path = Path(__file__).resolve().parents[1]

    # Data folders (do not hardcode personal locations)
    DATA: Path = ROOT / "data"
    RAW: Path = DATA / "raw"
    INTERIM: Path = DATA / "interim"
    PROCESSED: Path = DATA / "processed"

    # Outputs (figures, logs, exported reports, etc.)
    OUTPUTS: Path = ROOT / "outputs"

    # Optional: docs folder
    DOCS: Path = ROOT / "docs"


# Single shared instance used throughout the project
PATHS = Paths()
