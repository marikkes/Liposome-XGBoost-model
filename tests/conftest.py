from pathlib import Path
import sys

# Allow tests to import modules from the ml folder (scripts use local-style imports).
ML_DIR = Path(__file__).resolve().parent.parent / "ml"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))
