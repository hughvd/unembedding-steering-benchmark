# init_env.py
import sys
from pathlib import Path

def set_source_root():
    # __file__ will be "<root>/init_env.py"
    root = Path(__file__).resolve().parent
    src  = root / "src"
    sys.path.insert(0, str(src))
    return root