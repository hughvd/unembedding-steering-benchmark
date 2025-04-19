# init_env.py
import sys
from pathlib import Path

def set_project_root(levels_up=1):
    """Add the project root to sys.path and return it as a Path object."""
    root = Path.cwd()
    for _ in range(levels_up):
        root = root.parent
    sys.path.insert(0, str(root))
    return root
