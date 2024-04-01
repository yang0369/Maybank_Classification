import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
if ROOT not in sys.path:
    sys.path.append(ROOT)
