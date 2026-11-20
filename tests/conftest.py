import sys
from pathlib import Path

# Make the repository root importable (benchmarks, core), as for the Main_*.py entrypoints
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
