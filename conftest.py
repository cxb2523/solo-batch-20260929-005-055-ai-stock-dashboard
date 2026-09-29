import sys
from pathlib import Path

# 保证 `python -m pytest -q` 在任何工作目录下都能 import dataflow 与 trace_app
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
