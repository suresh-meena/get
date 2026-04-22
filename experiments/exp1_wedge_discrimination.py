from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stage1.wedge_discrimination import generate_matched_dataset, main

if __name__ == "__main__":
    main()
