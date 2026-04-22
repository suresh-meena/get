from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stage4.transfer_tasks import main

if __name__ == "__main__":
    raise SystemExit(main())
