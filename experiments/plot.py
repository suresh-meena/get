from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.shared.plotting import load_and_plot  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate seaborn plots from experiment result JSON files.")
    parser.add_argument("json_path", nargs="+", type=Path, help="One or more result JSON files")
    parser.add_argument("--title", default=None, help="Optional figure title")
    parser.add_argument("--output", default=None, help="Optional output path when plotting a single JSON file")
    args = parser.parse_args()

    if len(args.json_path) > 1 and args.output is not None:
        raise SystemExit("--output can only be used with a single JSON file")

    for json_path in args.json_path:
        output_path = Path(args.output) if args.output is not None else json_path.with_suffix(".png")
        saved = load_and_plot(json_path, output_path=output_path, title=args.title)
        print(f"Saved {saved}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())