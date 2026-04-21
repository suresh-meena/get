import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_MAP = {
    "exp1": "exp1_wedge_discrimination.py",
    "exp2": "exp2_triangle_counting.py",
}


def _build_command(python_executable, script_path, passthrough_args):
    return [python_executable, str(script_path), *passthrough_args]


def _run_command(cmd, dry_run=False):
    printable = " ".join(cmd)
    print(f"[run_stage1] {printable}")
    if dry_run:
        return 0
    return subprocess.call(cmd)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Unified Stage-1 runner for GET experiments. "
            "Use --task exp1, --task exp2, or --task all and pass remaining flags through."
        )
    )
    parser.add_argument("--task", choices=["exp1", "exp2", "all"], default="all")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")

    args, passthrough = parser.parse_known_args()

    scripts_dir = Path(__file__).resolve().parent
    tasks = ["exp1", "exp2"] if args.task == "all" else [args.task]

    for task in tasks:
        script_file = scripts_dir / SCRIPT_MAP[task]
        cmd = _build_command(args.python, script_file, passthrough)
        exit_code = _run_command(cmd, dry_run=args.dry_run)
        if exit_code != 0:
            print(f"[run_stage1] Task {task} failed with exit code {exit_code}.")
            return exit_code

    print("[run_stage1] Completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
