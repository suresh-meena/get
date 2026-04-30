import subprocess
import concurrent.futures
import os
import queue
import shlex
import signal
import threading
import atexit
from experiments.shared.common import set_seed

# Real-world only experiments
EXPERIMENTS = [
    "experiments/stage3/zinc_regression.py",
    "experiments/stage3/molhiv_classification.py",
    "experiments/stage3/peptides_transfer.py --task func",
    "experiments/stage3/peptides_transfer.py --task struct",
    "experiments/stage4/runner.py --task graph_classification --dataset MUTAG --epochs 30",
    "experiments/stage4/runner.py --task graph_anomaly --dataset YelpChi --ego_limit 1000 --epochs 30",
]

_CHILDREN: set[subprocess.Popen] = set()
_CHILDREN_LOCK = threading.Lock()


def _cleanup_children():
    with _CHILDREN_LOCK:
        children = list(_CHILDREN)
    for child in children:
        if child.poll() is not None:
            continue
        try:
            child.terminate()
        except Exception:
            continue
    for child in children:
        if child.poll() is not None:
            continue
        try:
            child.wait(timeout=5)
        except Exception:
            try:
                child.kill()
            except Exception:
                pass


def _signal_handler(_signum, _frame):
    _cleanup_children()
    raise SystemExit(1)


atexit.register(_cleanup_children)
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def run_experiment(script_args, gpu_id):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["EXPERIMENT_OUTPUT_SUFFIX"] = f"_gpu{gpu_id}"
    env["PYTHONPATH"] = "."
    os.makedirs("outputs", exist_ok=True)

    cmd = ["conda", "run", "--no-capture-output", "-n", "get", "python", *shlex.split(script_args)]
    print(f"Starting: {script_args} on GPU {gpu_id}")

    try:
        log_file = f"outputs/gpu{gpu_id}.log"
        with open(log_file, "a", encoding="utf-8") as log:
            log.write(f"\n--- Starting {script_args} ---\n")
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            with _CHILDREN_LOCK:
                _CHILDREN.add(proc)
            code = proc.wait()
            with _CHILDREN_LOCK:
                _CHILDREN.discard(proc)

        if code == 0:
            print(f"Finished: {script_args} on GPU {gpu_id}")
        else:
            print(f"Error in: {script_args} on GPU {gpu_id} (check {log_file})")
    except Exception as e:
        print(f"Exception running {script_args}: {e}")

def main():
    set_seed(42)
    
    # We have 2 GPUs: 0 and 1
    gpus = queue.Queue()
    gpus.put(0)
    gpus.put(1)
    
    def worker(exp):
        gpu_id = gpus.get()
        try:
            run_experiment(exp, gpu_id)
        finally:
            gpus.put(gpu_id)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(worker, exp) for exp in EXPERIMENTS]
        concurrent.futures.wait(futures)

if __name__ == "__main__":
    main()
