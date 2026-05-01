import subprocess
import os
import time

def run_cmd(cmd, gpu_id, log_file):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONPATH"] = "."
    print(f"Starting: {cmd} on GPU {gpu_id}")
    with open(log_file, "a") as f:
        return subprocess.Popen(cmd, shell=True, env=env, stdout=f, stderr=subprocess.STDOUT)

def main():
    # GPU 0 queue
    q0 = [
        "conda run --no-capture-output -n get python experiments/stage1/wedge_discrimination.py --num_pairs 2000 --epochs 50",
        "conda run --no-capture-output -n get python experiments/stage3/zinc_regression.py",
    ]
    
    # GPU 1 queue
    q1 = [
        "conda run --no-capture-output -n get python experiments/stage3/molhiv_classification.py",
    ]
    
    # Simple sequential runner for each GPU
    p0 = None
    p1 = None
    
    while q0 or q1 or p0 or p1:
        # Check GPU 0
        if p0 is None and q0:
            cmd = q0.pop(0)
            p0 = run_cmd(cmd, 0, "outputs/gpu0.log")
        elif p0 is not None:
            if p0.poll() is not None:
                print(f"GPU 0 task finished with code {p0.returncode}")
                p0 = None
        
        # Check GPU 1
        if p1 is None and q1:
            cmd = q1.pop(0)
            p1 = run_cmd(cmd, 1, "outputs/gpu1.log")
        elif p1 is not None:
            if p1.poll() is not None:
                print(f"GPU 1 task finished with code {p1.returncode}")
                p1 = None
                
        time.sleep(5)

    print("All runs finished. Generating plots...")
    subprocess.run("find outputs -name '*.json' -not -path '*/archive/*' -not -path '*/runs/*' -exec python experiments/plot.py {} +", shell=True, env=os.environ)
    print("Plots generated.")

if __name__ == "__main__":
    main()
