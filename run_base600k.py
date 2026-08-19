import os, sys, subprocess, time
from concurrent.futures import ThreadPoolExecutor, as_completed
METHODS = ["PPO", "A2C", "DDPG", "MADDPG"]; SEEDS = [11, 22, 33, 44, 55]
STEPS = 600000; WORKERS = 5


def env():
    e = os.environ.copy()
    e.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="-1")
    return e


def job(m, s):
    with open(f"results/logs/{m}_600k_s{s}.log", "w") as f:
        rc = subprocess.run(
            [sys.executable, "train.py", "--method", m, "--seed", str(s),
             "--steps", str(STEPS), "--results", "results", "--device", "cpu"],
            stdout=f, stderr=subprocess.STDOUT, env=env()).returncode
    return f"{m}_s{s}", rc


jobs = [(m, s) for m in METHODS for s in SEEDS]
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    for i, fu in enumerate(as_completed([ex.submit(job, *j) for j in jobs]), 1):
        n, rc = fu.result()
        print(f"[{i}/20] {n}: {'ok' if rc == 0 else 'FAIL'}", flush=True)
print("BASE 600k DONE")
