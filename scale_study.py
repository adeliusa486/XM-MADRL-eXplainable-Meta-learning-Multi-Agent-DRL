"""Scalability study: XM-MADRL vs PPO across swarm sizes.

Runs at most 4 single-threaded workers at once (safe on a laptop). Produces a
scalability figure (mission success + PDR vs number of agents).
"""
import os, sys, glob, json, subprocess, time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

AGENT_COUNTS = [6, 12, 18]
METHODS = ["XM-MADRL", "PPO"]
SEEDS = [11, 22]
STEPS = 300_000
WORKERS = 4


def env():
    e = os.environ.copy()
    e.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="-1")
    return e


def job(method, seed, n):
    rd = f"results_scale/n{n}"
    os.makedirs(f"{rd}/logs", exist_ok=True)
    if os.path.exists(f"{rd}/{method}_seed{seed}_eval.json"):
        return f"{method}_n{n}_s{seed}", 0, 0.0
    t0 = time.time()
    with open(f"{rd}/logs/{method}_s{seed}.log", "w") as f:
        rc = subprocess.run(
            [sys.executable, "train.py", "--method", method, "--seed", str(seed),
             "--steps", str(STEPS), "--n_agents", str(n), "--results", rd, "--device", "cpu"],
            stdout=f, stderr=subprocess.STDOUT, env=env()).returncode
    return f"{method}_n{n}_s{seed}", rc, time.time() - t0


def main():
    jobs = [(m, s, n) for n in AGENT_COUNTS for m in METHODS for s in SEEDS]
    print(f">>> scalability: {len(jobs)} jobs, {WORKERS} workers")
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(job, *j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            name, rc, dt = f.result()
            print(f"[{i}/{len(jobs)}] {name}: {'ok' if rc==0 else 'FAIL'} ({dt/60:.1f}m)", flush=True)

    # aggregate + plot
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    C = {"XM-MADRL": "#4C72B0", "PPO": "#DD8452"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4))
    for m in METHODS:
        xs, succ, pdr = [], [], []
        for n in AGENT_COUNTS:
            fs = glob.glob(f"results_scale/n{n}/{m}_seed*_eval.json")
            if not fs:
                continue
            xs.append(n)
            succ.append(np.mean([json.load(open(f))["mission_success_pct"] for f in fs]))
            pdr.append(np.mean([json.load(open(f))["pdr_pct"] for f in fs]))
        a1.plot(xs, succ, "o-", label=m, color=C[m])
        a2.plot(xs, pdr, "o-", label=m, color=C[m])
    a1.set_xlabel("Number of UAVs"); a1.set_ylabel("Mission success (%)"); a1.set_title("Scalability: mission success"); a1.legend(); a1.grid(alpha=.3)
    a2.set_xlabel("Number of UAVs"); a2.set_ylabel("PDR (%)"); a2.set_title("Scalability: communication"); a2.legend(); a2.grid(alpha=.3)
    os.makedirs("figures", exist_ok=True)
    fig.tight_layout(); fig.savefig("figures/fig_scalability.png", dpi=150)
    fig.savefig("paper/figures/fig_scalability.png", dpi=150)
    print(">>> wrote figures/fig_scalability.png")


if __name__ == "__main__":
    main()
