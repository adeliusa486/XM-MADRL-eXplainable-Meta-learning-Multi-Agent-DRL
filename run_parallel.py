"""Parallel experiment launcher.

The training bottleneck is the (single-threaded, CPU-bound) RL rollout, not GPU
compute -- the networks are tiny.  So the fastest way to run the full study on a
multi-core machine is to run many *single-threaded* jobs at once, one per core,
rather than one job that under-utilises the GPU.

    python run_parallel.py --workers 12          # full study, 5 seeds
    python run_parallel.py --workers 12 --seeds 11 22 33 44 55 66 77 88 99 100
    QUICK=1 python run_parallel.py --workers 4    # tiny sanity pass

Each worker runs single-threaded on CPU (OMP/MKL threads pinned to 1) so N
workers give ~N x throughput.  Finished runs (eval JSON present) are skipped, so
this is safe to stop and re-run.  After all training + signal jobs finish it runs
stats -> SHAP -> figures automatically.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PROPOSED = ["XM-MADRL"]
BASELINES = ["PPO", "A2C", "DDPG", "MADDPG"]
ABLATIONS = ["XM-noMAML", "XM-noGNN", "XM-noTrans", "XM-noXAI"]
ALL_METHODS = PROPOSED + BASELINES + ABLATIONS


def worker_env():
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "-1"        # force CPU; parallel CPU beats a shared GPU here
    return env


def run_job(cmd, logdir: Path, name: str):
    log = logdir / f"{name}.log"
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=worker_env()).returncode
    return name, rc, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55])
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    if os.environ.get("QUICK") == "1":
        args.seeds = [11]; args.steps = 4000
        print(">>> QUICK MODE")

    results = Path(args.results); results.mkdir(parents=True, exist_ok=True)
    logdir = results / "logs"; logdir.mkdir(exist_ok=True)
    py = sys.executable

    # ---- build job list (skip finished) ---------------------------------- #
    jobs = []
    for m in ALL_METHODS:
        for s in args.seeds:
            if (results / f"{m}_seed{s}_eval.json").exists():
                continue
            jobs.append((f"{m}_s{s}",
                         [py, "train.py", "--method", m, "--seed", str(s),
                          "--steps", str(args.steps), "--results", args.results,
                          "--device", "cpu"]))
    for s in args.seeds:
        if not (results / f"signal_maml_seed{s}.json").exists():
            jobs.append((f"signal_s{s}",
                         [py, "signal_maml.py", "--seed", str(s), "--results", args.results]))

    print(f">>> {len(jobs)} jobs, {args.workers} parallel workers, {args.steps} steps/run")
    if not jobs:
        print(">>> nothing to do (all finished)")
    else:
        t0 = time.time()
        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_job, cmd, logdir, name): name for name, cmd in jobs}
            for fut in as_completed(futs):
                name, rc, dt = fut.result()
                done += 1
                status = "ok" if rc == 0 else f"FAIL(rc={rc})"
                print(f"[{done}/{len(jobs)}] {name}: {status} ({dt/60:.1f} min, "
                      f"elapsed {(time.time()-t0)/60:.1f} min)", flush=True)
        print(f">>> all jobs finished in {(time.time()-t0)/60:.1f} min")

    # ---- analysis (sequential) ------------------------------------------- #
    print(">>> stats / SHAP / figures ...")
    subprocess.run([py, "stats.py", "--results", args.results, "--baseline", "PPO"], env=worker_env())
    w = results / f"XM-MADRL_seed{args.seeds[0]}.pt"
    if w.exists():
        subprocess.run([py, "run_shap.py", "--weights", str(w),
                        "--seed", str(args.seeds[0]), "--results", args.results], env=worker_env())
    subprocess.run([py, "make_figures.py", "--results", args.results, "--out", "figures"], env=worker_env())
    print(">>> DONE. See results/ and figures/.")


if __name__ == "__main__":
    main()
