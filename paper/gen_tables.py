"""Generate LaTeX tables for the paper from the real result JSONs.

Reads:
  results/        -> v2 proposed (XM-MADRL) + v1 baselines & ablations
  results_v1/     -> original single-head XM-MADRL (used as the -ChannelHead ablation)
Writes:
  paper/tables.tex
"""
import json, glob, os
import numpy as np

R = "results"
R1 = "results_v1"
OUT = "paper/tables.tex"

METRICS = [("mission_success_pct", "Mission Success (\\%)", "up"),
           ("pdr_pct", "PDR (\\%)", "up"),
           ("antijam_pct", "Anti-Jam (\\%)", "up"),
           ("mean_sinr", "SINR (dB)", "up"),
           ("energy_used", "Energy", "down"),
           ("collisions", "Collisions", "down")]


def load(path, method, key):
    vals = []
    for f in glob.glob(f"{path}/{method}_seed*_eval.json"):
        vals.append(json.load(open(f))[key])
    return np.array(vals) if vals else None


def ms(path, method, key):
    v = load(path, method, key)
    if v is None or len(v) == 0:
        return None
    return v.mean(), v.std(), len(v)


def fmt(x):
    return f"{x[0]:.1f}$\\pm${x[1]:.1f}" if x else "--"


def main_table():
    methods = [("XM-MADRL", R, "\\textbf{XM-MADRL (ours)}"),
               ("PPO", R, "PPO"), ("A2C", R, "A2C"),
               ("DDPG", R, "DDPG"), ("MADDPG", R, "MADDPG")]
    lines = [r"\begin{table*}[!t]", r"\centering",
             r"\caption{Performance comparison on the cognitive UAV-swarm EW benchmark "
             r"(mean $\pm$ standard deviation over 5 random seeds). Best value per column in \textbf{bold}.}",
             r"\label{tab:main}",
             r"\begin{tabular}{l" + "c" * len(METRICS) + "}", r"\hline",
             "Method & " + " & ".join(m[1] for m in METRICS) + r" \\ \hline"]
    # find best per metric
    best = {}
    for key, _, direction in METRICS:
        vals = {}
        for m, path, _ in methods:
            x = ms(path, m, key)
            if x:
                vals[m] = x[0]
        if vals:
            best[key] = (max if direction == "up" else min)(vals, key=vals.get)
    for m, path, label in methods:
        cells = [label]
        for key, _, _ in METRICS:
            x = ms(path, m, key)
            s = fmt(x)
            if x and best.get(key) == m:
                s = r"\textbf{" + s + "}"
            cells.append(s)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def ablation_table():
    variants = [("XM-MADRL", R, "XM-MADRL (full)"),
                ("XM-noMAML", R, "\\quad w/o MAML"),
                ("XM-noGNN", R, "\\quad w/o GNN comm."),
                ("XM-noTrans", R, "\\quad w/o Transformer"),
                ("XM-MADRL", R1, "\\quad w/o local channel head")]
    keys = [("mission_success_pct", "Success (\\%)"), ("pdr_pct", "PDR (\\%)"),
            ("antijam_pct", "Anti-Jam (\\%)"), ("mean_sinr", "SINR (dB)")]
    lines = [r"\begin{table}[!t]", r"\centering",
             r"\caption{Ablation study (mean $\pm$ SD over seeds). Each row removes one component.}",
             r"\label{tab:ablation}",
             r"\begin{tabular}{l" + "c" * len(keys) + "}", r"\hline",
             "Configuration & " + " & ".join(k[1] for k in keys) + r" \\ \hline"]
    for m, path, label in variants:
        cells = [label]
        for key, _ in keys:
            cells.append(fmt(ms(path, m, key)))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def stats_table():
    f = f"{R}/stats_table.json"
    if not os.path.exists(f):
        return "% stats table not available"
    tab = json.load(open(f))
    name = {"mission_success_pct": "Mission Success", "pdr_pct": "PDR",
            "antijam_pct": "Anti-Jam", "mean_sinr": "SINR", "energy_used": "Energy"}
    lines = [r"\begin{table}[!t]", r"\centering",
             r"\caption{Statistical comparison of XM-MADRL vs.\ the strongest baseline "
             r"(Welch's $t$-test, 95\% CI, Cohen's $d$).}", r"\label{tab:stats}",
             r"\begin{tabular}{lcccc}", r"\hline",
             r"Metric & Proposed & Baseline & $p$-value & Cohen's $d$ \\ \hline"]
    for k, s in tab.items():
        lines.append(f"{name.get(k,k)} & {s['proposed_mean']:.1f}$\\pm${s['proposed_sd']:.1f} & "
                     f"{s['baseline_mean']:.1f}$\\pm${s['baseline_sd']:.1f} & "
                     f"{s['p_value']:.1e} & {s['cohens_d']:.2f} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def signal_table():
    files = glob.glob(f"{R}/signal_maml_seed*.json")
    if not files:
        return "% signal-maml table not available"
    ma = np.array([json.load(open(f))["maml_acc"] for f in files])
    sc = np.array([json.load(open(f))["scratch_acc"] for f in files])
    ds = json.load(open(files[0]))["dataset"]
    return "\n".join([
        r"\begin{table}[!t]", r"\centering",
        r"\caption{Few-shot 5-way signal classification accuracy (\%%) on %s: MAML vs.\ training from scratch.}" % ds.replace("_","\\_"),
        r"\label{tab:signal}", r"\begin{tabular}{lc}", r"\hline",
        r"Method & Accuracy (\%) \\ \hline",
        f"MAML (ours) & \\textbf{{{ma.mean():.1f}$\\pm${ma.std():.1f}}} \\\\",
        f"From scratch & {sc.mean():.1f}$\\pm${sc.std():.1f} \\\\",
        r"\hline", r"\end{tabular}", r"\end{table}"])


if __name__ == "__main__":
    os.makedirs("paper", exist_ok=True)
    with open(OUT, "w") as f:
        f.write("% auto-generated from result JSONs -- do not edit by hand\n")
        f.write(main_table() + "\n\n")
        f.write(ablation_table() + "\n\n")
        f.write(stats_table() + "\n\n")
        f.write(signal_table() + "\n")
    print(f"wrote {OUT}")
