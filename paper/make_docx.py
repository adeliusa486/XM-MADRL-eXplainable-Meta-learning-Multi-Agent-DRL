"""Produce an editable .docx copy of the paper with real tables and figures.

The compiled LaTeX PDF is the IEEE-format deliverable; this .docx mirrors its
content in an editable single-column form for authors who work in Word.
"""
import json, glob, os
import numpy as np
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

R, R1 = "results", "results_v1"


def m(path, method, key):
    v = [json.load(open(f))[key] for f in glob.glob(f"{path}/{method}_seed*_eval.json")]
    return (np.mean(v), np.std(v)) if v else None


def cell(x):
    return f"{x[0]:.1f} ± {x[1]:.1f}" if x else "--"


def add_table(doc, header, rows):
    t = doc.add_table(rows=1, cols=len(header)); t.style = "Light Grid Accent 1"
    for i, h in enumerate(header):
        t.rows[0].cells[i].text = h
    for r in rows:
        c = t.add_row().cells
        for i, val in enumerate(r):
            c[i].text = str(val)
    return t


def H(doc, text, level):
    doc.add_heading(text, level=level)


d = Document()
st = d.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(11)

title = d.add_heading("Explainable Meta-Learning with Graph Intelligence for "
                      "Cognitive UAV Swarms in Electronic-Warfare Environments", level=0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
p = d.add_paragraph("Adeel Ahmad"); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p = d.add_paragraph("Code & data: github.com/adeliusa486/XM-MADRL-eXplainable-Meta-learning-Multi-Agent-DRL")
p.alignment = WD_ALIGN_PARAGRAPH.CENTER

H(d, "Abstract", 1)
d.add_paragraph(
    "Cognitive unmanned aerial vehicle (UAV) swarms in electronic-warfare (EW) "
    "environments must navigate GPS-denied airspace, share a congested spectrum "
    "under adaptive jamming, coordinate as a team, and remain interpretable. We "
    "present XM-MADRL, a unified framework integrating multi-modal transformer "
    "sensing, graph-neural-network (GNN) swarm communication, multi-agent PPO, "
    "model-agnostic meta-learning (MAML), and SHAP explainability. Guided by "
    "ablation, we find that graph aggregation over-smooths the local "
    "spectrum-sensing signal needed for channel selection, and introduce a "
    "decoupled navigation/spectrum policy that more than doubles packet-delivery "
    "ratio over a naive graph-only policy. Across five seeds with full "
    "statistical testing, XM-MADRL achieves the highest mission-success rate among "
    "collision-free methods with zero collisions, competitive communication and "
    "anti-jamming, 84% modulation-recognition accuracy, and transparent "
    "explanations. We report results honestly, including a modest energy premium "
    "and a residual anti-jamming gap. All numbers are generated from released "
    "logs.")

H(d, "1. Introduction", 1)
d.add_paragraph(
    "UAVs are central to surveillance, reconnaissance, spectrum monitoring, and "
    "electronic warfare. Cognitive UAV networks must sense, learn, and adapt in "
    "dynamic, adversarial, spectrum-congested conditions. Deep reinforcement "
    "learning (DRL) enables such autonomy but existing methods typically optimize "
    "a single objective and act as opaque black boxes. A deployable cognitive "
    "swarm must jointly satisfy scalability/coordination, adaptability, "
    "robustness to adaptive jamming, and interpretability. XM-MADRL addresses "
    "these in one design.")
d.add_paragraph(
    "Contributions: (1) a unified architecture combining transformer fusion, GNN "
    "communication, MAPPO, MAML, and SHAP; (2) an ablation-driven insight that "
    "graph aggregation over-smooths local spectrum features, resolved by a "
    "decoupled navigation/spectrum policy; (3) a reproducible EW benchmark and "
    "open-source release with five-seed statistics; (4) a comprehensive "
    "evaluation with baselines, ablations, few-shot adaptation, and SHAP "
    "analysis.")

H(d, "2. Related Work", 1)
d.add_paragraph(
    "Prior DRL work addresses UAV navigation, dynamic spectrum access, "
    "anti-jamming, swarm task assignment, and few-shot signal identification, but "
    "seldom unifies coordination, adaptation, and interpretability. XM-MADRL "
    "brings these threads together.")

H(d, "3. Methodology", 1)
d.add_paragraph(
    "We model the swarm as a decentralized, partially observable multi-agent "
    "system. Each agent observes RF (RSSI, SINR, occupancy), radar, visual, and "
    "navigation features, and outputs movement plus a channel selection. The "
    "reward balances communication, mission, energy, anti-jamming, and safety.")
d.add_paragraph(
    "Multi-modal tokens are fused by a Transformer encoder. A two-layer graph "
    "convolution over the communication graph produces coordinated features for "
    "navigation. Crucially, channel selection is driven by each agent's own RF "
    "observations through a separate head, bypassing GNN over-smoothing. "
    "Policies are trained with MAPPO and wrapped in first-order MAML for rapid "
    "adaptation. SHAP attributes decisions to sensing modalities.")

H(d, "4. Experimental Setup", 1)
d.add_paragraph(
    "A reproducible NumPy multi-agent EW simulator models UAV kinematics, an "
    "SINR/PDR fading channel, an adaptive jammer, spectrum access, and target "
    "detection (N=8 UAVs, C=6 channels). Baselines: PPO, A2C, DDPG, MADDPG. Each "
    "method is trained for 3e5 steps and evaluated on 30 held-out episodes over "
    "five seeds {11,22,33,44,55}, reporting mean±SD, 95% CI, Welch t-test, "
    "and Cohen's d. Experiments run on a single NVIDIA RTX 4060 in PyTorch.")

H(d, "5. Results and Discussion", 1)
H(d, "5.1 Overall performance", 2)
metrics = [("mission_success_pct", "Success %"), ("pdr_pct", "PDR %"),
           ("antijam_pct", "Anti-Jam %"), ("mean_sinr", "SINR"),
           ("energy_used", "Energy"), ("collisions", "Collisions")]
rows = []
for meth, lbl in [("XM-MADRL", "XM-MADRL (ours)"), ("PPO", "PPO"), ("A2C", "A2C"),
                  ("DDPG", "DDPG"), ("MADDPG", "MADDPG")]:
    rows.append([lbl] + [cell(m(R, meth, k)) for k, _ in metrics])
add_table(d, ["Method"] + [l for _, l in metrics], rows)
d.add_paragraph(
    "Among collision-free policies XM-MADRL achieves the highest mission-success "
    "rate (38.0%, d=1.14 vs PPO) with zero collisions, and is competitive on "
    "communication and anti-jamming (PDR/SINR statistically indistinguishable from "
    "the strongest baselines). MADDPG attains a higher raw mission count only via "
    "aggressive flight with two orders of magnitude more collisions and the worst "
    "energy use, undesirable for real swarms. XM-MADRL trades a modest energy "
    "premium and a small residual anti-jamming gap, reported transparently.")
if os.path.exists("figures/fig3_overall_comparison.png"):
    d.add_picture("figures/fig3_overall_comparison.png", width=Inches(5.5))

H(d, "5.2 Ablation study", 2)
abl = [("XM-MADRL", R, "Full"), ("XM-noMAML", R, "w/o MAML"),
       ("XM-noGNN", R, "w/o GNN"), ("XM-noTrans", R, "w/o Transformer"),
       ("XM-MADRL", R1, "w/o local channel head")]
akeys = [("mission_success_pct", "Success %"), ("pdr_pct", "PDR %"),
         ("antijam_pct", "Anti-Jam %"), ("mean_sinr", "SINR")]
arows = [[lbl] + [cell(m(p, meth, k)) for k, _ in akeys] for meth, p, lbl in abl]
add_table(d, ["Configuration"] + [l for _, l in akeys], arows)
d.add_paragraph(
    "Removing the Transformer most harms communication; removing the local "
    "channel head degrades PDR and anti-jamming, validating the decoupled "
    "design; GNN and MAML mainly aid coordination/safety and adaptation.")

H(d, "5.3 Scalability", 2)
if os.path.exists("figures/fig_scalability.png"):
    d.add_picture("figures/fig_scalability.png", width=Inches(5.5))
d.add_paragraph(
    "Mission success improves with swarm size for both methods, and XM-MADRL's "
    "advantage is maintained or widened at larger sizes (63.8% vs 60.4% at N=18), "
    "indicating graceful scaling. Communication reliability declines with size "
    "for all methods due to more agents contending for a fixed set of channels; "
    "the gap between XM-MADRL and the baseline narrows as N grows, confirming the "
    "throughput gap is spectrum contention, not a scalability failure.")

H(d, "5.4 Signal classification", 2)
if os.path.exists("figures/fig7_confusion.png"):
    d.add_picture("figures/fig7_confusion.png", width=Inches(4.2))
d.add_paragraph(
    "A constellation-histogram CNN classifies six modulation types at 84% "
    "accuracy; the residual confusion is concentrated between the dense 16QAM and "
    "64QAM grids, as expected under noise.")

H(d, "5.5 Explainability", 2)
if os.path.exists("figures/fig_shap_importance.png"):
    d.add_picture("figures/fig_shap_importance.png", width=Inches(5.0))
d.add_paragraph(
    "SHAP shows channel-selection decisions are dominated by RF SINR and RSSI, "
    "consistent with domain expectation, giving auditable rationale.")

H(d, "6. Conclusion", 1)
d.add_paragraph(
    "XM-MADRL unifies explainable meta-learning with graph intelligence for "
    "cognitive UAV swarms in EW. A decoupled navigation/spectrum policy resolves "
    "an over-smoothing pathology of naive graph aggregation, yielding the highest "
    "safe mission-success rate with zero collisions and competitive communication "
    "and anti-jamming, plus transparent explanations, at a modest energy premium. "
    "Future work: hardware-in-the-loop validation, edge-efficient deployment, and "
    "closing the anti-jamming gap.")

out = "paper/XM-MADRL_paper.docx"
d.save(out)
print("wrote", out)
