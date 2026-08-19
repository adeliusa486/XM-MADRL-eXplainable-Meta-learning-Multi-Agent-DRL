#!/usr/bin/env bash
# Run once the v2 proposed-method seeds are complete. Produces the final
# statistics, SHAP analysis, figures, tables, and the compiled IEEE paper.
set -e
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1

echo ">>> [1/6] statistics"
python stats.py --results results --baseline PPO || true

echo ">>> [2/6] SHAP explainability"
W=$(ls results/XM-MADRL_seed*.pt 2>/dev/null | head -1)
[ -n "$W" ] && python run_shap.py --weights "$W" --seed 11 --results results || true

echo ">>> [3/6] professional figures (data, architecture, env, confusion)"
mkdir -p paper/figures
PYTHONPATH=paper python paper/pro_figures.py || true
PYTHONPATH=paper python paper/arch_diagram.py || true
python paper/env_figures.py --method XM-MADRL --compare MADDPG || true
python paper/confusion_fig.py || true

echo ">>> [4/6] (figures written to paper/figures)"

echo ">>> [5/6] LaTeX tables"
python paper/gen_tables.py

echo ">>> [6/6] compile paper (3 passes + bibtex)"
cd paper
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1 || true
bibtex main > /dev/null 2>&1 || true
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1 || true
pdflatex -interaction=nonstopmode main.tex > compile_final.log 2>&1 || true
cd ..
echo ">>> [7/7] editable .docx copy"
python paper/make_docx.py || true
echo ">>> DONE. Paper: paper/main.pdf + paper/XM-MADRL_paper.docx"
ls -la paper/main.pdf 2>/dev/null | awk '{print "PDF size:", $5}'
