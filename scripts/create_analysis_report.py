import pandas as pd
import json
import os
from pathlib import Path

root = Path("results/california_housing")
out = root / "ALL_RESULTS_FOR_ANALYSIS.txt"
MAX_SIZE = 49 * 1024 * 1024

sections = []

def add(title, content):
    sections.append(f"\n{'='*90}\n{title}\n{'='*90}\n{content}\n")

# Experiment metadata and configuration
for p in sorted(root.rglob("*.json")):
    if p.name.lower() not in ("latest.json",):
        try:
            add(f"JSON: {p.relative_to(root)}", json.dumps(json.loads(p.read_text()), indent=2))
        except Exception:
            pass

# Main metrics and statistical tests
for name in ["per_seed_metrics.csv", "paired_statistical_tests.csv",
             "summary_metrics.csv", "aggregate_metrics.csv"]:
    p = root / name
    if p.exists():
        df = pd.read_csv(p)
        add(name, df.to_string(index=False))

# Compact convergence traces: retain every method's first, last,
# best-so-far and periodic observations.
p = root / "optimization_traces.csv"
if p.exists():
    chunks = []
    for chunk in pd.read_csv(p, chunksize=100000):
        chunks.append(chunk)
    traces = pd.concat(chunks, ignore_index=True)
    add("OPTIMIZATION TRACE COLUMNS", str(list(traces.columns)))

    group_cols = [c for c in ["seed", "family", "method"] if c in traces.columns]
    step_col = next((c for c in ["evaluation", "iteration", "step", "evaluation_number"] if c in traces.columns), None)

    if group_cols and step_col:
        traces = traces.sort_values(group_cols + [step_col])
        sampled = []
        for _, g in traces.groupby(group_cols, sort=False):
            selected = pd.concat([
                g.head(3),
                g.iloc[::5],
                g.tail(3)
            ]).drop_duplicates()
            sampled.append(selected)
        compact = pd.concat(sampled, ignore_index=True)
        add("SAMPLED OPTIMIZATION TRACES", compact.to_string(index=False))

# HMM diagnostics and ablations
for p in sorted(root.rglob("*")):
    if p.is_file() and p.suffix.lower() in [".txt", ".log", ".yaml", ".yml"]:
        try:
            add(str(p.relative_to(root)), p.read_text(errors="replace")[:200000])
        except Exception:
            pass

content = "\n".join(sections)
if len(content.encode("utf-8")) > MAX_SIZE:
    content = content.encode("utf-8")[:MAX_SIZE].decode("utf-8", errors="ignore")

out.write_text(content, encoding="utf-8")
print(f"Created: {out.resolve()}")
print(f"Size: {out.stat().st_size / 1024 / 1024:.2f} MB")
