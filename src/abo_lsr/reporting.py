from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

def export_run(records, output_dir):
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(records)
    df.to_csv(out/"evaluations.csv",index=False)
    if len(df) and "score" in df:
        df["best_so_far"]=df["score"].cummax()
        fig,ax=plt.subplots(figsize=(8,4.5))
        ax.plot(range(1,len(df)+1),df["best_so_far"],label="best so far")
        ax.set(xlabel="Evaluation",ylabel="Best objective (maximize)",title="Optimization trace")
        ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
        fig.savefig(out/"optimization_trace.png",dpi=160); plt.close(fig)
    return df
