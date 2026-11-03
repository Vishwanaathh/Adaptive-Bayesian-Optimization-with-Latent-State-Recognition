# Ablation definitions

| Variant | HMM filtering | HMM EM updates | State policy |
|---|---|---|---|
| `abo_no_hmm` | No | No | Fixed exploit belief (EI expert) |
| `abo_fixed_hmm` | Yes | No | Soft belief-weighted |
| `abo_lsr_hard` | Yes | Periodic | MAP state |
| `abo_lsr_soft` | Yes | Periodic | Belief-weighted mixture |

These variants isolate useful implementation components, but are not a complete factorial causal study. In particular, `abo_no_hmm` is an EI-equivalent fixed-policy reference rather than a distinct new optimizer.
