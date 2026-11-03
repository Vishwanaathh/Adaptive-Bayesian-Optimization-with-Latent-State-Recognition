```mermaid
flowchart TD
  E[Explore] -->|A_EE| E
  E -->|A_EX| X[Exploit]
  E -->|A_ER| R[Refine]
  X -->|A_XE| E
  X -->|A_XX| X
  X -->|A_XR| R
  R -->|A_RE| E
  R -->|A_RX| X
  R -->|A_RR| R
```

At each completed evaluation, the prequential GP residual updates the posterior state belief.
The transition matrix expresses persistence and switching probabilities; the diagram is not a
deterministic stage sequence.
