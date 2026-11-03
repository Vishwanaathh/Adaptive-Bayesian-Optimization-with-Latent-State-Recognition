# Method specification

## Optimization target

For each family, optimize validation R² over normalized hyperparameter coordinates. Ridge and Lasso use one log-scaled alpha coordinate; ElasticNet uses log-scaled alpha and a continuous L1 ratio. Features are standardized by a pipeline whose scaler is fitted only on the training partition.

## Prequential innovation

At iteration t, the GP posterior prediction at the selected candidate is recorded before objective evaluation. After observing validation R², compute

`z_t = (y_t - mu_(t-1)(lambda_t)) / max(sigma_(t-1)(lambda_t), epsilon)`.

The test partition is never evaluated by the optimizer. Final test metrics are computed after selecting the best validation configuration and refitting on train plus validation.

## GP and acquisition methods

The surrogate is an exact ARD-RBF GP with Gaussian observation noise and marginal-likelihood hyperparameter fitting. Candidate maximization uses a scrambled Sobol pool. UCB, EI, scheduled UCB, and a local UCB proposal are implemented. Random search samples the normalized domain directly and does not fit a GP.

## GP-Hedge

Three experts propose UCB, EI, and local-UCB points. Expert probabilities are proportional to the softmax of cumulative gains. After the selected expert's candidate is evaluated, the GP is refit including the new observation. The posterior means at all three expert proposal points under this updated posterior are added to the corresponding cumulative gains. This ordering is intentional and tested.

## HMM and state policy

The HMM has three scalar Gaussian emission states and log-space forward filtering. The default initial standard deviations are ordered Explore > Exploit > Refine. Periodic forward-backward batch EM updates parameters using all accumulated innovations. After EM, the model replays the sequence to reconstruct the terminal filtering belief. States are sorted by descending emission variance after updates to preserve the declared labels. This is an identifying constraint, not evidence of state semantics.

Hard control uses the maximum-probability state. Soft control normalizes the three expert scores over the candidate pool and combines them using the current state belief. Explore uses UCB, Exploit uses EI, and Refine uses negative distance from the incumbent (with a local candidate pool in hard mode).

## Statistical protocol

Each seed creates one train/validation/test split. All methods and families for that seed use the same split; within each family and seed, all methods receive the same initial normalized design and evaluation budget. Aggregate metrics include mean, sample standard deviation, and percentile-bootstrap 95% intervals over seeds. Paired two-sided Wilcoxon signed-rank tests compare each method against ABO-LSR soft within family, with Holm adjustment within each family.

## Ablations and interpretation

`abo_no_hmm` fixes the belief to the exploit expert; `abo_fixed_hmm` filters without EM updates; hard and soft variants compare state-conditioned policy forms. Residual-only emissions may be uninformative under a well-calibrated GP. State labels and phase interpretations require empirical diagnostics and should not be asserted from implementation alone.

## Scope

The implemented experiment uses California Housing only. It is a real-data regression study, not a synthetic benchmark suite and not evidence of general performance across unrelated tasks.
