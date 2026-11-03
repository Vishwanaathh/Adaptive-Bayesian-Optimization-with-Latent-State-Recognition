# Experiment protocol

1. Download/cache California Housing using scikit-learn.
2. For each random seed, create one fixed train/validation/test split.
3. Share that split across all methods and regression families for the seed.
4. Standardize using training-only fit inside each estimator pipeline.
5. Optimize validation R² with the same number of objective evaluations and shared initial designs within each family/seed.
6. Never query the test set during optimization or model selection.
7. Refit the selected configuration on train + validation and evaluate test R², RMSE, and MAE.
8. Include OLS as a non-optimized reference.
9. Preserve per-seed metrics and per-evaluation traces.
10. Report mean, sample standard deviation, percentile-bootstrap 95% confidence intervals, and paired Wilcoxon tests with Holm adjustment.
11. Report runtime and objective evaluation counts separately from validation quality.
12. Inspect HMM transition matrices, emission means/variances, state occupancy and belief traces before assigning semantic meaning to states.
13. Run no-HMM, fixed-HMM, hard-policy and soft-policy ablations.
14. Treat small seed counts as exploratory; increase seeds and consider external datasets before publication claims.

No experimental result should be described as observed until the command has completed successfully and its output files have been inspected.
