# Validation protocol

Use the train split for weight fitting and class weights, validation split for early stopping, model selection, calibration and threshold selection, and the untouched test split once for final evidence.

Primary selection measure is quadratic weighted Cohen’s kappa because grading is ordered. Final reporting also includes accuracy, macro and weighted F1, per-grade precision/recall/F1/sensitivity/specificity, raw and normalized confusion matrices, one-vs-rest ROC/PR, mean absolute grade error, adjacent errors, severe under-classification, referable/high-risk metrics, calibration error and latency distribution.

A dangerous under-classification includes true grade 3 or 4 predicted as grade 0 or 1. Every such example requires explicit review.

Thresholds must be selected on validation data. If target sensitivity is used, save the target, achieved sensitivity, specificity and selected threshold. Never tune using the test set.

Confidence intervals should use reproducible patient-level bootstrap sampling when reliable grouping exists. If the test sample or positive-class count is insufficient, report that limitation instead of unstable intervals.

External validation must retain its own source identity and label mapping.

