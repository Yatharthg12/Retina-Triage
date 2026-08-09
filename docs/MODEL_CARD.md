# Model card — RetinaTriage presentation baseline

## Status

The presentation configuration uses the MIT-licensed
[`Aldahmashi/DR-EfficientNetB0`](https://huggingface.co/Aldahmashi/DR-EfficientNetB0)
artifact at revision `fb8d14c59bd56aa17fe0dfdea04a83ecd2f2eeac`.
The artifact is installed by `scripts/setup_presentation.py` and verified by
SHA-256 before loading.

This is a software-demonstration baseline, not a clinically validated model or
certified medical device.

## Intended use

Reproducible demonstrations and research education involving:

- retinal image validation and quality gating;
- five-grade diabetic-retinopathy model inference;
- uncertainty-aware review prioritization;
- Grad-CAM model-influence visualization;
- anonymous audit and reporting workflows.

A qualified eye-care professional must review the original image and all
relevant clinical evidence.

## Prohibited use

Autonomous diagnosis, treatment recommendations, emergency triage, exclusion
of disease, unsupervised patient communication, or representation as a
certified medical device.

## Architecture

The installed artifact contains model-side rescaling and normalization,
EfficientNetB0, global average pooling, dropout, and a five-unit softmax output.
Its input is 224×224 RGB and its output classes map to grades 0–4.

Referable DR is derived as grades 2–4. High-risk DR is derived as grades 3–4.
Those derived probabilities and the deterministic queue rules have not been
clinically validated.

## Provenance

- Author: Nasser Aldahmashi
- Source dataset: APTOS 2019 Blindness Detection
- Reported training scope: 3,662 fundus images
- Model license: MIT
- Artifact size: 33,436,274 bytes
- SHA-256:
  `e7aa6b69911a2a913a03a6a5669bb7aaeb6bf8f8c81a2a3d92aa2420e5d297d8`
- Required Keras version: 3.13.2

## Published evidence

The model author reports evaluation on 550 held-out APTOS images:

| Measure | Published value |
|---|---:|
| Accuracy | 0.72 |
| Macro F1 | 0.57 |
| Weighted F1 | 0.73 |

Per-class F1 reported by the author is 0.95 for no DR, 0.43 for mild, 0.57 for
moderate, 0.42 for severe, and 0.47 for proliferative DR.

These are model-author figures. They are not an independent evaluation of the
RetinaTriage preprocessing, quality gate, routing, or target presentation
environment. The application therefore labels this evidence **published
baseline**, not local test-set validation.

## Calibration

Compatible validation logits were not distributed with the external artifact.
The installed `calibration.json` consequently uses identity temperature 1.0 and
has status `not_calibrated`. Displayed confidence is raw softmax confidence.

The confidence, referable probability, high-risk probability, and configured
review thresholds must not be described as clinically calibrated.

## Limitations

- Minority-class performance is materially weaker than no-DR performance.
- Adjacent grades are frequently confused.
- The training data are imbalanced and come from one competition dataset.
- Generalization to other cameras, populations, acquisition protocols, and
  non-retinal images is untested.
- The composite quality gate passes scores of 75/100 or above and requires retake/manual review below 75. This cutoff is interpretable but not clinically validated.
- Grad-CAM is not lesion segmentation, clinical annotation, or causal proof.
- A passing quality gate does not guarantee clinical gradability.
- A low-grade output does not exclude disease.

## Evidence required before study or clinical use

A locked artifact, patient-aware split where legitimate identifiers exist,
independent calibration, untouched-test evaluation, external multi-centre
validation, subgroup analysis, dangerous under-classification review,
prospective workflow validation, cybersecurity assessment, privacy review, and
clinical/regulatory governance approval.
