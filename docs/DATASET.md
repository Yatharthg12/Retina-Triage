# Dataset and splitting

APTOS 2019 is the primary adapter. It expects `id_code,diagnosis` metadata and grade 0–4 PNG images. Dataset files are never committed.

Preparation verifies required columns, resolves every path, decodes each file, calculates SHA-256, removes exact duplicate hashes, records rejected rows and creates deterministic 70/15/15 train/validation/test manifests. Stratification falls back safely when class counts are too small. Duplicate hashes are asserted not to cross splits.

The untouched test split is reserved for final evaluation. Class weights use training rows only. Calibration, thresholds, early stopping and selection use validation data only.

EyePACS or Messidor-2 must be introduced through a separately documented adapter and mapping. Labels from distinct grading protocols must never be silently combined. Record dataset source, mapping, license/terms, acquisition population and fingerprint.

Patient-group splitting is preferred whenever reliable patient identity is legitimately available. The APTOS public metadata does not consistently provide a validated patient identifier, so the default adapter makes no unsafe patient-group claim.

