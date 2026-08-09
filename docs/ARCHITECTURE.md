# Architecture

RetinaTriage separates data preparation, model lifecycle, inference, clinical routing, persistence and presentation.

The Flask application factory loads one `Predictor` per process. It validates model class mapping and dimensions before readiness. A lock protects prediction calls when the runtime requires it. Routes never contain priority SQL or model architecture logic.

The canonical image path is Pillow decode and EXIF transpose, RGB conversion, retinal-field crop, square padding, resize, optional conservative luminance CLAHE, then the representation expected by Keras EfficientNet. The unmodified RGB display copy is distinct from model input.

The quality service runs before disease presentation. It combines resolution, blur, brightness, contrast, exposure, retinal-field coverage and circular-field evidence into a composite score. The configured rule is explicit: scores of 75/100 or above pass the software quality gate; scores below 75 withhold the disease grade and produce retake/manual-review routing. Individual heuristic observations remain visible but do not independently veto an image above the composite threshold. The 75/100 threshold is not clinically validated, and passing it does not guarantee clinical gradability.

Triage combines grade, quality, model confidence, entropy, top-two margin and high-risk probability using configuration-controlled rules. Confidence is described as calibrated only when the installed calibration artifact has `status: validated`; the bundled external demonstration model is explicitly marked uncalibrated.

SQLite stores anonymous audit metadata, not original pixels. Reports are generated on demand. Evaluation content is read only from generated artifacts.

## Trust boundaries

Uploads are untrusted. Extension, size, decoded format, dimensions and decompression limits are enforced before inference. Client filenames never determine server paths. The local database and model artifacts are trusted deployment inputs and require controlled installation.

## Scaling

For multi-worker deployment, each process holds its own model. SQLite is suitable for a demonstration or low-write single-host workflow; a governed production study should use authenticated access, centralized audit storage, encryption, monitoring and formal retention controls.
