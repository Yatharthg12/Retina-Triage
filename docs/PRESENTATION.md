# RetinaTriage professional presentation runbook

## What this presentation proves

The demonstration proves that the software can validate a retinal upload, apply
an interpretable image-quality gate, run a real five-class model, quantify
uncertainty, assign a deterministic review priority, generate a Grad-CAM
influence map, persist an anonymous audit record, and create a PDF report.

It does not prove clinical safety, diagnostic accuracy in a new population, or
regulatory readiness.

## One-time preparation

From the repository root:

```powershell
python -m pip install -r requirements-dev.txt
python scripts\setup_presentation.py
python scripts\verify_presentation.py
```

The setup script downloads the model from a pinned Hugging Face revision,
verifies its SHA-256 checksum, checks Keras compatibility, verifies the sample
pack, and initializes SQLite.

The verification script exercises:

- model checksum and class mapping;
- sample checksums;
- a real CPU inference;
- raw-softmax confidence labeling;
- Grad-CAM generation;
- all main pages and status endpoints;
- multipart HTTP upload;
- SQLite audit persistence;
- PDF report generation and deletion.

Do not present if `PRESENTATION READY` is not printed.

## Start the presentation

Either run:

```powershell
.\scripts\launch_presentation.ps1
```

or:

```powershell
python scripts\run_presentation.py
```

Open `http://127.0.0.1:5000`. Keep the terminal visible until the application
reports that it is serving. Use a current Chrome, Edge, or Firefox window at
100% zoom.

## Recommended five-minute sequence

### 1. Command centre

Say:

> RetinaTriage is an explainable research workflow for quality-gated diabetic
> retinopathy screening and human-review prioritization. It does not replace an
> ophthalmologist.

Point out the model-ready badge, recent audit queue, review counts, and research
disclaimer.

### 2. Complete live inference

Open **Analyze retina**. Under **Presentation samples**, select **CDC diabetic
retinopathy**, then select **Run quality-gated analysis**.

Explain:

- the file is a bundled public-domain, de-identified image;
- the composite quality gate passes at 75/100 or above and fails below 75;
- all five probabilities are shown;
- confidence is explicitly marked as raw softmax and uncalibrated;
- uncertainty can force manual review;
- Grad-CAM shows model influence, not a lesion boundary or diagnosis.

Do not promise a specific grade. The source describes diabetic retinopathy, but
the model output is an investigational estimate.

### 3. Quality-threshold policy

Point to the quality card and the visible 75/100 disclaimer. If demonstrating a
quality rejection, use a deliberately dark, low-contrast or strongly blurred
test image whose composite score is below 75. The system withholds the disease
grade and routes it to **RETAKE / MANUAL REVIEW**.

Say:

> The system fails closed: insufficient image quality suppresses the public
> disease result instead of presenting a confident-looking answer.

State explicitly that 75 is a configurable software threshold, not a clinically
validated cutoff, and that passing it does not guarantee clinical gradability.

### 4. Batch workflow

Open **Batch screening** and select all three files from `demo/samples/`.
Run the batch and show that cases are sorted by review priority and can be
exported as CSV.

### 5. Audit and report

Open **Screening history**, select the completed CDC case, and download its PDF.
Explain that original pixels are not persisted; the SQLite audit contains
anonymous metadata, probabilities, routing, model version, and processing time.

### 6. Evidence boundary

Open **Model intelligence**. Clearly distinguish:

- published model-author validation: accuracy 0.72, macro F1 0.57, weighted F1
  0.73 on 550 held-out APTOS images;
- local application evaluation: not yet independently evaluated on a locked
  test set;
- confidence calibration: unavailable for this external artifact.

Finish on **Responsible use**.

## Presentation assets and provenance

- Model: `Aldahmashi/DR-EfficientNetB0`
- Revision: `fb8d14c59bd56aa17fe0dfdea04a83ecd2f2eeac`
- Model license: MIT
- Artifact SHA-256:
  `e7aa6b69911a2a913a03a6a5669bb7aaeb6bf8f8c81a2a3d92aa2420e5d297d8`
- Sample attribution: [`demo/ATTRIBUTION.md`](../demo/ATTRIBUTION.md)

## Recovery checklist

If the model badge is unavailable:

```powershell
python scripts\doctor.py
python scripts\setup_presentation.py
python scripts\verify_presentation.py
```

If port 5000 is occupied:

```powershell
$env:PORT = "5050"
python scripts\run_presentation.py
```

Then open `http://127.0.0.1:5050`.

If TensorFlow cannot import, create a clean Python 3.13 virtual environment and
reinstall `requirements-dev.txt`.

If a selected image scores below 75, present the manual-review route as the
configured quality-safety behavior. Never modify thresholds during a live
presentation.

## Required language

Use the phrases **research screening output**, **model estimate**, **manual
review**, and **published baseline**.

Avoid **diagnosis**, **clinically accurate**, **perfect result**, **validated
medical device**, or any claim that a low-risk result excludes disease.
