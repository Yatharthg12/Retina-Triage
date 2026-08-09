# Third-party notices

This file records the provenance and licensing boundary for external artifacts referenced or distributed by RetinaTriage AI. The repository-level [MIT License](LICENSE) applies only to original RetinaTriage code and documentation unless a file says otherwise. It does not replace upstream terms.

This notice is informational and is not legal advice. Users and redistributors must verify the current upstream terms for their intended use.

## Demonstration model

### Aldahmashi/DR-EfficientNetB0

- Author: Nasser Aldahmashi
- Source: <https://huggingface.co/Aldahmashi/DR-EfficientNetB0>
- Pinned revision: `fb8d14c59bd56aa17fe0dfdea04a83ecd2f2eeac`
- Downloaded filename in this application: `artifacts/models/best_model.keras`
- SHA-256: `e7aa6b69911a2a913a03a6a5669bb7aaeb6bf8f8c81a2a3d92aa2420e5d297d8`
- License declared on the source model card: MIT
- Training data stated by the source: APTOS 2019 Blindness Detection

The binary is not distributed in the Git repository. `scripts/setup_presentation.py` downloads it from the pinned source revision and verifies its checksum. The source model card states that it is an ImageNet-pretrained EfficientNetB0 fine-tuned for five-class diabetic-retinopathy grading.

The upstream model card reports 72% accuracy, macro F1 0.57, and weighted F1 0.73 on 550 held-out APTOS images. These are the model author’s published figures, not an independent RetinaTriage evaluation and not evidence of clinical fitness.

## Dataset

### APTOS 2019 Blindness Detection

- Organizer: Asia Pacific Tele-Ophthalmology Society
- Host: Kaggle
- Competition: <https://www.kaggle.com/competitions/aptos2019-blindness-detection>
- Competition rules: <https://www.kaggle.com/competitions/aptos2019-blindness-detection/rules>
- APTOS challenge page: <https://2019.asiateleophth.org/big-data-competition/>
- Image provider acknowledged by APTOS: Aravind Eye Care System

Suggested attribution:

> Asia Pacific Tele-Ophthalmology Society. *APTOS 2019 Blindness Detection*. Kaggle, 2019. <https://www.kaggle.com/competitions/aptos2019-blindness-detection>

No APTOS data is distributed by this repository. Dataset files remain subject to the Kaggle competition rules and any provider rights; they are not covered by RetinaTriage’s MIT license. Each user must obtain access from the official source and determine whether the intended research, publication, redistribution, or commercial use is permitted.

## Bundled presentation images

These images are included under public-domain or CC0 terms. They are demonstration inputs, not application-specific five-grade ground truth, and their inclusion does not imply endorsement by their creators or source institutions.

### CDC diabetic retinopathy image

- Local file: `demo/samples/diabetic_retinopathy_cdc_pd.jpg`
- Source: CDC Public Health Image Library, ID 20467
- Source page: <https://phil.cdc.gov/Details.aspx?pid=20467>
- Credit: CDC / Lucille H. Young
- Rights statement: public domain; source states no copyright restrictions
- Local SHA-256: `e77dc54ab2a7715a403b25a5ea440be018bbde093b3b785a445597e3f3e1b70f`

### NIH background retinopathy image

- Local file: `demo/samples/background_retinopathy_nih_pd.jpg`
- Source page: <https://commons.wikimedia.org/wiki/File:Fundus_retinopathy_EDA03.JPG>
- Credit: National Eye Institute, National Institutes of Health, reference EDA03
- Rights statement: public domain as a work of the United States federal government
- Local SHA-256: `75c839c98d533a12c50bb367285fccee3b3e6a5af8405ac0a053cfea371a3d18`

### Normal right-eye fundus image

- Local file: `demo/samples/normal_fundus_cc0.jpg`
- Source page: <https://commons.wikimedia.org/wiki/File:Fundus_photograph_of_normal_right_eye.jpg>
- Creator: Mikael Häggström
- Rights statement: Creative Commons CC0 1.0 Universal public-domain dedication
- License deed: <https://creativecommons.org/publicdomain/zero/1.0/>
- Local SHA-256: `6bd97a7bfc54c2c794ad08771ce8f9e9223486f21d922934cfc11cb01b2efdae`

The machine-readable source and checksum record is in [`demo/samples/manifest.json`](demo/samples/manifest.json). A concise image attribution file is in [`demo/ATTRIBUTION.md`](demo/ATTRIBUTION.md).

## Architecture, frameworks, and dependencies

The model architecture is based on EfficientNet:

- Mingxing Tan and Quoc V. Le. “EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.” ICML 2019. <https://proceedings.mlr.press/v97/tan19a.html>

RetinaTriage uses third-party Python packages including Flask, TensorFlow, Keras, NumPy, pandas, Pillow, OpenCV, scikit-learn, Matplotlib, ReportLab, Waitress, pytest, and Coverage.py. Those packages are installed from the Python package index and are not relicensed by RetinaTriage. Their own licenses and notices apply. Exact versions are listed in `requirements.txt` and `requirements-dev.txt`.

Names and trademarks of third parties belong to their respective owners. Their appearance identifies provenance or interoperability and does not imply endorsement.
