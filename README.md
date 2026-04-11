# Skin Lesion Classification Benchmark with Efficient Vision Models

A notebook-first comparative study of lightweight deep learning models for **multi-class dermatoscopic skin lesion classification**. The repository evaluates multiple pretrained backbones (CNN and hybrid efficient architectures), logs per-model test performance, and includes interpretability workflows (Grad-CAM/Captum) to support model behavior analysis in a medical-imaging context.

## Why this repository is useful
- **Comparative scope:** 11 architecture families are explored in separate experiment notebooks.
- **Practical focus:** emphasizes compact/efficient models rather than only large backbones.
- **Evidence available:** notebooks contain saved training/evaluation outputs; report files summarize outcomes.
- **Medical-imaging relevance:** includes confusion-matrix/report workflows and attribution-based inspection.

## At a glance
| Item | Current status |
|---|---|
| Workflow style | Notebook-first experiments |
| Data loading pattern | `load_dataset("marmal88/skin_cancer")` in notebooks |
| Number of classes observed in outputs | 7 |
| Best reported test accuracy in saved outputs | GhostNetV2 (`98.60%`) |
| Reproducibility maturity | Moderate (no unified training script/env lockfile yet) |

## Table of contents
- [Project overview](#project-overview)
- [Dataset](#dataset)
- [Models explored](#models-explored)
- [Repository structure](#repository-structure)
- [Methodology / pipeline](#methodology--pipeline)
- [Setup and installation](#setup-and-installation)
- [How to run](#how-to-run)
- [Results and evaluation](#results-and-evaluation)
- [Explainability / visual analysis](#explainability--visual-analysis)
- [Limitations](#limitations)
- [Medical disclaimer](#medical-disclaimer)
- [Future work](#future-work)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Project overview
Skin lesion triage and diagnosis depend on reliable visual assessment, but lesion appearance varies substantially across classes and acquisition conditions. This project benchmarks efficient transfer-learning backbones for lesion classification and compares their observed test behavior in a consistent repository context.

The engineering value of this repository is the side-by-side evaluation of many efficient architectures under similar notebook workflows, making it useful for model selection and follow-up reproducibility work.

## Dataset
Notebooks load data from Hugging Face with:

- `load_dataset("marmal88/skin_cancer")`

Project materials reference HAM10000 context for dermatoscopic lesion classification. Obtain dataset assets from official sources and verify licensing/usage constraints before downstream use.

**Dataset notes (verified from repository content):**
- Raw image files are not stored in this repository.
- Notebook outputs show 7 class labels.
- Class counts/distribution are not centrally documented in the repo and should be added for reproducibility.

## Models explored
| Backbone | Notebook | Implementation note |
|---|---|---|
| `ghostnetv2_100.in1k` | [`ghostnetv2_100.in1k_batch size 128_Run3.ipynb`](./ghostnetv2_100.in1k_batch%20size%20128_Run3.ipynb) | TIMM pretrained model; batch size 128 run. |
| `efficientvit_b0.r224_in1k` | [`efficientvit_b0.r224_in1k _batch size 256_Run1.ipynb`](./efficientvit_b0.r224_in1k%20_batch%20size%20256_Run1.ipynb) | TIMM pretrained model; batch size 256 run. |
| `fastvit_t8.apple_in1k` | [`fastvit_t8.apple_in1k _batch size 128_Run1.ipynb`](./fastvit_t8.apple_in1k%20_batch%20size%20128_Run1.ipynb) | TIMM pretrained model; batch size 128 run. |
| `shufflenet_v2_x1_0` | [`ShuffleNet _batch size 256_Run2.ipynb`](./ShuffleNet%20_batch%20size%20256_Run2.ipynb) | Torchvision pretrained ShuffleNet run. |
| `levit_conv_128s.fb_dist_in1k` | [`levit_conv_128s.fb_dist_in1k _batch size 256_Run1.ipynb`](./levit_conv_128s.fb_dist_in1k%20_batch%20size%20256_Run1.ipynb) | TIMM pretrained model; batch size 256 run. |
| `mobilevit_xxs` | [`mobilevit_xxs _batch size 128_Run1.ipynb`](./mobilevit_xxs%20_batch%20size%20128_Run1.ipynb) | TIMM pretrained model; batch size 128 run. |
| `mobilenetv4_conv_small.e2400_r224_in1k` | [`mobilenetv4_conv_small.e2400_r224_in1k_batch size 256_Run3.ipynb`](./mobilenetv4_conv_small.e2400_r224_in1k_batch%20size%20256_Run3.ipynb) | TIMM pretrained model; batch size 256 run. |
| `mobileone_s0.apple_in1k` | [`mobileone.ipynb`](./mobileone.ipynb) | Includes Captum-based attribution workflow. |
| `resnext50_32x4d.a1h_in1k` | [`resnext50.ipynb`](./resnext50.ipynb) | Includes classification report + Grad-CAM pipeline. |
| `resnetv2_50.a1h_in1k` | [`resnet50.ipynb`](./resnet50.ipynb) | Includes classification report + Grad-CAM pipeline. |
| `efficientformerv2_s0` | [`efficientformer.ipynb`](./efficientformer.ipynb) | Includes confusion matrix/report workflow. |

## Repository structure
```text
.
├── README.md
├── Skin Cancer_CAPSTONE Report_Akhil Sibi.docx
├── Skin Cancer_CAPSTONE Report_Akhil Sibi_pdf file.pdf
├── ghostnetv2_100.in1k_batch size 128_Run3.ipynb
├── efficientvit_b0.r224_in1k _batch size 256_Run1.ipynb
├── fastvit_t8.apple_in1k _batch size 128_Run1.ipynb
├── ShuffleNet _batch size 256_Run2.ipynb
├── levit_conv_128s.fb_dist_in1k _batch size 256_Run1.ipynb
├── mobilevit_xxs _batch size 128_Run1.ipynb
├── mobilenetv4_conv_small.e2400_r224_in1k_batch size 256_Run3.ipynb
├── mobileone.ipynb
├── resnext50.ipynb
├── resnet50.ipynb
└── efficientformer.ipynb
```

## Methodology / pipeline
Observed common pipeline across notebooks:

1. **Dataset loading:** Hugging Face dataset splits (`train`, `validation`, `test`).
2. **Preprocessing and augmentation:** resize to 224×224, normalization, random horizontal flip, random rotation, and (in several notebooks) color jitter.
3. **Transfer learning:** pretrained TIMM/Torchvision backbone with classification head adjusted to lesion classes.
4. **Optimization:** AdamW used broadly; class-weighted losses and LR scheduling appear in multiple notebooks.
5. **Evaluation:** test accuracy and confusion matrices are common; classification reports appear in several notebooks.
6. **Interpretability:** Captum `LayerGradCam` is used in selected experiments.

Implementation details vary per notebook (batch size, epochs, scheduler choices), so direct comparisons should be interpreted with that context.

## Setup and installation
A single pinned environment file is not included.

### Suggested environment
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install torch torchvision torchaudio
pip install timm datasets scikit-learn matplotlib seaborn captum opencv-python
```

For GPU training, install the PyTorch build that matches your CUDA runtime.

## How to run
This repository is **notebook-driven**.

1. Create environment and install dependencies.
2. Start Jupyter:
   ```bash
   jupyter notebook
   ```
3. Open any model notebook (e.g., `ghostnetv2_100.in1k_batch size 128_Run3.ipynb`).
4. Run cells sequentially.
5. Confirm data loading (`load_dataset("marmal88/skin_cancer")`) and hardware device selection.
6. Repeat for other notebooks to compare model behavior.

For cleaner reproducibility, keep seed policy, epoch counts, and logging schema consistent when re-running experiments.

## Results and evaluation
Saved notebook outputs report test metrics for multiple models, and the capstone report contains a consolidated summary table.

### Reported test accuracies from notebook outputs
| Notebook | Reported test accuracy |
|---|---:|
| `ghostnetv2_100.in1k_batch size 128_Run3.ipynb` | 98.60% |
| `efficientvit_b0.r224_in1k _batch size 256_Run1.ipynb` | 98.29% |
| `fastvit_t8.apple_in1k _batch size 128_Run1.ipynb` | 97.90% |
| `ShuffleNet _batch size 256_Run2.ipynb` | 97.43% |
| `levit_conv_128s.fb_dist_in1k _batch size 256_Run1.ipynb` | 97.43% |
| `mobilevit_xxs _batch size 128_Run1.ipynb` | 97.20% |
| `mobilenetv4_conv_small.e2400_r224_in1k_batch size 256_Run3.ipynb` | 96.81% |
| `resnext50.ipynb` | 0.9619 (decimal format in output) |

Based on currently saved outputs, **GhostNetV2 is the top reported performer** in this repository snapshot. Recompute and update this section after any retraining.

### Placeholder: standardized benchmark sheet
Add one canonical comparison table with:
- top-1 accuracy
- macro/weighted F1
- per-class recall
- parameter count and throughput/latency
- exact run config (seed, epochs, batch size, image size)

## Explainability / visual analysis
`resnet50.ipynb`, `resnext50.ipynb`, and `mobileone.ipynb` include Captum/Grad-CAM style attribution workflows. In this domain, interpretability supports:
- localization checks (is attention on lesion region?),
- failure analysis for misclassifications,
- stronger qualitative confidence before broader deployment studies.

### Placeholder: qualitative figure panel
Add representative attribution examples (correct + failure cases) with short notes on model focus behavior.

## Limitations
- **Notebook-first experimentation:** reproducibility is weaker than script/config-driven pipelines.
- **Cross-notebook variance:** hyperparameters and training settings differ across runs.
- **Incomplete centralized metadata:** class distribution and final harmonized metrics are not yet maintained in one machine-readable results artifact.
- **Generalization uncertainty:** experiments are centered on one dataset source and need external validation.
- **Non-clinical scope:** repository content is research/education oriented.

## Medical disclaimer
This repository is for research and educational use only. It is **not** a clinical diagnostic system and must not be used as a substitute for medical judgment.

## Future work
- Refactor notebooks into reproducible training/evaluation scripts with shared config files.
- Add experiment tracking (e.g., MLflow or Weights & Biases) for versioned runs.
- Publish a single benchmark artifact with harmonized metrics across all models.
- Add external-dataset validation and subgroup error analysis.
- Measure compression/inference trade-offs for deployment-constrained settings.
- Expand interpretability analysis with systematic false-positive/false-negative review.

## Contributing
Contributions are welcome.

1. Open an issue describing the intended change.
2. Keep experiment settings explicit (seed, split, model, epochs, batch size).
3. Report metrics in a consistent tabular format.
4. Prefer focused pull requests with clear scope.

## License
License not specified yet. Add a LICENSE file to clarify reuse permissions.

## Acknowledgments
- Capstone report author and supervisor listed in project report files.
- Open-source communities behind PyTorch, TIMM, Hugging Face Datasets, and Captum.
