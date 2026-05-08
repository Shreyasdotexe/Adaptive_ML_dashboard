# Adaptive ML System for Concept Drift Detection and Online Model Re-Training



---

## Project Overview

This project implements a fully streaming, adaptive machine learning system that detects concept drift in real-world data streams and automatically retrains an online classifier to maintain predictive performance.

Concept drift refers to the phenomenon where the statistical properties of a data stream change over time, degrading the performance of a deployed model. This system detects such changes automatically using the ADWIN algorithm and compares it against DDM and EDDM, then triggers model retraining to restore accuracy.

---

## System Architecture

```
RealStreamLoader
      │
      ▼
OnlinePreprocessor  (incremental StandardScaler via partial_fit)
      │
      ▼
OnlineModel  (SGDClassifier — partial_fit per sample)
      │
      ▼
DetectorManager  ──► ADWINDetector  (PRIMARY — triggers retraining)
      │          ──► DDMDetector    (comparison only)
      │          ──► EDDMDetector   (comparison only)
      │
      ▼
RetrainingManager  (resets model on ADWIN drift signal)
      │
      ▼
EvaluationModule  →  Confusion matrix, F1, Kappa, windowed accuracy,
      │               recovery time analysis
      ▼
StreamLogger  →  data/logs/stream_log_<run_id>.csv
      │
      ▼
ResultVisualizer  →  outputs/plots/  (7 plot types)
RunReporter       →  outputs/reports/
      │
      ▼
Dashboard  (streamlit run dashboard.py)  →  11 sections
```

---

## Project Structure

```
adaptive_ml/
│
├── data/
│   ├── electricity.csv               ← ELEC2 dataset (convert from ARFF)
│   └── logs/
│       └── stream_log_<run_id>.csv
│
├── src/
│   ├── real_stream_loader.py         ← Loads ELEC2, yields stream samples
│   ├── preprocessing.py              ← Incremental StandardScaler (partial_fit)
│   ├── online_model.py               ← SGDClassifier with partial_fit
│   ├── retraining_manager.py         ← Resets model on confirmed drift
│   ├── evaluation.py                 ← Streaming metrics: confusion matrix, F1,
│   │                                    Kappa, windowed accuracy, recovery time
│   ├── experiment_runner.py          ← Structured experiments:
│   │                                    with/without retraining comparison,
│   │                                    ADWIN delta sensitivity ablation
│   ├── logger.py                     ← Row-by-row CSV logger
│   ├── visualizer.py                 ← Matplotlib plots (7 outputs)
│   ├── utils.py                      ← Console formatting, timer, run ID
│   ├── adwin.py                      ← Pure-Python ADWIN implementation
│   ├── drift_detection.py            ← ADWIN wrapper (Phase 1/2/3 compat)
│   └── drift_detectors/
│       ├── __init__.py
│       ├── adwin_detector.py         ← ADWIN with DetectorManager interface
│       ├── ddm_detector.py           ← DDM (Gama et al., 2004)
│       ├── eddm_detector.py          ← EDDM (Baena-García et al., 2006)
│       └── detector_manager.py       ← Runs all 3 detectors in parallel
│
├── tests/
│   ├── test_adwin.py                 ← ADWIN algorithm tests
│   ├── test_ddm.py                   ← DDM detector tests
│   ├── test_eddm.py                  ← EDDM detector tests
│   ├── test_online_model.py          ← SGDClassifier wrapper tests
│   ├── test_preprocessing.py         ← Online preprocessing tests
│   ├── test_stream_loader.py         ← Dataset loader tests
│   ├── test_retraining_manager.py    ← Retraining logic tests
│   ├── test_evaluation.py            ← Evaluation metrics tests
│   └── test_integration.py           ← End-to-end pipeline tests
│
├── outputs/
│   ├── plots/                        ← PNG visualisation outputs (7 types)
│   ├── reports/
│   │   ├── run_report_<run_id>.txt   ← Human-readable report
│   │   └── meta_<run_id>.json        ← Machine-readable (used by dashboard)
│   └── experiments/
│       ├── comparison_<run_id>.json   ← With vs without retraining results
│       ├── comparison_<run_id>.png    ← Comparison plot
│       ├── ablation_<run_id>.json     ← Delta sensitivity results
│       └── ablation_<run_id>.png      ← Ablation plot
│
├── main.py                           ← Pipeline entry point
├── dashboard.py                      ← Streamlit dashboard (11 sections)
├── requirements.txt
├── pyproject.toml                    ← Project metadata & test config
└── README.md
```

---

## Phase Summary

| Phase | Description | Key Addition |
|---|---|---|
| **Phase 1** | Streaming pipeline with synthetic data | ADWIN drift detection, CSV logging, plots |
| **Phase 2** | Online model integration | SGDClassifier, error-based detection, retraining |
| **Phase 3** | Real dataset (ELEC2) | RealStreamLoader replaces simulator |
| **Phase 4** | Multi-detector comparison + evaluation | DDM, EDDM, DetectorManager, evaluation metrics, experiments, tests, Streamlit dashboard |

---

## Dataset

**ELEC2 — Electricity Pricing Dataset**

| Property | Value |
|---|---|
| Source | Harries, M. (1999). SPLICE-2 Comparative Evaluation: Electricity Pricing. UNSW. |
| Samples | 45,312 |
| Features | 8 (date, day, period, nswprice, nswdemand, vicprice, vicdemand, transfer) |
| Target | class — UP (1) or DOWN (0) |
| Normalisation | All features pre-normalised to [0, 1] |

The ELEC2 dataset is the standard benchmark for concept drift research. Real drift occurs naturally due to seasonal pricing changes, demand growth, and policy shifts — no artificial injection is needed.

**Dataset preparation:**
```bash
# The ARFF file must be converted to CSV before running.
# Place electricity.csv in the data/ folder.
# The real_stream_loader.py handles loading automatically.
```

---

## Installation

### 1. Clone / extract the project

```bash
cd adaptive_ml
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# macOS / Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## How to Run

### Run the pipeline

```bash
python main.py
```

This streams all 45,312 samples through the full pipeline. Expected runtime: ~5 minutes on a standard laptop.

For a quick test run, set `"max_samples": 5000` in the `CONFIG` dict in `main.py`.

### Launch the dashboard

```bash
streamlit run dashboard.py
```

Opens at `http://localhost:8501`. The dashboard automatically loads the most recent run from `outputs/reports/meta_<run_id>.json`. Use the sidebar to switch between previous runs.

### Run experiments

```bash
python -m src.experiment_runner
```

Runs two experiments:
1. **With vs Without Retraining** — proves adaptive retraining improves accuracy
2. **ADWIN Delta Sensitivity** — ablation study across 5 delta values

Results saved to `outputs/experiments/`.

### Run tests

```bash
pytest
```

Runs the full test suite (unit tests + integration tests). Integration tests require `electricity.csv` in the `data/` directory.

---

## Expected Output Files

After a successful `python main.py` run:

```
data/logs/
  stream_log_<run_id>.csv              ← 45312 rows, 13 columns

outputs/plots/
  accuracy_<run_id>.png               ← Running accuracy + retraining markers
  drift_signal_<run_id>.png           ← ADWIN error signal rolling mean
  detection_timeline_<run_id>.png     ← Two-panel detection timeline
  feature_distribution_<run_id>.png   ← Feature histogram pre/post midpoint
  multi_feature_dist_<run_id>.png     ← Multi-feature distribution comparison
  cm_*_<run_id>.png                   ← Confusion matrix heatmap
  windowed_accuracy_<run_id>.png      ← Sliding window accuracy plot

outputs/reports/
  run_report_<run_id>.txt             ← Full text report (with eval metrics)
  meta_<run_id>.json                  ← Dashboard data file (with evaluation)
```

---

## Evaluation Metrics

The system computes the following streaming evaluation metrics:

| Metric | Description |
|---|---|
| **Prequential Accuracy** | Test-then-train cumulative accuracy |
| **Windowed Accuracy** | Accuracy over a sliding window of 500 predictions |
| **Cohen's Kappa** | Agreement beyond chance (handles class imbalance) |
| **F1 Score** | Harmonic mean of precision and recall |
| **Precision / Recall** | Per-class and overall |
| **Specificity** | True negative rate |
| **Confusion Matrix** | TP, TN, FP, FN — overall and per-half |
| **Recovery Time** | Samples needed to recover 85%+ accuracy after retraining |

---

## CSV Log Schema

| Column | Description |
|---|---|
| `sample_index` | Row index in the stream (0-based) |
| `feature_0` … `feature_7` | Scaled feature values |
| `true_label` | Ground truth (0=DOWN, 1=UP) |
| `detector_input` | Binary error fed to ADWIN (NaN during warmup) |
| `drift_detected` | 1 if a cooldown-gated drift event fired this step |
| `phase` | Always "real_stream" for ELEC2 |
| `prediction` | Model output (-1 during warmup) |
| `error` | 1 if prediction was wrong, 0 if correct (-1 during warmup) |
| `running_acc` | Cumulative accuracy at this sample |
| `retrained` | 1 if model was reset at this step |

---

## Dashboard Sections

The Streamlit dashboard provides 11 interactive sections:

| Section | Content |
|---|---|
| 1 | Run Summary (15 metric cards including F1, Kappa, Precision, Recall) |
| 2 | Running Accuracy Over Time (dual-panel) |
| 3 | ADWIN Error Signal |
| 4 | Drift Detection Timeline |
| 5 | Windowed Accuracy (with interactive Streamlit chart) |
| 6 | Confusion Matrix & Classification Report (with first-half vs second-half) |
| 7 | Recovery Time Analysis |
| 8 | Detector Comparison Table + Bar Chart |
| 9 | Retraining Events Timeline |
| 10 | Feature Distribution Shift (single + multi-feature) |
| 11 | Raw Log Data Explorer (filterable, downloadable) |

---

## Experiments

### With vs Without Retraining (Thesis Proof)

Demonstrates that adaptive retraining under concept drift objectively improves model performance. Runs the pipeline twice:
- **With retraining** — ADWIN triggers model reset on drift
- **Without retraining** — drift detected but model continues with old weights

Compares: accuracy, F1, Kappa, windowed accuracy.

### ADWIN Delta Sensitivity (Ablation Study)

Tests ADWIN with δ ∈ {0.001, 0.01, 0.05, 0.1, 0.5}. Lower δ = more sensitive (more drift detections). Compares accuracy, F1, Kappa, and drift count across configurations.

---

## Phase 4 Results (Full Dataset)

| Metric | Value |
|---|---|
| Samples processed | 45,312 |
| Final running accuracy | **91.6%** |
| First-half average accuracy | 93.9% |
| Second-half average accuracy | 91.9% |
| ADWIN drifts detected | **3** (samples 23919, 25153, 41879) |
| DDM drifts detected | 21 |
| EDDM drifts detected | 2 |
| Retraining events | 3 |

### Detector Comparison

| Detector | Drifts | Avg Gap | Role |
|---|---|---|---|
| ADWIN | 3 | 8,980 samples | Primary — triggers retraining |
| DDM | 21 | 1,083 samples | Comparison — sensitive to error spikes |
| EDDM | 2 | 18,175 samples | Comparison — targets gradual drift |

---

## Key Design Decisions

**Why ADWIN as primary detector?**
ADWIN uses adaptive window sizing — it requires the *entire* window mean to shift before declaring drift, making it robust to noise. DDM responds to any local error spike, producing more false positives on noisy data.

**Why SGDClassifier?**
It is the only scikit-learn classifier with native `partial_fit` support, enabling true sample-by-sample learning without storing the dataset in memory.

**Why full model reset (not partial adaptation)?**
The ELEC2 drift events are abrupt. Retaining old weights after a confirmed distribution shift actively harms the model. Full reset is standard practice for sudden drift (Gama et al., 2014).

**Why `partial_fit` for preprocessing?**
`sklearn.StandardScaler.partial_fit()` uses Welford's online algorithm internally — O(1) memory, no batch required. This is the theoretically correct approach for streaming normalisation.

**Why prequential evaluation?**
Prequential (interleaved test-then-train) evaluation is the standard for data stream classification (Gama et al., 2013). Each sample is first used for testing, then for training — no held-out set needed.

---

## Test Suite

The project includes comprehensive tests:

| Test File | Tests | Coverage |
|---|---|---|
| `test_adwin.py` | 9 | ADWIN algorithm: init, constant stream, sudden drift, sensitivity |
| `test_ddm.py` | 7 | DDM: init, no-drift, burst detection, min_samples, reset |
| `test_eddm.py` | 7 | EDDM: init, error counting, gap-based detection, reset |
| `test_online_model.py` | 8 | SGD wrapper: readiness, predict, reset, multi-feature |
| `test_preprocessing.py` | 8 | Scaler: warmup, scaling, shape, stats |
| `test_stream_loader.py` | 7 | Loader: format, labels, features, missing file |
| `test_retraining_manager.py` | 7 | Manager: model reset, detector reset, accumulation |
| `test_evaluation.py` | 18 | Metrics: confusion matrix, windowed accuracy, recovery time |
| `test_integration.py` | 3 | End-to-end: pipeline run, output validation |

---

## Academic References

1. Bifet, A., & Gavalda, R. (2007). *Learning from Time-Changing Data with Adaptive Windowing.* SIAM SDM 2007.
2. Gama, J., Medas, P., Castillo, G., & Rodrigues, P. (2004). *Learning with Drift Detection.* SBIA 2004, LNAI 3171.
3. Baena-García, M. et al. (2006). *Early Drift Detection Method.* ECML PKDD Workshop.
4. Bayram, F., Ahmed, B. S., & Kassler, A. (2022). *From Concept Drift to Model Degradation: An Overview on Performance-Aware Drift Detectors.* Knowledge-Based Systems, 245, 108632.
5. Lu, J., Liu, A., Dong, F., Gu, F., Gama, J., & Zhang, G. (2018). *Learning under Concept Drift: A Review.* IEEE Transactions on Knowledge and Data Engineering, 31(12), 2346–2363.
6. Bifet, A., Gavaldà, R., Holmes, G., & Pfahringer, B. (2018). *Machine Learning for Data Streams: With Practical Examples in MOA.* MIT Press.
7. Montiel, J. et al. (2021). *River: Machine Learning for Streaming Data in Python.* Journal of Machine Learning Research, 22(110), 1–8.
8. Webb, G. I., Hyde, R., Cao, H., Nguyen, H. L., & Petitjean, F. (2016). *Characterizing Concept Drift.* Data Mining and Knowledge Discovery, 30(4), 964–994.
9. Losing, V., Hammer, B., & Wersing, H. (2018). *Incremental On-Line Learning: A Review and Comparison of State of the Art Algorithms.* Neurocomputing, 275, 1261–1274.
10. Pesaranghader, A., & Viktor, H. L. (2016). *Fast Hoeffding Drift Detection Method for Evolving Data Streams.* ECML-PKDD 2016.
11. Sethi, T. S., & Kantardzic, M. (2017). *On the Reliable Detection of Concept Drift from Streaming Unlabeled Data.* Expert Systems with Applications, 82, 77–99.
12. Gama, J. et al. (2013). *On Evaluating Stream Learning Algorithms.* Machine Learning, 90(3), 317–346.
13. Gama, J. et al. (2014). *A Survey on Concept Drift Adaptation.* ACM Computing Surveys, 46(4), Article 44.
14. Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR, 12, 2825–2830.
15. Harries, M. (1999). *SPLICE-2 Comparative Evaluation: Electricity Pricing.* UNSW Technical Report.

---

## License

Submitted as part of an academic final year project. All code is original.
