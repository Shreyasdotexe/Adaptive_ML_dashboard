# Adaptive ML System for Concept Drift Detection and Online Model Retraining

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-SGDClassifier-F7931E?style=flat&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Dataset](https://img.shields.io/badge/Dataset-ELEC2%2045%2C312%20samples-4CAF50?style=flat)](https://www.openml.org/d/151)
[![Tests](https://img.shields.io/badge/Tests-79%20passing-brightgreen?style=flat)](tests/)

Final Year Capstone Project — MIT World Peace University, Pune

---

## Overview

Most ML models are trained once and deployed forever. That works fine until the world changes — and it always does. Concept drift is when the statistical properties of a data stream shift over time, silently degrading the performance of a deployed model.

This project builds a fully streaming, adaptive machine learning pipeline that:
- Streams 45,312 real electricity pricing samples one at a time — no batching, no lookback
- Detects distribution shifts automatically using ADWIN, with DDM and EDDM running in parallel for comparison
- Retrains the classifier on confirmed drift, recovering above 85% accuracy within ~95 samples on average
- Tracks everything through a live Streamlit dashboard with comparison charts

**Final result: 89.94% prequential accuracy — +15.3 percentage points over a static frozen baseline.**

---

## Results

### Core Metrics (Full ELEC2 Dataset — 45,312 samples)

| Metric | Value |
|---|---|
| Prequential Accuracy | **89.94%** |
| First-half Accuracy | 93.09% |
| Second-half Accuracy | 90.39% |
| F1-Score | 0.8808 |
| Cohen's Kappa | 0.7937 |
| ADWIN Drift Events | 27 |
| Avg Recovery Time | ~95 samples |
| Events Fully Recovered | 27 / 27 |

### Three-Model Comparison

| Model | Accuracy | F1 | Kappa | vs Static |
|---|---|---|---|---|
| Static (frozen after 10%) | 74.62% | 0.679 | 0.473 | — |
| Continuous (blind updates) | 91.60% | 0.901 | 0.829 | +17.0 pp |
| Adaptive ADWIN (ours) | **89.94%** | **0.881** | **0.794** | **+15.3 pp** |

The continuous model edges ahead on raw accuracy by never discarding learned weights, but it has zero awareness of when drift occurred and cannot respond to abrupt shifts. The adaptive model trades a small accuracy margin for explicit drift detection, recovery tracking, and full observability.

### Detector Comparison

| Detector | Drifts Detected | Avg Gap | Role |
|---|---|---|---|
| ADWIN | 27 | 1,549 samples | Primary — triggers retraining |
| DDM | 15 | 1,897 samples | Comparison — fires on noise bursts |
| EDDM | 3 | 10,534 samples | Comparison — misses subtle shifts |

### ADWIN Delta Sensitivity

| Delta | Accuracy | F1 | Kappa | Drift Events |
|---|---|---|---|---|
| 0.001 (conservative) | 91.09% | 0.8950 | 0.8176 | 5 |
| 0.01 | 90.49% | 0.8879 | 0.8054 | 14 |
| 0.05 | 90.15% | 0.8835 | 0.7982 | 22 |
| 0.10 (default) | 89.94% | 0.8808 | 0.7937 | 27 |
| 0.50 (aggressive) | 87.63% | 0.8571 | 0.7459 | 41 |

Lower delta demands stronger statistical evidence before triggering, which means fewer resets and better overall accuracy. At delta=0.001, only 5 resets were needed and accuracy nearly matched the continuous baseline.

---

## System Architecture

```
RealStreamLoader
      |
      v
OnlinePreprocessor  (incremental StandardScaler via Welford's algorithm)
      |
      v
OnlineModel  (SGDClassifier — partial_fit per sample)
      |
      v
DetectorManager  --> ADWINDetector   (PRIMARY — triggers retraining)
      |          --> DDMDetector     (comparison only)
      |          --> EDDMDetector    (comparison only)
      |
      v
RetrainingManager  (resets model on ADWIN drift, 300-sample cooldown)
      |
      v
EvaluationModule  -->  F1, Kappa, confusion matrix, recovery time
      |
      v
StreamLogger  -->  data/logs/stream_log_<run_id>.csv
      |
      v
ResultVisualizer  -->  outputs/plots/
RunReporter       -->  outputs/reports/
comparison.py     -->  outputs/paper_figures/
      |
      v
Dashboard.py  -->  pages/2_Results.py
```

---

## Project Structure

```
adaptive_ml/
|
+-- data/
|   +-- electricity.csv               <- ELEC2 dataset (45,312 samples)
|   +-- logs/                         <- Per-run stream logs (gitignored)
|
+-- src/
|   +-- real_stream_loader.py         <- Streams ELEC2 row by row
|   +-- preprocessing.py              <- Welford online StandardScaler
|   +-- online_model.py               <- SGDClassifier with partial_fit
|   +-- retraining_manager.py         <- Resets model on confirmed drift
|   +-- evaluation.py                 <- F1, Kappa, windowed accuracy, recovery time
|   +-- experiment_runner.py          <- With/without retraining + delta ablation
|   +-- logger.py                     <- Row-by-row CSV logger
|   +-- visualizer.py                 <- 7 per-run matplotlib plots
|   +-- utils.py                      <- Timer, run ID, console formatting
|   +-- adwin.py                      <- Pure-Python ADWIN implementation
|   +-- drift_detection.py            <- ADWIN wrapper
|   +-- drift_detectors/
|       +-- adwin_detector.py         <- ADWIN (Bifet & Gavalda, 2007)
|       +-- ddm_detector.py           <- DDM (Gama et al., 2004)
|       +-- eddm_detector.py          <- EDDM (Baena-Garcia et al., 2006)
|       +-- detector_manager.py       <- Runs all 3 in parallel
|
+-- tests/                            <- 79 unit tests, all passing
|   +-- test_adwin.py
|   +-- test_ddm.py
|   +-- test_eddm.py
|   +-- test_preprocessing.py
|   +-- test_model.py
|   +-- test_evaluation.py
|   +-- test_retraining_manager.py
|   +-- test_dashboard.py
|
+-- outputs/
|   +-- plots/                        <- Per-run dark-theme plots (gitignored)
|   +-- reports/                      <- run_report & meta JSON (gitignored)
|   +-- experiments/                  <- Ablation & comparison outputs (gitignored)
|   +-- paper_figures/                <- Publication-quality comparison charts
|       +-- fig1_detector_comparison.png
|       +-- fig2_running_accuracy.png
|       +-- fig3_model_comparison.png
|       +-- fig4_delta_ablation.png
|       +-- fig5_cm_and_recovery.png
|
+-- pages/
|   +-- 1_Run_Detection.py            <- Run the pipeline from the UI
|   +-- 2_Results.py                  <- Results viewer with comparison charts
|
+-- main.py                           <- Pipeline entry point
+-- Dashboard.py                      <- Streamlit multi-page app
+-- comparison.py                     <- Generates paper_figures/ (run once)
+-- requirements.txt
+-- pyproject.toml
+-- README.md
```

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/Shreyasdotexe/Adaptive_ML_dashboard.git
cd Adaptive_ML_dashboard
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Place `electricity.csv` in the `data/` folder. The file can be obtained from [OpenML (ELEC2, ID 151)](https://www.openml.org/d/151) by converting the ARFF to CSV.

---

## Running the System

**Run the full pipeline:**
```bash
python main.py
```
Streams all 45,312 samples. Expected runtime ~5 minutes on a standard laptop.
For a quick test, set `"max_samples": 5000` in the `CONFIG` dict in `main.py`.

**Launch the dashboard:**
```bash
streamlit run Dashboard.py
```
Opens at `http://localhost:8501`. Loads the most recent run automatically.

**Generate comparison figures:**
```bash
python comparison.py
```
Produces 5 publication-quality charts in `outputs/paper_figures/` (white background, 300 DPI). These are shown directly on the Results page.

**Run experiments:**
```bash
python -m src.experiment_runner
```
Runs the with/without retraining comparison and the ADWIN delta ablation study.

**Run tests:**
```bash
pytest
```

---

## Dashboard

The Results page (`pages/2_Results.py`) has these sections:

| Section | Content |
|---|---|
| Overview | 10 metric cards — accuracy, F1, Kappa, Precision, Recall, half-splits |
| Running Accuracy | Full accuracy trace with every retraining event marked |
| Confusion Matrix | TP/TN breakdown + first vs second half comparison |
| Recovery Analysis | Per-event table showing samples to recover after each reset |
| Detector Comparison | ADWIN vs DDM vs EDDM — counts, gaps, full timeline chart |
| Retraining Events | Sample index and accuracy at each of the 27 drift points |
| Model Comparison | Static vs Continuous vs Adaptive — Accuracy, F1, Kappa |
| Delta Sensitivity | Ablation study across 5 delta values |
| CM + Recovery | Combined confusion matrix and recovery-time histogram |
| Raw Log Explorer | Filter, browse and download the 45,312-row stream log |

---

## Test Suite

| File | Tests | Coverage |
|---|---|---|
| test_adwin.py | 9 | Window growth, drift trigger, delta sensitivity, edge cases |
| test_ddm.py | 7 | Init, all-correct stream, error burst, min-samples guard, reset |
| test_eddm.py | 6 | Consecutive error tracking, threshold logic, reset |
| test_preprocessing.py | 11 | Scaling, Welford stability, unseen features |
| test_model.py | 14 | Predict, partial fit, reset, warm-up logic |
| test_evaluation.py | 17 | Prequential accuracy, Kappa, confusion matrix, windowed |
| test_retraining_manager.py | 8 | Cooldown, reset trigger, false alarm filtering |
| test_dashboard.py | 7 | Chart rendering, metric display, live updates |
| **Total** | **79** | **All passing** |

---

## Key Design Decisions

**Why ADWIN as the primary detector?**
ADWIN requires the entire window mean to shift before declaring drift, which gives it formal guarantees on false-alarm rate. DDM fires on any local error spike and produces too many false positives on noisy data. EDDM only reacts to sustained accuracy collapses and misses abrupt shifts. ADWIN sits in the middle with statistical backing for every trigger.

**Why SGDClassifier?**
It is the only scikit-learn classifier with native `partial_fit` support, enabling true sample-by-sample learning without ever storing the dataset in memory.

**Why a hard model reset on drift?**
ELEC2 drift events are abrupt. Retaining old weights after a confirmed distribution shift actively harms accuracy. Full reset is standard practice for sudden drift (Gama et al., 2014). For gradual drift, exponential weight decay would be the better approach.

**Why Welford's algorithm for preprocessing?**
`StandardScaler.partial_fit()` uses Welford's algorithm internally — O(1) memory, no batch required. This is the theoretically correct approach for streaming normalisation.

**Why prequential evaluation?**
Each sample is tested before the model trains on it — no held-out set needed, and the accuracy estimate is unbiased across the full stream (Gama et al., 2013).

---

## Dataset

**ELEC2 — Electricity Pricing (Harries, 1999, UNSW)**

| Property | Value |
|---|---|
| Source | Harries (1999), UNSW Technical Report |
| Samples | 45,312 |
| Features | 8 — period, nswprice, nswdemand, vicprice, vicdemand, transfer, day |
| Target | Binary — UP (1) or DOWN (0) |
| Drift Type | Gradual, real-world (seasonal pricing and demand shifts) |

The ELEC2 dataset is the standard benchmark for concept drift research. Drift occurs naturally due to seasonal pricing changes, demand growth, and policy shifts — no artificial injection needed.

---

## References

1. Bifet, A., & Gavalda, R. (2007). Learning from Time-Changing Data with Adaptive Windowing. SIAM SDM.
2. Gama, J., Medas, P., Castillo, G., & Rodrigues, P. (2004). Learning with Drift Detection. SBIA 2004.
3. Baena-Garcia, M. et al. (2006). Early Drift Detection Method. ECML PKDD Workshop.
4. Gama, J. et al. (2013). On Evaluating Stream Learning Algorithms. Machine Learning, 90(3), 317–346.
5. Gama, J. et al. (2014). A Survey on Concept Drift Adaptation. ACM Computing Surveys, 46(4).
6. Losing, V., Hammer, B., & Wersing, H. (2018). Incremental On-Line Learning: A Review. Neurocomputing, 275.
7. Bifet, A. et al. (2018). Machine Learning for Data Streams. MIT Press.
8. Lu, J. et al. (2018). Learning under Concept Drift: A Review. IEEE TKDE, 31(12), 2346–2363.
9. Bayram, F. et al. (2022). From Concept Drift to Model Degradation. Knowledge-Based Systems, 245.
10. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. JMLR, 12.
11. Harries, M. (1999). SPLICE-2 Comparative Evaluation: Electricity Pricing. UNSW.
12. Montiel, J. et al. (2021). River: ML for Streaming Data in Python. JMLR, 22(110).

---

## Team

| Name | Role |
|---|---|
| Shreyas Kshirsagar | Pipeline architecture, ADWIN integration, evaluation |
| Ketavya Chitransh | Drift detectors (DDM, EDDM), experiment runner |
| Neel Walke | Streamlit dashboard, visualiser, test suite |
| Dr. Vaishali Suryavanshi | Project supervisor |

B-Tech, Dept. of Computer Engineering & Technology, MIT World Peace University, Pune, India

---

## License

Submitted as part of an academic final year capstone project. All code is original.
