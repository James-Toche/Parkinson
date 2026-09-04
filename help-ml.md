# Parkinson's Prediction Model: AUC Performance Summary

## Metric Evaluations
* **AUC = 0.55 (Very Poor):** Only 5% above random chance (`0.50`). The model has virtually no discriminative power.
* **AUC = 0.72 (Moderate / Acceptable):** There is a **72% probability** the model ranks a true Parkinson's case higher than a control. Real-world quality depends on input data and clinical task.

---

## Performance Context by Data & Objective

| Use Case Category | Typical Data Sources | Benchmark AUC Range | Interpretation of 0.72 |
| :--- | :--- | :--- | :--- |
| **High-Signal Diagnostics** | Gait sensors, voice acoustics, DAT-SPECT / MRI brain imaging | `0.85 – 0.98` | **Low / Underperforming** (physical signal should yield higher separation). |
| **Low-Signal / Early Screening** | Routine EHR, standard blood panels, demographic surveys | `0.60 – 0.75` | **Strong / Respectable** (valuable for noisy, population-level triage). |
| **Long-Term Prognostics** | Progression tracking (e.g., motor decline, cognitive onset over 5 years) | `0.65 – 0.75` | **Solid Baseline** (disease trajectories are inherently stochastic). |

---

## Recommended Next Steps to Improve Beyond 0.72
1. **Address Class Imbalance:** Check positive-to-negative ratios; evaluate minority oversampling (e.g., SMOTE) or class-weighted loss functions.
2. **Refine Domain Features:** Extract specialized markers (e.g., MFCCs for voice recordings, gait stride variability metrics).
3. **Verify Data Leakage:** Ensure features raising performance from 0.55 to 0.72 do not inadvertently encode diagnostic outcomes (e.g., post-diagnosis prescriptions like Levodopa).
4. **Evaluate Precision-Recall (PR-AUC):** If Parkinson's prevalence is very low in the cohort, inspect PR-AUC and F1-score alongside ROC-AUC.


# Model Evaluation: ROC AUC vs. Accuracy

| Attribute | **ROC AUC** | **Accuracy** |
| :--- | :--- | :--- |
| **Core Measurement** | **Relative Ranking:** The probability that a randomly chosen positive case is scored higher than a randomly chosen negative case. | **Absolute Classification:** The percentage of total predictions that the model got exactly right (True Positives + True Negatives). |
| **Baseline Value** | **0.50** represents a purely random guess (coin flip). | **Varies** based on class distribution (e.g., 90% baseline if 90% of samples belong to one class). |
| **Threshold Dependance** | **Independent.** Evaluates model performance across *all possible decision thresholds* simultaneously. | **Highly Dependent.** Requires choosing a fixed decision threshold (typically `0.5`) to turn probabilities into hard classes. |
| **Handling of Imbalanced Data** | **Robust.** Maintains a steady `0.5` random baseline even if one class vastly outperforms or outnumbers the other. | **Fragile.** Can mask terrible models on imbalanced sets (e.g., a broken model guessing "no fraud" on a 99% non-fraud dataset yields 99% accuracy). |
| **Best Used For** | Evaluating overall model separation strength, grading probability rankings, and tuning models before a threshold is locked. | Understanding real-world business impact and operational performance *after* a specific decision threshold is locked in. |

---

## Key Takeaway
* **Accuracy** asks: *"How many absolute labels did the model get right at a single cutoff point?"*
* **ROC AUC** asks: *"How well does the model sort and rank positive cases above negative cases across the entire spectrum?"*
