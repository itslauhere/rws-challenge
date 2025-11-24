# RWS Challenge – Short-Horizon ES Direction Forecasting

**Author:** Laurentia Liennart

This project investigates whether the **30-minute ahead direction of ES (S&P 500 E-mini Futures)** can be predicted using short-horizon cross-asset features such as FX pairs and volatility indices.

---

## 1. Problem Overview

* **Target asset:** ES (S&P 500 E-mini futures)
* **Data frequency:** Minute-level OHLCV
* **Prediction horizon:** 30 minutes
* **Task:** Binary classification (up = 1, down = 0)

The target is constructed as:

```python
target_ES_30m  = ret_ES.shift(-30)
target_label   = (target_ES_30m > 0).astype(int)
```

---

## 2. Hypothesis

Short-term ES direction may be predictable using signals from:

* 1-minute lagged returns across ES, FX, VIX, and GVZ
* Rolling volatility (short and long windows)
* Cross-asset rolling correlations
* Momentum measures
* Volatility regime indicators

**Expectation:**
Cross-asset structure and volatility information should improve accuracy beyond simple lagged returns.

---

## 3. Data and Preprocessing

### 3.1 Data Sources

Data is taken from the RWS Challenge dataset and includes:

* ES
* USDJPY
* USDCAD
* VIX
* GVZ

### 3.2 Cleaning and Loading

* Load parquet (preferred) or CSV
* Standardize timestamps (`ts → timestamp`)
* Convert to timezone-naive datetime
* Sort chronologically
* Compute per-minute returns

### 3.3 Combined Dataset

All assets are merged on timestamp, producing aligned return columns:

```
timestamp | ret_ES | ret_USDJPY | ret_USDCAD | ret_VIX | ret_GVZ
```

---

## 4. Feature Engineering

### Lagged Returns

* `lag_ES`, `lag_USDJPY`, `lag_USDCAD`, `lag_VIX`, `lag_GVZ`

### Volatility Features (Rolling Std)

* `vol_ES_60`, `vol_ES_120`, `vol_ES_300`
* Vol features for other assets in the full feature set

### Cross-Asset Correlations

* `corr_ES_VIX_60`, `corr_ES_VIX_300`
* `corr_ES_USDJPY_60`, `corr_ES_USDJPY_300`
* `corr_ES_USDCAD_60`, `corr_ES_USDCAD_300`

### Momentum Features

* `mom_ES_5`, `mom_ES_10`, `mom_ES_30`

### Volatility Regime

High-volatility flag based on the 70th percentile of 300-min ES volatility:

```
regime_high_vol = (vol_ES_300 > vol_ES_300.quantile(0.7)).astype(int)
```

---

## 5. Modeling Approach

### Time-Respecting Evaluation

All models use **TimeSeriesSplit (5 folds)**:

* Training = past
* Testing = future
* Prevents information leakage
* Same fold boundaries for all models and baselines

### Models

* **Logistic Regression** (`max_iter=2000`)
* **Random Forest Classifier** (200 trees, depth-controlled)

---

## 6. Baseline Performance

Two baselines are evaluated using identical folds.

| Baseline          | Mean Accuracy |
| ----------------- | ------------- |
| Random baseline   | 0.513         |
| Majority baseline | **0.564**     |

The majority baseline is strong due to mild label imbalance.

---

## 7. Ablation Study

Four feature groups were tested:

1. **lags_only** (5 features)
2. **lags_plus_ES_vol** (8 features)
3. **lags_ES_vol_plus_corr** (14 features)
4. **full_features** (27 features)

### Logistic Regression Ablation Results

| Feature Group         | # Features | Mean Accuracy | Std Dev |
| --------------------- | ---------- | ------------- | ------- |
| lags_only             | 5          | **0.5641**    | 0.0275  |
| lags_plus_ES_vol      | 8          | **0.5641**    | 0.0275  |
| lags_ES_vol_plus_corr | 14         | 0.5640        | 0.0276  |
| full_features         | 27         | 0.5639        | 0.0277  |

### Interpretation

* **Lag-only model performs best**.
* Volatility, momentum, and correlation features **do not improve performance**.
* Larger feature sets add noise, not signal.
* Predictive information is contained almost entirely in **1-minute lagged returns**.

---

## 8. Final Model Comparison

| Model                         | Accuracy  |
| ----------------------------- | --------- |
| Random baseline               | 0.513     |
| Majority baseline             | **0.564** |
| Logistic (lags only)          | **0.564** |
| Logistic (full features)      | 0.564     |
| Random Forest (full features) | ~0.56     |

**No model outperforms the majority baseline.**

---

## 9. Limitations

* Predictive signal at 30-minute horizon is extremely weak
* Class imbalance makes majority baseline strong
* Only ES is modeled
* No transaction cost analysis
* Only logistic regression and RF evaluated

---

## 10. Future Work

* Try alternative horizons (5, 10, 60 min)
* Train separate models for high/low volatility regimes
* Add macroeconomic or cross-asset features
* Explore nonlinear sequence models (LSTM, Transformers)
* Evaluate probabilistic metrics (log-loss, Brier score)

---

## 11. Repository Structure

```
.
├── README.md
├── requirements.txt
├── notebooks/
│   └── RWS_Challenge.ipynb
└── src/
    ├── data.py
    ├── features.py
    └── models.py
```

---

## 12. How to Run

### Google Colab

* Open `notebooks/RWS_Challenge.ipynb`
* Mount Google Drive
* Set `DATA_DIR`
* Run all cells

### Local

```
pip install -r requirements.txt
```

---

## 13. Summary

This project analyzes whether short-horizon ES direction can be predicted using cross-asset features.
Results show that **lagged returns carry all measurable signal**, while volatility, correlation, and momentum features do not add predictive value.
No model meaningfully exceeds the majority baseline, underscoring the difficulty of short-horizon forecasting in liquid futures markets.

---

# If you want, I can also:

* Apply **GitHub styling** (badges, emojis, callouts)
* Add a **table of contents**
* Add **figures or plots** from your notebook
* Rewrite in **research paper style**

Just tell me!
RWS Challenge – Short-Horizon ES Direction Forecasting
Author: Laurentia Liennart
This repository contains my submission for the RWS ML Research Challenge. The project investigates whether the 30-minute ahead direction of ES (S&P 500 E-mini Futures) can be predicted using short-horizon cross-asset features.

1. Problem Overview
	•	Target asset: ES (S&P 500 E-mini futures)
	•	Data frequency: Minute-level OHLCV
	•	Prediction horizon: 30 minutes
	•	Task: Binary classification (up vs down)
The binary label is defined by shifting ES returns 30 minutes into the future:

target_ES_30m = ret_ES.shift(-30)
target_ES_30m_updown = (target_ES_30m > 0).astype(int)

2. Hypothesis
Short-term ES direction may be predictable using:
	•	1-minute lagged returns across ES, FX, and volatility indices
	•	rolling volatility (short and long windows)
	•	cross-asset rolling correlations
	•	momentum features
	•	volatility regime indicators
The hypothesis is that cross-asset structure and volatility information will improve predictive accuracy beyond simple lagged returns.

3. Data and Preprocessing
3.1 Data Sources
Data comes from the RWS challenge dataset and includes: ES, USDJPY, USDCAD, VIX, and GVZ.
3.2 Cleaning and Loading
	•	Load data by inferring asset class using symbol_metadata.csv
	•	Prefer parquet; fall back to CSV
	•	Standardize timestamps and convert to timezone-naive
	•	Sort chronologically
	•	Compute per-minute returns: pct_change()
3.3 Multi-Asset Merge
All assets are aligned on timestamp using their return columns, forming a unified modeling dataframe.

4. Feature Engineering
Lagged Returns
	•	lag_ES
	•	lag_USDJPY
	•	lag_USDCAD
	•	lag_VIX
	•	lag_GVZ
Volatility Features (Rolling Std)
	•	vol_ES_60, vol_ES_120, vol_ES_300
Cross-Asset Rolling Correlations
	•	corr_ES_VIX_60, corr_ES_VIX_300
	•	corr_ES_USDJPY_60, corr_ES_USDJPY_300
	•	corr_ES_USDCAD_60, corr_ES_USDCAD_300
Momentum Features
	•	mom_ES_5, mom_ES_10, mom_ES_30
Volatility Regime Indicator
A high-volatility regime flag based on the 70th percentile of 300-minute ES volatility.

5. Modeling Approach
Time-Respecting Validation
All experiments use TimeSeriesSplit (5 folds) to avoid leakage:
	•	Training = past data
	•	Testing = future data
	•	Same fold boundaries for all models and baselines
Models Evaluated
	•	Logistic Regression (max_iter=2000)
	•	Random Forest Classifier (200 trees, depth-controlled)

6. Baseline Models
Two baselines are used for comparison:
Baseline
Mean Accuracy
Random baseline
0.513
Majority-class baseline
0.564
The majority class baseline is strong due to mild label imbalance.

7. Ablation Study
Four feature groups are evaluated:
	1	lags_only (5 features)
	2	lags_plus_ES_vol (8 features)
	3	lags_ES_vol_plus_corr (14 features)
	4	full_features (27 features)
Ablation Results (Logistic Regression)
Feature Group
# Features
Mean Accuracy
Std Dev
lags_only
5
0.5641
0.0275
lags_plus_ES_vol
8
0.5641
0.0275
lags_ES_vol_plus_corr
14
0.5640
0.0276
full_features
27
0.5639
0.0277
Interpretation
	•	The lag-only model performs best.
	•	Adding volatility, momentum, or correlation features does not improve accuracy.
	•	Larger feature sets slightly reduce accuracy → added noise, not signal.
	•	Predictive information is contained almost entirely in 1-minute lagged returns.

8. Final Model Comparison
Model
Accuracy
Random baseline
0.513
Majority baseline
0.564
Logistic (lags only)
0.564
Logistic (full features)
0.564
Random Forest (full features)
~0.56
No ML model outperforms the majority baseline.

9. Limitations
	•	Predictive signal for 30-minute ES direction is extremely weak.
	•	Mild class imbalance makes the majority baseline difficult to beat.
	•	Only one horizon (30 minutes) and one primary asset (ES) were studied.
	•	No transaction cost modeling.
	•	Only logistic regression and random forest were evaluated.

10. Future Work
	•	Explore alternative horizons (5, 10, 60 minutes).
	•	Train models separately by volatility regime.
	•	Add more macro and cross-asset inputs.
	•	Evaluate non-linear sequence models.
	•	Use probabilistic metrics (log-loss, Brier score).

11. Repository Structure

.
├── README.md
├── notebooks/
│   └── RWS_Challenge.ipynb
├── src/
│   ├── data.py
│   ├── features.py
│   └── models.py
└── requirements.txt

12. How to Run
Google Colab
	•	Open notebooks/RWS_Challenge.ipynb
	•	Mount Google Drive
	•	Set DATA_DIR
	•	Run all cells
Local

pip install -r requirements.txt

13. Summary
The project finds that the 30-minute ahead direction of ES is extremely difficult to predict. Lagged returns contain all measurable signal; additional features do not improve performance. No model meaningfully exceeds the majority baseline, illustrating the limits of short-horizon predictability in liquid futures markets.
