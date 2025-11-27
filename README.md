# Short-Horizon ES Direction Forecasting with Cross-Asset Features

**RWS ML Research Challenge**
**Author:** Laurentia Liennart

---

## Abstract

This project investigates whether the **30-minute ahead direction** of ES (S&P 500 E-mini Futures) can be predicted using minute-level financial features. I analyze return, volatility, and cross-asset structure, engineer multiple feature families, and evaluate logistic regression and random forest models using time-respecting cross-validation.

Results show that:

* **Lagged returns alone** achieve ~0.564 accuracy
* **No feature set outperforms the majority baseline**
* Cross-asset correlations, volatility, and momentum do **not** improve predictability

This highlights the difficulty of short-horizon forecasting in liquid futures markets.

---

# 1. Introduction

Equity index futures such as ES exhibit rapid mean reversion, noise, and short-lived autocorrelation, making directional prediction extremely challenging.

This study asks:

**Can we predict whether ES will move up or down 30 minutes ahead using cross-asset information?**

I construct a unified dataset across ES, FX pairs (USDJPY, USDCAD), and volatility indices (VIX, GVZ), engineer predictive features, and benchmark statistical models against simple baselines.

---

# 2. Data

## 2.1 Instruments and Sampling

* ES — S&P 500 E-mini futures
* USDJPY, USDCAD — FX
* VIX, GVZ — volatility indices

Data is sampled at **1-minute frequency** and merged on timestamp.

---

## 2.2 Return Distributions

The per-minute returns exhibit strong concentration around zero and fat tails.

**Figure 1 — ES return distribution**
![Figure 1](./images/es_return_dist.png)

**Figure 2 — BTC return distribution**
![Figure 2](./images/btc_return_dist.png)

**Figure 3 — RTY return distribution**
![Figure 3](./images/rty_return_dist.png)

**Figure 4 — NQ return distribution**
![Figure 4](./images/nq_return_dist.png)

---

## 2.3 Rolling Volatility

Volatility clusters strongly across all markets.

**Figure 5 — ES rolling volatility (60-minute window)**
![Figure 5](./images/es_vol_60min.png)

**Figure 6 — BTC rolling volatility**
![Figure 6](./images/btc_vol_60min.png)

**Figure 7 — RTY rolling volatility**
![Figure 7](./images/rty_vol_60min.png)

**Figure 8 — NQ rolling volatility**
![Figure 8](./images/nq_vol_60min.png)

---

# 3. Cross-Asset Structure

## 3.1 Static Correlation Structure

**Figure 9 — Cross-asset correlation matrix**
![Figure 9](./images/cross_asset_corr_heatmap.png)

ES exhibits:

* Mild negative correlation with VIX
* Weak correlation with FX pairs
* Weak but non-zero correlation with GVZ

## 3.2 Time-Varying Correlation

**Figure 10 — Rolling correlation between ES and VIX (300-min window)**
![Figure 10](./images/es_vix_rolling_corr_300min.png)

---

# 4. Problem Formulation and Features

## 4.1 Target Label (Fixed Math)

Let **r_ES(t)** denote the per-minute return of ES at time *t*.

The **30-minute ahead return** is computed by shifting ES returns backward by 30 minutes:

```python
combined["target_ES_30m"] = combined["ret_ES"].shift(-30)
```

We convert this to a **binary classification label**:

* **1** → ES goes **up** over the next 30 minutes
* **0** → ES goes **down or stays flat**

```python
combined["target_ES_30m_updown"] = (combined["target_ES_30m"] > 0).astype(int)
```

This label serves as the target for all predictive models.

---

## 4.2 Feature Families

### **Lagged Returns**

Captures immediate past movement:

* `lag_ES`, `lag_USDJPY`, `lag_USDCAD`, `lag_VIX`, `lag_GVZ`

### **Volatility**

Rolling standard deviation over multiple windows:

* `vol_ES_60`, `vol_ES_120`, `vol_ES_300`
* Similar features for FX and volatility indices (in the full feature set)

### **Cross-Asset Correlations**

Measures how ES co-moves with VIX and major FX pairs:

* `corr_ES_VIX_60`, `corr_ES_VIX_300`
* `corr_ES_USDJPY_60`, `corr_ES_USDJPY_300`
* `corr_ES_USDCAD_60`, `corr_ES_USDCAD_300`

### **Momentum**

Short-term cumulative returns:

* `mom_ES_5`, `mom_ES_10`, `mom_ES_30`

### **Volatility Regime Indicator**

Binary regime = 1 if ES volatility over the last 300 minutes is in the top 30%.

---

# 5. Experimental Setup

## 5.1 Models

* **Logistic Regression** (linear baseline)
* **Random Forest Classifier** (nonlinear benchmark)

## 5.2 Time-Respecting Cross-Validation

I use **TimeSeriesSplit (5 folds)**:

* Train on **past**
* Test on **future**
* No leakage
* Same folds for all models and baselines

---

# 6. Baseline Performance

| Baseline          | Accuracy  |
| ----------------- | --------- |
| Random baseline   | 0.513     |
| Majority baseline | **0.564** |

The majority class is slightly more common, making it a strong baseline.

---

# 7. Model Results and Feature Importance

## 7.1 Logistic Regression (full features)

**Figure 11 — Logistic regression coefficients (sorted)**
![Figure 11](./images/logistic_coeffs_sorted.png)

Even with all engineered features, logistic regression does **not** outperform the lag-only model or majority baseline.

## 7.2 Random Forest Feature Importance

(Random forest plots can be added here if desired.)

---

# 8. Ablation Study

| Feature Group           | # Features | Mean Accuracy | Std Dev |
| ----------------------- | ---------: | ------------: | ------: |
| `lags_only`             |          5 |    **0.5641** |  0.0275 |
| `lags_plus_ES_vol`      |          8 |    **0.5641** |  0.0275 |
| `lags_ES_vol_plus_corr` |         14 |        0.5640 |  0.0276 |
| `full_features`         |         27 |        0.5639 |  0.0277 |

### Interpretation

* The **lag-only model performs best**
* Volatility, correlation, and momentum features provide **no** incremental value
* Larger feature sets introduce noise

---

# 9. Discussion

The 30-minute horizon exhibits **extremely low signal-to-noise ratio**.
Despite strong economic intuition linking equity indices, FX, and volatility:

* Cross-asset features do **not** enhance forecasting performance
* Short-term ES movement remains effectively unpredictable
* Even sophisticated features cannot beat a majority baseline

---

# 10. Limitations

* Only ES and 30-minute horizon tested
* No transaction cost modeling
* Only two model families (LogReg, RF)
* Features are return-based, not microstructure-based

---

# 11. Future Work

* Evaluate multiple horizons (5, 10, 60 minutes)
* Add macro futures (ZN, CL, GC), sector indices
* Explore nonlinear temporal models (LSTM, Transformers)
* Use probabilistic scoring metrics (log loss, Brier score)
* Try regime-specific models

---

# 12. Repository Structure

```text
.
├── README.md
├── requirements.txt
├── notebooks/
│   └── RWS_Challenge.ipynb
├── src/
│   ├── data.py
│   ├── features.py
│   └── models.py
└── images/
    ├── es_return_dist.png
    ├── es_vol_60min.png
    ├── btc_return_dist.png
    ├── btc_vol_60min.png
    ├── rty_return_dist.png
    ├── rty_vol_60min.png
    ├── nq_return_dist.png
    ├── nq_vol_60min.png
    ├── cross_asset_corr_heatmap.png
    ├── es_vix_rolling_corr_300min.png
    └── logistic_coeffs_sorted.png
```

---

# 13. How to Run

## In Google Colab

Open the notebook and set the data path:

```python
DATA_DIR = "/content/drive/MyDrive/project_data"
```

Then run all cells.

## Locally

```
pip install -r requirements.txt
```

Open the notebook in Jupyter or VS Code.

---
