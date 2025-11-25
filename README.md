Nice, I’ll auto-name the important plots and then give you a **final research-style README** that assumes you’ll put them in an `images/` folder.

---

## 1. Suggested filenames for your plots

When you download the images from Colab / this chat and add them to your repo, rename them like this:

**From the `Unknown-*.png` files (the nice, labeled ones):**

* `Unknown-6.png`  → `images/es_return_dist.png`
  *ES – Return distribution*

* `Unknown-7.png`  → `images/es_vol_60min.png`
  *ES – Rolling volatility (60-min window)*

* `Unknown-8.png`  → `images/btc_return_dist.png`
  *BTC – Return distribution*

* `Unknown-9.png`  → `images/btc_vol_60min.png`
  *BTC – Rolling volatility (60-min window)*

* `Unknown-10.png` → `images/rty_return_dist.png`
  *RTY – Return distribution*

* `Unknown-11.png` → `images/rty_vol_60min.png`
  *RTY – Rolling volatility (60-min window)*

* `Unknown-12.png` → `images/nq_return_dist.png`
  *NQ – Return distribution*

* `Unknown-13.png` → `images/nq_vol_60min.png`
  *NQ – Rolling volatility (60-min window)*

* `Unknown-14.png` → `images/cross_asset_corr_heatmap.png`
  *Cross-asset return correlation heatmap*

* `Unknown-15.png` → `images/es_vix_rolling_corr_300min.png`
  *Rolling correlation: ES vs VIX (300-minute window)*

* `Unknown-16.png` → `images/logistic_coeffs_sorted.png`
  *Logistic Regression Coefficients (Sorted)*

The `plot_cell5_*.png` / `plot_cell10_0.png` images look like earlier/raw versions of some of these plots. You **don’t need** them for the README; you can ignore or keep them as scratch images.

So in your repo you’ll have:

```text
images/
  es_return_dist.png
  es_vol_60min.png
  btc_return_dist.png
  btc_vol_60min.png
  rty_return_dist.png
  rty_vol_60min.png
  nq_return_dist.png
  nq_vol_60min.png
  cross_asset_corr_heatmap.png
  es_vix_rolling_corr_300min.png
  logistic_coeffs_sorted.png
```

---

## 2. Final research-style README with figures

Below is your final **README.md** in clean Markdown.
Paste this straight into your `README.md` (after you’ve renamed the images as above).

---

# Short-Horizon ES Direction Forecasting with Cross-Asset Features

**RWS ML Research Challenge**
**Author:** Laurentia Liennart

---

## Abstract

This project studies whether the **30-minute ahead direction** of **ES (S&P 500 E-mini futures)** can be predicted using minute-level features from ES and related cross-asset instruments (USDJPY, USDCAD, VIX, GVZ).

I construct lag, volatility, correlation, momentum, and regime features and evaluate logistic regression and random forest classifiers under **strict time-respecting cross-validation**.

The main findings are:

* Simple **lagged returns alone** achieve ≈0.564 accuracy.
* Adding volatility, correlation, and momentum features **does not improve performance**.
* No model outperforms a **majority-class baseline (0.564)**.

These results suggest that short-horizon ES direction is extremely difficult to forecast and that most engineered features add noise rather than usable signal.

---

## 1. Introduction

Short-horizon forecasting in liquid futures markets is notoriously challenging. Despite clear economic links between equity indices, FX, and volatility products, it remains unclear whether these relationships can be exploited to predict index futures over tens of minutes.

In this study, I ask:

> **Can we predict whether ES will move up or down over the next 30 minutes using cross-asset information and volatility structure?**

To answer this, I:

1. Build a unified minute-level dataset across ES, USDJPY, USDCAD, VIX, and GVZ.
2. Engineer features that capture lagged returns, volatility, cross-asset correlations, momentum, and volatility regimes.
3. Evaluate models with time-respecting cross-validation and compare to simple baselines.

---

## 2. Data

### 2.1 Instruments and frequency

* **ES** – S&P 500 E-mini futures
* **USDJPY**, **USDCAD** – FX pairs
* **VIX**, **GVZ** – volatility indices

All series are at **minute frequency**. After cleaning, they are merged on timestamp into a single dataframe containing aligned returns.

### 2.2 Return distributions

The per-minute return distributions exhibit fat tails and strong concentration near zero.

**Figure 1 – ES minute-return distribution**
![Figure 1: ES return distribution](images/es_return_dist.png)

**Figure 2 – BTC minute-return distribution**
![Figure 2: BTC return distribution](images/btc_return_dist.png)

**Figure 3 – RTY minute-return distribution**
![Figure 3: RTY return distribution](images/rty_return_dist.png)

**Figure 4 – NQ minute-return distribution**
![Figure 4: NQ return distribution](images/nq_return_dist.png)

### 2.3 Rolling volatility

Rolling volatility exhibits clear clustering across all assets.

**Figure 5 – ES 60-minute rolling volatility**
![Figure 5: ES rolling volatility (60-min window)](images/es_vol_60min.png)

**Figure 6 – BTC 60-minute rolling volatility**
![Figure 6: BTC rolling volatility (60-min window)](images/btc_vol_60min.png)

**Figure 7 – RTY 60-minute rolling volatility**
![Figure 7: RTY rolling volatility (60-min window)](images/rty_vol_60min.png)

**Figure 8 – NQ 60-minute rolling volatility**
![Figure 8: NQ rolling volatility (60-min window)](images/nq_vol_60min.png)

---

## 3. Cross-Asset Structure

To understand relationships between assets, I examine both static and rolling correlations.

**Figure 9 – Cross-asset return correlation matrix**
![Figure 9: Cross-asset return correlation](images/cross_asset_corr_heatmap.png)

ES shows:

* Mild negative correlation with VIX.
* Weak but non-zero relationships with FX pairs.

I also examine the time-varying ES–VIX relationship with rolling correlations.

**Figure 10 – Rolling correlation between ES and VIX (300-minute window)**
![Figure 10: Rolling correlation ES vs VIX](images/es_vix_rolling_corr_300min.png)

The correlation fluctuates over time, but as later results show, these fluctuations do not translate into improved predictive power for 30-minute direction.

---

## 4. Problem Formulation and Features

### 4.1 Target label

Let ( r^{ES}_t ) denote the per-minute ES return.
The 30-minute ahead cumulative return is:

[
r^{ES}*{t:t+30} = \sum*{k=1}^{30} r^{ES}_{t+k}
]

The binary label is:

* `1` if ( r^{ES}_{t:t+30} > 0 ) (up),
* `0` otherwise (down).

In code:

```python
combined["target_ES_30m"] = combined["ret_ES"].shift(-30)
combined["target_ES_30m_updown"] = (combined["target_ES_30m"] > 0).astype(int)
```

### 4.2 Feature families

I construct the following feature groups:

* **Lagged returns**

  * `lag_ES`, `lag_USDJPY`, `lag_USDCAD`, `lag_VIX`, `lag_GVZ`
* **Rolling volatility**

  * `vol_ES_60`, `vol_ES_120`, `vol_ES_300`
  * Equivalent vol features for FX and volatility indices in the full feature set
* **Cross-asset rolling correlations**

  * `corr_ES_VIX_60`, `corr_ES_VIX_300`
  * `corr_ES_USDJPY_60`, `corr_ES_USDJPY_300`
  * `corr_ES_USDCAD_60`, `corr_ES_USDCAD_300`
* **Momentum features**

  * `mom_ES_5`, `mom_ES_10`, `mom_ES_30`
* **Volatility regime indicator**

  * `regime_high_vol` based on the 70th percentile of `vol_ES_300`

These features are combined into four groups used in the ablation study:

1. `lags_only`
2. `lags_plus_ES_vol`
3. `lags_ES_vol_plus_corr`
4. `full_features`

---

## 5. Experimental Setup

### 5.1 Models

I evaluate two model families:

* **Logistic Regression** (`max_iter = 2000`)
* **Random Forest Classifier**

  * `n_estimators = 200`
  * `max_depth = 8`
  * `min_samples_split = 50`
  * `min_samples_leaf = 25`

### 5.2 Time-respecting cross-validation

To avoid look-ahead bias, I use **TimeSeriesSplit** with 5 folds:

* Each fold trains on a contiguous block of **past** data.
* Evaluation is on the subsequent **future** block.
* The same folds are used for baselines and all models.

---

## 6. Baseline Performance

Two baselines are constructed:

* **Random baseline** – predicts labels according to the training-set class distribution.
* **Majority baseline** – always predicts the most frequent class in the training set.

| Baseline       | Mean Accuracy |
| -------------- | ------------- |
| Random         | 0.513         |
| Majority class | **0.564**     |

The majority baseline is surprisingly strong due to mild class imbalance in the direction label.

---

## 7. Model Results and Feature Importance

### 7.1 Overall model performance

Across all feature sets and models:

* Logistic regression accuracies are ≈0.564.
* Random forest accuracies are ≈0.56.
* Neither model significantly exceeds the majority baseline.

### 7.2 Logistic regression coefficients

To interpret the linear model, I inspect the sorted coefficients of the logistic regression fitted on the full feature set.

**Figure 11 – Logistic regression coefficients (full features, sorted)**
![Figure 11: Logistic regression coefficients (sorted)](images/logistic_coeffs_sorted.png)

The largest-magnitude coefficients correspond to:

* Long-window cross-asset correlations (`corr_ES_USDCAD_300`, `corr_ES_VIX_300`, `corr_ES_USDJPY_300`)
* The high-volatility regime indicator

However, even with these features, predictive accuracy does not improve beyond the lag-only model, suggesting that their apparent importance does not translate into a usable edge.

---

## 8. Ablation Study

To quantify the contribution of each feature family, I run logistic regression with different feature groups:

| Feature Group           | # Features | Mean Accuracy | Std Dev |
| ----------------------- | ---------: | ------------: | ------: |
| `lags_only`             |          5 |    **0.5641** |  0.0275 |
| `lags_plus_ES_vol`      |          8 |    **0.5641** |  0.0275 |
| `lags_ES_vol_plus_corr` |         14 |        0.5640 |  0.0276 |
| `full_features`         |         27 |        0.5639 |  0.0277 |

### Interpretation

* The **simplest model (lags only)** performs best.
* Adding ES volatility features **does not change accuracy**.
* Adding correlation and momentum features **does not improve accuracy** and slightly increases variance.
* The full feature set is marginally worse, consistent with the idea that most added features are noise at this horizon.

---

## 9. Discussion

The empirical results support a pessimistic view of short-horizon predictability in ES:

* Cross-asset correlations and volatility structure are intuitively appealing, but they do **not** provide measurable predictive power for 30-minute direction beyond what is already captured by simple lagged returns.
* The tiny performance differences across feature sets indicate that the signal-to-noise ratio is extremely low.
* The fact that the **majority baseline** matches or slightly exceeds the ML models underscores the importance of always benchmarking against simple baselines.

---

## 10. Limitations

This study has several limitations:

* Only a **single horizon** (30 minutes) and primary asset (ES) are considered.
* No transaction costs, slippage, or position sizing rules are modeled, so even small edges would not necessarily be tradable.
* Only logistic regression and random forest classifiers are evaluated; more sophisticated sequence models might capture different structure.
* Feature engineering is restricted to simple statistical features rather than full limit-order-book or microstructure data.

---

## 11. Future Work

Future extensions could explore:

* Multiple horizons (5, 10, 60 minutes) to see where any signal is strongest.
* Regime-specific models trained separately on high- and low-volatility regimes.
* Inclusion of more macro futures (rates, commodities) and sector indices.
* Non-linear and sequence models such as gradient boosting, LSTMs, or Transformers.
* Evaluation using probabilistic scoring rules (log loss, Brier score) rather than pure accuracy.

---

## 12. Repository Structure

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

## 13. How to Run

### In Google Colab

1. Open `notebooks/RWS_Challenge.ipynb`.
2. Mount Google Drive and set `DATA_DIR` to the RWS dataset location.
3. Run all cells in order.

### Locally

```bash
pip install -r requirements.txt
```

Then open the notebook with Jupyter or VS Code and run the analysis.

---
