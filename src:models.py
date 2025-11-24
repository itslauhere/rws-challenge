{\rtf1\ansi\ansicpg1252\cocoartf2818
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import numpy as np\
import pandas as pd\
from sklearn.model_selection import TimeSeriesSplit\
from sklearn.metrics import accuracy_score\
from sklearn.linear_model import LogisticRegression\
from sklearn.ensemble import RandomForestClassifier\
\
\
def eval_baselines(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> pd.DataFrame:\
    """\
    Evaluate random and majority-class baselines using TimeSeriesSplit.\
    Returns a DataFrame with per-fold accuracies.\
    """\
    tscv = TimeSeriesSplit(n_splits=n_splits)\
    results = []\
\
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):\
        y_train = y.iloc[train_idx]\
        y_test = y.iloc[test_idx]\
\
        # Majority baseline\
        majority_class = y_train.mode()[0]\
        majority_preds = np.full_like(y_test, fill_value=majority_class)\
        majority_acc = accuracy_score(y_test, majority_preds)\
\
        # Random baseline (respecting class distribution)\
        class_probs = y_train.value_counts(normalize=True)\
        rng = np.random.RandomState(42 + fold)\
        random_preds = rng.choice(\
            class_probs.index, size=len(y_test), p=class_probs.values\
        )\
        random_acc = accuracy_score(y_test, random_preds)\
\
        results.append(\
            \{\
                "fold": fold,\
                "majority_acc": majority_acc,\
                "random_acc": random_acc,\
            \}\
        )\
\
    return pd.DataFrame(results)\
\
\
def eval_logistic(\
    X: pd.DataFrame, y: pd.Series, n_splits: int = 5\
) -> (pd.DataFrame, list):\
    """\
    Evaluate a logistic regression model with TimeSeriesSplit.\
    Returns (results_df, models_list).\
    """\
    tscv = TimeSeriesSplit(n_splits=n_splits)\
    accuracies = []\
    models = []\
\
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):\
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]\
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]\
\
        clf = LogisticRegression(max_iter=2000)\
        clf.fit(X_train, y_train)\
        preds = clf.predict(X_test)\
        acc = accuracy_score(y_test, preds)\
        accuracies.append(acc)\
        models.append(clf)\
\
    results_df = pd.DataFrame(\
        \{"fold": np.arange(1, n_splits + 1), "accuracy": accuracies\}\
    )\
    return results_df, models\
\
\
def eval_random_forest(\
    X: pd.DataFrame, y: pd.Series, n_splits: int = 5\
) -> (pd.DataFrame, list):\
    """\
    Evaluate a RandomForest classifier with TimeSeriesSplit.\
    Returns (results_df, models_list).\
    """\
    tscv = TimeSeriesSplit(n_splits=n_splits)\
    accuracies = []\
    models = []\
\
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):\
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]\
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]\
\
        rf = RandomForestClassifier(\
            n_estimators=200,\
            max_depth=8,\
            min_samples_split=50,\
            min_samples_leaf=25,\
            random_state=42,\
            n_jobs=-1,\
        )\
\
        rf.fit(X_train, y_train)\
        preds = rf.predict(X_test)\
        acc = accuracy_score(y_test, preds)\
        accuracies.append(acc)\
        models.append(rf)\
\
    results_df = pd.DataFrame(\
        \{"fold": np.arange(1, n_splits + 1), "accuracy": accuracies\}\
    )\
    return results_df, models\
\
\
def eval_logistic_ablation(\
    df_model: pd.DataFrame, target_col: str, feature_groups: dict, n_splits: int = 5\
) -> pd.DataFrame:\
    """\
    Run logistic regression for multiple feature groups (ablation study).\
    feature_groups: dict name -> list of feature names.\
    Returns a DataFrame summarizing mean/std accuracy per group.\
    """\
    rows = []\
\
    for group_name, feats in feature_groups.items():\
        df_group = df_model.dropna(subset=feats + [target_col])\
        X_sub = df_group[feats]\
        y_sub = df_group[target_col]\
\
        tscv = TimeSeriesSplit(n_splits=n_splits)\
        accs = []\
\
        for train_idx, test_idx in tscv.split(X_sub):\
            X_train, X_test = X_sub.iloc[train_idx], X_sub.iloc[test_idx]\
            y_train, y_test = y_sub.iloc[train_idx], y_sub.iloc[test_idx]\
\
            clf = LogisticRegression(max_iter=2000)\
            clf.fit(X_train, y_train)\
            preds = clf.predict(X_test)\
            accs.append(accuracy_score(y_test, preds))\
\
        rows.append(\
            \{\
                "group": group_name,\
                "num_features": len(feats),\
                "mean_accuracy": float(np.mean(accs)),\
                "std_accuracy": float(np.std(accs)),\
            \}\
        )\
\
    return pd.DataFrame(rows)\
}