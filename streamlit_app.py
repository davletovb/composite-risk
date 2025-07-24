# composite_risk_app.py – v3.2 (LightGBM & CatBoost + extra metrics)
"""Composite early‑warning dashboard with pluggable ML models.

New in v3.2
============
1. **Added LightGBM and CatBoost options** (appear in sidebar if libraries are
   installed). App silently hides them if imports fail.
2. **Extra evaluation metrics** shown below the gauge: AUC‑ROC (in‑sample) and
   Brier score for the selected model; Ensemble shows average values.
3. Model cache keyed on model set + horizon so new models don’t clash with old
   pickles.

Install optional libs – any that fail will simply disable that choice:
```bash
pip install --upgrade lightgbm catboost
```
Then run:
```bash
streamlit run composite_risk_app.py
```
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from dateutil.relativedelta import relativedelta
from fredapi import Fred
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
from xgboost import XGBClassifier

# Optional models
try:
    from lightgbm import LGBMClassifier  # type: ignore
except ImportError:
    LGBMClassifier = None  # type: ignore
try:
    from catboost import CatBoostClassifier  # type: ignore
except ImportError:
    CatBoostClassifier = None  # type: ignore

try:
    import shap  # optional for importance
except ImportError:
    shap = None

# ────────────────────────────────────────────────────────────────────────────────
# Config & constants
# ────────────────────────────────────────────────────────────────────────────────
CACHE_DIR = Path(".cache"); CACHE_DIR.mkdir(exist_ok=True)
FRED_API_KEY = os.getenv("FRED_API_KEY")
START_DATE = "1960-01-01"
DEFAULT_HORIZON = 12

FRED_SERIES = {
    "DGS10": "10‑Year Treasury Constant Maturity Rate",
    "TB3MS": "3‑Month Treasury Bill Secondary Market Rate",
    "BAA": "Moody's Seasoned Baa Corporate Bond Yield",
    "AAA": "Moody's Seasoned Aaa Corporate Bond Yield",
    "UNRATE": "Unemployment Rate",
    "VIXCLS": "CBOE VIX Close",
    "USRECD": "NBER Recession Indicator",
    "SP500": "S&P 500 Index (Daily Close)",
    "USSLIND": "Philly Fed Leading Index",
    "NFCI": "Chicago Fed NFCI",
    "ANFCI": "Chicago Fed Adjusted NFCI",
}

st.set_page_config(page_title="Composite Risk Gauge", layout="wide")

# ────────────────────────────────────────────────────────────────────────────────
# Data helpers
# ────────────────────────────────────────────────────────────────────────────────

def load_fred(series_map: dict[str, str]) -> pd.DataFrame:
    fred = Fred(api_key=FRED_API_KEY)
    data, skipped = {}, []
    for sid, name in series_map.items():
        try:
            data[name] = fred.get_series(sid, observation_start=START_DATE)
        except ValueError:
            skipped.append(sid)
    if skipped:
        st.warning("Skipped missing series: " + ", ".join(skipped))
    df = pd.DataFrame(data)
    df.index = pd.to_datetime(df.index)
    return df.resample("M").last().ffill()


def engineer_features(raw: pd.DataFrame, horizon: int) -> pd.DataFrame:
    feats = pd.DataFrame(index=raw.index)
    feats["term_spread"] = raw[FRED_SERIES["DGS10"]] - raw[FRED_SERIES["TB3MS"]]
    feats["credit_spread"] = raw[FRED_SERIES["BAA"]] - raw[FRED_SERIES["AAA"]]
    feats["vix"] = raw[FRED_SERIES["VIXCLS"]]
    rolling_min_u = raw[FRED_SERIES["UNRATE"]].rolling(12, min_periods=1).min()
    feats["sahm_rule"] = raw[FRED_SERIES["UNRATE"]] - rolling_min_u
    feats["lei_6m_pct"] = raw[FRED_SERIES["USSLIND"]].pct_change(6)
    feats["nfci"] = raw[FRED_SERIES["NFCI"]]
    feats["an_fci"] = raw[FRED_SERIES["ANFCI"]]

    base_cols = ["term_spread", "credit_spread", "vix", "sahm_rule", "lei_6m_pct", "nfci"]
    for col in base_cols:
        feats[f"{col}_chg6"] = feats[col] - feats[col].shift(6)

    z = (feats - feats.expanding(60).mean()) / feats.expanding(60).std(ddof=0)
    feats = pd.concat([feats, z.add_prefix("z_")], axis=1)
    feats.replace([np.inf, -np.inf], np.nan, inplace=True)
    feats.dropna(inplace=True)

    fut_sp = raw[FRED_SERIES["SP500"]].resample("M").last()
    fut_ret = fut_sp.pct_change(horizon)
    rec = raw[FRED_SERIES["USRECD"]].shift(-horizon).rolling(horizon).max()
    feats["target"] = ((rec > 0) | (fut_ret <= -0.20)).astype(int)
    return feats

# ────────────────────────────────────────────────────────────────────────────────
# Model factory
# ────────────────────────────────────────────────────────────────────────────────

def build_models(X: pd.DataFrame, y: pd.Series, choices: list[str]):
    models: dict[str, object] = {}
    if "Logit" in choices:
        models["Logit"] = Pipeline([
            ("std", StandardScaler()),
            ("clf", LogisticRegression(max_iter=600, solver="liblinear", penalty="l1")),
        ]).fit(X, y)

    if "Random Forest" in choices:
        models["Random Forest"] = RandomForestClassifier(
            n_estimators=400, max_depth=6, random_state=42, n_jobs=-1
        ).fit(X, y)

    if "Gradient Boosting" in choices:
        models["Gradient Boosting"] = GradientBoostingClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=3, random_state=42
        ).fit(X, y)

    if "XGBoost" in choices:
        models["XGBoost"] = XGBClassifier(
            n_estimators=400, learning_rate=0.03, max_depth=3, subsample=0.8,
            colsample_bytree=0.8, eval_metric="logloss", random_state=42, n_jobs=4
        ).fit(X, y)

    if "LightGBM" in choices and LGBMClassifier is not None:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=400, learning_rate=0.03, max_depth=-1, subsample=0.8,
            colsample_bytree=0.8, objective="binary", random_state=42
        ).fit(X, y)
    elif "LightGBM" in choices:
        st.warning("LightGBM not installed – skipping.")

    if "CatBoost" in choices and CatBoostClassifier is not None:
        models["CatBoost"] = CatBoostClassifier(
            iterations=400, learning_rate=0.03, depth=4, loss_function="Logloss", verbose=False
        ).fit(X, y)
    elif "CatBoost" in choices:
        st.warning("CatBoost not installed – skipping.")

    return models


def predict_prob(models: dict[str, object], X: pd.DataFrame, selection: str) -> np.ndarray:
    if selection == "Ensemble":
        probs = np.column_stack([m.predict_proba(X)[:, 1] for m in models.values()])
        return probs.mean(axis=1)
    return models[selection].predict_proba(X)[:, 1]


def get_metrics(y_true: pd.Series, probs: np.ndarray):
    return {
        "AUC": roc_auc_score(y_true, probs),
        "Brier": brier_score_loss(y_true, probs),
    }

# ────────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ────────────────────────────────────────────────────────────────────────────────

def main():
    st.title("🛡️ Composite Market‑Risk Gauge (v3.2)")
    st.caption("Early‑warning probabilities with interchangeable ML models.")

    # Sidebar controls
    horizon = st.sidebar.slider("Look‑ahead horizon (months)", 6, 18, DEFAULT_HORIZON, 3)
    model_options = ["Logit", "Random Forest", "Gradient Boosting", "XGBoost"]
    if LGBMClassifier is not None:
        model_options.append("LightGBM")
    if CatBoostClassifier is not None:
        model_options.append("CatBoost")

    model_options.append("Ensemble")

    model_choice = st.sidebar.selectbox("Model", model_options, index=model_options.index("Ensemble"))
    alert_thr = st.sidebar.slider("Alert threshold", 0.1, 0.9, 0.45, 0.05)

    # --- Data & feature set
    raw = load_fred(FRED_SERIES)
    feats = engineer_features(raw, horizon)
    X, y = feats.drop(columns=["target"]), feats["target"]
    cutoff = feats.index[-1] - relativedelta(years=5)
    X_tr, y_tr = X[X.index < cutoff], y[y.index < cutoff]

    # --- Model caching keyed by horizon + choice set
    choices = model_options if model_choice == "Ensemble" else [model_choice]
    cache_key = f"{horizon}_{'-'.join(sorted(choices))}.pkl"
    cache_file = CACHE_DIR / cache_key
    if cache_file.exists():
        models = joblib.load(cache_file)
    else:
        models = build_models(X_tr, y_tr, choices)
        joblib.dump(models, cache_file)

    # --- Predictions & metrics
    latest_prob = float(predict_prob(models, X.iloc[-1:], model_choice)[0])
    hist_probs = predict_prob(models, X, model_choice)
    metrics = get_metrics(y, hist_probs)

    # --- Top widgets
    st.metric("Current probability", f"{latest_prob:.1%}")
    st.progress(int(latest_prob * 100))
    st.caption(f"AUC‑ROC {metrics['AUC']:.2f}   |   Brier {metrics['Brier']:.3f}")
    if latest_prob >= alert_thr:
        st.error(f"⚠️ Prob {latest_prob:.1%} ≥ {alert_thr:.0%}")

    # --- Tabs
    tab_hist, tab_feat, tab_imp = st.tabs(["Probability history", "Feature snapshot", "Importance"])

    with tab_hist:
        st.line_chart(pd.Series(hist_probs, index=feats.index, name="prob"))
        csv = pd.Series(hist_probs, index=feats.index, name="prob").to_csv().encode()
        st.download_button("Download CSV", csv, "risk_prob.csv", "text/csv")

    with tab_feat:
        st.dataframe(X.iloc[-1:].T)

    with tab_imp:
        st.write(f"Feature importance – {model_choice}")
        if model_choice in models:
            model = models[model_choice]
        else:
            st.write("Ensemble importance varies per model. Select an individual model to view details.")
            st.stop()

        if isinstance(model, (RandomForestClassifier, GradientBoostingClassifier)):
            st.bar_chart(pd.Series(model.feature_importances_, index=X.columns))
        elif isinstance(model, XGBClassifier):
            if shap:
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(X_tr, check_additivity=False)
                shap_sum = pd.Series(np.abs(shap_vals).mean(axis=0), index=X.columns)
                st.bar_chart(shap_sum.sort_values(ascending=False))
            else:
                gain = model.get_booster().get_score(importance_type="gain")
                st.bar_chart(pd.Series(gain).reindex(X.columns).fillna(0))
        elif LGBMClassifier is not None and isinstance(model, LGBMClassifier):
            st.bar_chart(pd.Series(model.feature_importances_, index=X.columns))
        elif CatBoostClassifier is not None and isinstance(model, CatBoostClassifier):
            imp = model.get_feature_importance(type="FeatureImportance")
            st.bar_chart(pd.Series(imp, index=X.columns))
        elif isinstance(model, Pipeline):  # Logit
            coefs = model.named_steps["clf"].coef_[0]
            st.bar_chart(pd.Series(np.abs(coefs), index=X.columns))

    st.caption("Data: FRED.  Code MIT‑licensed.")

if __name__ == "__main__":
    main()
