# composite_risk_app.py – v4.3 (Scenario Lab ++)
"""Unified UI with **four tabs** *plus* rich Scenario Lab:
• Pre‑baked shocks (Dot‑com, Covid‑flash)  • Monte‑Carlo fan chart  • Reverse stress‑test.
Also squashes the pandas ‘M’ deprecation by switching to ‘ME’ resampling.
"""
from __future__ import annotations
import os, numpy as np, pandas as pd, streamlit as st
from datetime import datetime
from pathlib import Path
from fredapi import Fred
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from scipy.optimize import minimize
try: from lightgbm import LGBMClassifier
except ImportError: LGBMClassifier=None
try: from catboost import CatBoostClassifier
except ImportError: CatBoostClassifier=None

CACHE_DIR=Path(".cache");CACHE_DIR.mkdir(exist_ok=True)
FRED_API_KEY=os.getenv("FRED_API_KEY")
START_DATE="1960-01-01"; DEFAULT_H=12
FRED_SERIES={"DGS10":"10‑Year Treasury Constant Maturity Rate","TB3MS":"3‑Month Treasury Bill Secondary Market Rate","BAA":"Moody's Seasoned Baa Corporate Bond Yield","AAA":"Moody's Seasoned Aaa Corporate Bond Yield","UNRATE":"Unemployment Rate","VIXCLS":"CBOE VIX Close","USRECD":"NBER Recession Indicator","SP500":"S&P 500 Index (Daily Close)","USSLIND":"Philly Fed Leading Index","NFCI":"Chicago Fed NFCI","ANFCI":"Chicago Fed Adjusted NFCI"}

st.set_page_config(page_title="Composite Risk Gauge",layout="wide")

# ⬇ Data helpers ----------------------------------------------------------------

def load_fred():
    fred=Fred(api_key=FRED_API_KEY)
    df=pd.DataFrame({n:fred.get_series(s,START_DATE) for s,n in FRED_SERIES.items()})
    df.index=pd.to_datetime(df.index)
    # use month‑end ('ME') to silence future warning
    return df.resample("ME").last().ffill()


def build_features(raw: pd.DataFrame, h: int):
    f=pd.DataFrame(index=raw.index)
    f["term_spread"]=raw[FRED_SERIES["DGS10"]]-raw[FRED_SERIES["TB3MS"]]
    f["credit_spread"]=raw[FRED_SERIES["BAA"]]-raw[FRED_SERIES["AAA"]]
    f["vix"]=raw[FRED_SERIES["VIXCLS"]]
    f["sahm_rule"]=raw[FRED_SERIES["UNRATE"]]-raw[FRED_SERIES["UNRATE"]].rolling(12,1).min()
    f["lei_6m_pct"]=raw[FRED_SERIES["USSLIND"]].pct_change(6)
    f["nfci"]=raw[FRED_SERIES["NFCI"]]
    base=["term_spread","credit_spread","vix","sahm_rule","lei_6m_pct","nfci"]
    for c in base: f[f"{c}_chg6"]=f[c]-f[c].shift(6)
    z=(f-f.expanding(60).mean())/f.expanding(60).std(ddof=0)
    f=pd.concat([f,z.add_prefix("z_")],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
    fut=raw[FRED_SERIES["SP500"]].resample("ME").last().pct_change(h)
    rec=raw[FRED_SERIES["USRECD"]].shift(-h).rolling(h).max()
    f["target"]=((rec>0)|(fut<=-0.20)).astype(int)
    return f

# ⬇ Model helpers ----------------------------------------------------------------

def train_models(X,y,names):
    m={}
    if "Logit" in names: m["Logit"]=Pipeline([("std",StandardScaler()),("clf",LogisticRegression(max_iter=600,solver="liblinear",penalty="l1"))]).fit(X,y)
    if "Random Forest" in names: m["Random Forest"]=RandomForestClassifier(400,max_depth=6,random_state=42).fit(X,y)
    if "Gradient Boosting" in names: m["Gradient Boosting"]=GradientBoostingClassifier(n_estimators=400,learning_rate=0.05,max_depth=3,random_state=42).fit(X,y)
    if "XGBoost" in names: m["XGBoost"]=XGBClassifier(n_estimators=400,learning_rate=0.03,max_depth=3,subsample=0.8,colsample_bytree=0.8,eval_metric="logloss",random_state=42).fit(X,y)
    if LGBMClassifier and "LightGBM" in names: m["LightGBM"]=LGBMClassifier(n_estimators=400,learning_rate=0.03,subsample=0.8,colsample_bytree=0.8).fit(X,y)
    if CatBoostClassifier and "CatBoost" in names: m["CatBoost"]=CatBoostClassifier(iterations=400,learning_rate=0.03,depth=4,loss_function="Logloss",verbose=False).fit(X,y)
    return m

def get_prob(m,X,sel):
    return np.mean([md.predict_proba(X)[:,1] for md in m.values()],axis=0) if sel=="Ensemble" else m[sel].predict_proba(X)[:,1]

# ⬇ Scenario functions -----------------------------------------------------------

def dotcom(s):
    s=s.copy(); s["term_spread"]+=1; s["credit_spread"]+=1.5; s["vix"]+=10; s["nfci"]+=0.5; return s

def covid(s):
    s=s.copy(); s["term_spread"]+=0.8; s["credit_spread"]+=3; s["vix"]+=40; s["nfci"]+=1.2; return s

SCENARIOS={"Dot‑com 2001":dotcom,"Covid‑flash 2020":covid}

# Monte Carlo helper -------------------------------------------------------------

def monte_carlo(base_row: pd.Series, cov: pd.DataFrame, n:int):
    draws=np.random.multivariate_normal(np.zeros(len(base_row)),cov.values,size=n)
    sims=base_row.values+draws
    return pd.DataFrame(sims,columns=base_row.index)

# Reverse stress -----------------------------------------------------------------

def min_shock(base_row, model, threshold):
    x0=np.zeros(len(base_row))
    bounds=[(-3,3)]*len(base_row)
    def obj(d):
        p=model.predict_proba(pd.DataFrame([base_row+d]))[0,1]
        return np.sum(np.abs(d))+100*max(0,threshold-p)**2
    res=minimize(obj,x0,bounds=bounds,options={"maxiter":500})
    return base_row+res.x, model.predict_proba(pd.DataFrame([base_row+res.x]))[0,1]

# ⬇ UI ---------------------------------------------------------------------------

def main():
    raw=load_fred(); horizon=st.sidebar.slider("Forecast horizon",6,18,DEFAULT_H,3)
    df=build_features(raw,horizon); X,y=df.drop(columns=["target"]),df["target"]
    split=df.index[-1]-relativedelta(years=5); Xtr,ytr=X[X.index<split],y[y.index<split]
    base_models=["Logit","Random Forest","Gradient Boosting","XGBoost"]+(["LightGBM"] if LGBMClassifier else [])+(["CatBoost"] if CatBoostClassifier else [])
    model_sel=st.sidebar.selectbox("Model",base_models+["Ensemble"],index=len(base_models))
    models=train_models(Xtr,ytr,base_models if model_sel=="Ensemble" else [model_sel])
    latest=X.iloc[[-1]]; p_now=get_prob(models,latest,model_sel)[0]

    st.title("🛡️ Composite Risk Gauge v4.3")
    st.metric("Current probability",f"{p_now:.1%}")

    tab_hist,tab_feat,tab_imp,tab_scen=st.tabs(["Probability history","Feature snapshot","Importance","Scenario Lab"])

    with tab_hist:
        st.line_chart(pd.Series(get_prob(models,X,model_sel),index=df.index,name="prob"))
    with tab_feat:
        st.dataframe(latest.T)
    with tab_imp:
        if model_sel!="Ensemble":
            m=models[model_sel]
            if hasattr(m,"feature_importances_"): st.bar_chart(pd.Series(m.feature_importances_,index=X.columns))
            elif isinstance(m,Pipeline): st.bar_chart(pd.Series(np.abs(m.named_steps["clf"].coef_[0]),index=X.columns))
        else: st.write("Select single model for importance")

    with tab_scen:
        st.subheader("Scenario Lab")
        mode=st.radio("Mode",["Pre‑defined","Monte‑Carlo","Reverse stress‑test"])
        if mode=="Pre‑defined":
            sc=st.selectbox("Scenario",list(SCENARIOS))
            shocked=SCENARIOS[sc](latest.iloc[0])
            p=get_prob(models,pd.DataFrame([shocked]),model_sel)[0]
            st.write(f"Probability under **{sc}**: {p:.1%}")
            st.dataframe(shocked.T)
        elif mode=="Monte‑Carlo":
            n=st.number_input("Simulations",100,5000,1000)
            sims=monte_carlo(latest.iloc[0],Xtr.cov(),int(n))
            p_mc=get_prob(models,sims,model_sel)
            st.write(f"5th‑95th pct: {np.percentile(p_mc,5):.1%} – {np.percentile(p_mc,95):.1%}")
            st.area_chart(pd.DataFrame({"probability":np.sort(p_mc)}).reset_index(drop=True))
        else:
            thr=st.slider("Target probability",0.5,0.9,0.6,0.05)
            shocked,p_t=min_shock(latest.iloc[0],models[next(iter(models))],thr)
            st.write(f"Minimal L1 shock hits {p_t:.1%}")
            st.dataframe(pd.Series(shocked,index=X.columns,name="shock"))

if __name__=="__main__": main()
