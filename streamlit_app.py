# composite_risk_app.py – v4.2 (Unified UI)
"""Dashboard now shows **four tabs**:
1. Probability history  2. Feature snapshot  3. Importance  4. Scenario Lab
So the original diagnostics stay intact while scenario testing lives in its own
 tab, not hidden above the fold.
"""
from __future__ import annotations
import os, numpy as np, pandas as pd, joblib, streamlit as st
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
except: LGBMClassifier=None
try: from catboost import CatBoostClassifier
except: CatBoostClassifier=None

CACHE_DIR=Path(".cache");CACHE_DIR.mkdir(exist_ok=True)
FRED_API_KEY=os.getenv("FRED_API_KEY"); START_DATE="1960-01-01"; DEFAULT_H=12
FRED_SERIES={"DGS10":"10‑Year Treasury Constant Maturity Rate","TB3MS":"3‑Month Treasury Bill Secondary Market Rate","BAA":"Moody's Seasoned Baa Corporate Bond Yield","AAA":"Moody's Seasoned Aaa Corporate Bond Yield","UNRATE":"Unemployment Rate","VIXCLS":"CBOE VIX Close","USRECD":"NBER Recession Indicator","SP500":"S&P 500 Index (Daily Close)","USSLIND":"Philly Fed Leading Index","NFCI":"Chicago Fed NFCI","ANFCI":"Chicago Fed Adjusted NFCI"}

st.set_page_config(page_title="Composite Risk Gauge",layout="wide")

def load_fred():
    fred=Fred(api_key=FRED_API_KEY)
    df=pd.DataFrame({n:fred.get_series(s,START_DATE) for s,n in FRED_SERIES.items()});df.index=pd.to_datetime(df.index)
    return df.resample("M").last().ffill()

def feats(raw,h):
    f=pd.DataFrame(index=raw.index)
    f["term_spread"]=raw["10‑Year Treasury Constant Maturity Rate"]-raw["3‑Month Treasury Bill Secondary Market Rate"]
    f["credit_spread"]=raw["Moody's Seasoned Baa Corporate Bond Yield"]-raw["Moody's Seasoned Aaa Corporate Bond Yield"]
    f["vix"]=raw["CBOE VIX Close"]
    f["sahm_rule"]=raw["Unemployment Rate"]-raw["Unemployment Rate"].rolling(12,1).min()
    f["lei_6m_pct"]=raw["Philly Fed Leading Index"].pct_change(6)
    f["nfci"]=raw["Chicago Fed NFCI"]
    for c in ["term_spread","credit_spread","vix","sahm_rule","lei_6m_pct","nfci"]:
        f[f"{c}_chg6"]=f[c]-f[c].shift(6)
    z=(f-f.expanding(60).mean())/f.expanding(60).std(ddof=0)
    f=pd.concat([f,z.add_prefix("z_")],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
    fut=raw["S&P 500 Index (Daily Close)"].resample("M").last().pct_change(h)
    rec=raw["NBER Recession Indicator"].shift(-h).rolling(h).max()
    f["target"]=((rec>0)|(fut<=-0.20)).astype(int)
    return f

def build_models(X,y,names):
    m={}
    if "Logit" in names:m["Logit"]=Pipeline([("std",StandardScaler()),("clf",LogisticRegression(max_iter=600,solver="liblinear",penalty="l1"))]).fit(X,y)
    if "Random Forest" in names:m["Random Forest"]=RandomForestClassifier(400,max_depth=6,random_state=42).fit(X,y)
    if "Gradient Boosting" in names:m["Gradient Boosting"]=GradientBoostingClassifier(400,0.05,3,random_state=42).fit(X,y)
    if "XGBoost" in names:m["XGBoost"]=XGBClassifier(n_estimators=400,learning_rate=0.03,max_depth=3,subsample=0.8,colsample_bytree=0.8,eval_metric="logloss",random_state=42).fit(X,y)
    if LGBMClassifier and "LightGBM" in names:m["LightGBM"]=LGBMClassifier(n_estimators=400,learning_rate=0.03,subsample=0.8,colsample_bytree=0.8).fit(X,y)
    if CatBoostClassifier and "CatBoost" in names:m["CatBoost"]=CatBoostClassifier(iterations=400,learning_rate=0.03,depth=4,loss_function="Logloss",verbose=False).fit(X,y)
    return m

def prob(m,X,sel):
    return np.mean([md.predict_proba(X)[:,1] for md in m.values()],axis=0) if sel=="Ensemble" else m[sel].predict_proba(X)[:,1]

# scenario builders
SCEN={"Dot‑com":lambda x: x.assign(term_spread=x.term_spread+1,credit_spread=x.credit_spread+1.5,vix=x.vix+10,nfci=x.nfci+0.5),"Covid‑flash":lambda x: x.assign(term_spread=x.term_spread+0.8,credit_spread=x.credit_spread+3,vix=x.vix+40,nfci=x.nfci+1.2)}

def main():
    raw=load_fred(); h=st.sidebar.slider("Horizon",6,18,DEFAULT_H,3); data=feats(raw,h)
    X,y=data.drop(columns=["target"]),data["target"]; split=data.index[-1]-relativedelta(years=5); Xtr,ytr=X[X.index<split],y[y.index<split]
    opts=["Logit","Random Forest","Gradient Boosting","XGBoost"]+(["LightGBM"] if LGBMClassifier else [])+(["CatBoost"] if CatBoostClassifier else [])+ ["Ensemble"]
    sel=st.sidebar.selectbox("Model",opts,index=len(opts)-1)
    models=build_models(Xtr,ytr,opts[:-1] if sel=="Ensemble" else [sel])
    latest=X.iloc[[-1]]; p_now=prob(models,latest,sel)[0]
    st.title("🛡️ Composite Risk Gauge v4.2")
    st.metric("Current prob",f"{p_now:.1%}")

    # Tabs
    tab_hist,tab_feat,tab_imp,tab_scen=st.tabs(["Probability history","Feature snapshot","Importance","Scenario Lab"])
    with tab_hist:
        st.line_chart(pd.Series(prob(models,X,sel),index=data.index,name="prob"))
    with tab_feat:
        st.dataframe(latest.T)
    with tab_imp:
        if sel!="Ensemble":
            m=models[sel]
            if hasattr(m,"feature_importances_"):
                st.bar_chart(pd.Series(m.feature_importances_,index=X.columns))
            elif isinstance(m,Pipeline):
                st.bar_chart(pd.Series(np.abs(m.named_steps["clf"].coef_[0]),index=X.columns))
        else: st.write("Select single model for importance")
    with tab_scen:
        st.subheader("Play a shock scenario")
        sc=st.selectbox("Scenario",list(SCEN)+["None"])
        if sc!="None":
            shocked=SCEN[sc](latest.copy().iloc[0]); p=prob(models,pd.DataFrame([shocked]),sel)[0]
            st.write(f"Probability → {p:.1%}")
            st.dataframe(shocked.T)

if __name__=="__main__": main()
