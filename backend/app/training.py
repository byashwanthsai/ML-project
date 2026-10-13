import json
from datetime import datetime, timezone
import hashlib
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (average_precision_score, precision_recall_curve, roc_auc_score,
    precision_score, recall_score, f1_score, confusion_matrix, roc_curve)
from .config import DATA_PATH, MODEL_PATH, METRICS_PATH, FEATURES, SEED
from .data_loader import load_data
from .fusion import risk_score, decide

def choose_threshold(y, score):
    p, r, t = precision_recall_curve(y, score)
    f = 2*p[:-1]*r[:-1]/np.maximum(p[:-1]+r[:-1], 1e-12)
    return float(t[int(np.argmax(f))])

def evaluate(y, score, threshold):
    pred = score >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"precision": float(precision_score(y, pred, zero_division=0)), "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)), "roc_auc": float(roc_auc_score(y, score)),
            "pr_auc": float(average_precision_score(y, score)), "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}

def curve_points(y, score):
    precision, recall, _ = precision_recall_curve(y, score)
    fpr, tpr, _ = roc_curve(y, score)
    def sample(x, z):
        ids=np.linspace(0,len(x)-1,min(200,len(x))).astype(int)
        return [{"x":float(x[i]),"y":float(z[i])} for i in ids]
    return {"pr":sample(recall,precision),"roc":sample(fpr,tpr)}

def eda_snapshot(df):
    fraud=df[df.Class==1]; q1=float(df.Amount.quantile(.25)); q3=float(df.Amount.quantile(.75)); iqr=q3-q1
    summary={"rows":len(df),"fraud_count":int(df.Class.sum()),"fraud_rate":float(df.Class.mean()),
      "duplicates":int(df.duplicated().sum()),"missing_values":int(df.isna().sum().sum()),
      "mean_amount":float(df.Amount.mean()),"median_amount":float(df.Amount.median()),
      "majority_accuracy":float((df.Class==0).mean()),
      "class_counts":{"legitimate":int((df.Class==0).sum()),"fraud":int(df.Class.sum())},
      "amount_by_class":{"legitimate":{"mean":float(df[df.Class==0].Amount.mean()),"median":float(df[df.Class==0].Amount.median())},
        "fraud":{"mean":float(fraud.Amount.mean()),"median":float(fraud.Amount.median())}},
      "iqr_amount":{"q1":q1,"q3":q3,"lower_bound":q1-1.5*iqr,"upper_bound":q3+1.5*iqr,
        "outlier_count":int(((df.Amount<q1-1.5*iqr)|(df.Amount>q3+1.5*iqr)).sum())}}
    amount_histograms={}
    for bins in range(10,81):
        amount=[]
        for cls,label in [(0,"legitimate"),(1,"fraud")]:
            h,e=np.histogram(df.loc[df.Class==cls,"Amount"],bins=bins)
            amount.extend([{"bin":float((e[i]+e[i+1])/2),"count":int(h[i]),"class":label} for i in range(len(h))])
        amount_histograms[str(bins)]=amount
    times,edges=np.histogram(df.Time,bins=40)
    corr=df[[*FEATURES,"Class"]].corr().round(3)
    kde=[]
    from scipy.stats import gaussian_kde
    for feature in FEATURES:
        xx=np.linspace(df[feature].quantile(.005),df[feature].quantile(.995),100)
        for cls,label in [(0,"legitimate"),(1,"fraud")]:
            vals=df.loc[df.Class==cls,feature].sample(min(12000,int((df.Class==cls).sum())),random_state=42)
            dens=gaussian_kde(vals)(xx)
            kde.extend([{"feature":feature,"x":float(x),"density":float(y),"class":label} for x,y in zip(xx,dens)])
    boxes=[]
    for cls,label in [(0,"legitimate"),(1,"fraud")]:
        vals=df.loc[df.Class==cls,"Amount"]; lo,med,hi=vals.quantile([.25,.5,.75]).tolist(); spread=hi-lo
        lower=float(vals[vals>=lo-1.5*spread].min()); upper=float(vals[vals<=hi+1.5*spread].max())
        outs=vals[(vals<lo-1.5*spread)|(vals>hi+1.5*spread)]
        boxes.append({"class":label,"q1":float(lo),"median":float(med),"q3":float(hi),"lower_whisker":lower,
          "upper_whisker":upper,"outlier_count":int(len(outs)),"outliers":[]})
    charts={"amount_histograms":amount_histograms,"amount_box":boxes,
      "time_histogram":[{"bin":float((edges[i]+edges[i+1])/2),"count":int(times[i])} for i in range(len(times))],
      "correlation":corr.to_dict(),"class_correlations":[{"feature":f,"value":float(corr.loc[f,"Class"])} for f in FEATURES],
      "kde":kde,"scatter":fraud.groupby(pd.cut(fraud.Time,bins=30),observed=True).agg(Time=("Time","mean"),Amount=("Amount","median")).dropna().to_dict(orient="records")}
    reference_quantiles={f:[float(x) for x in df[f].quantile(np.linspace(0,1,11)).tolist()] for f in FEATURES}
    reference_profiles={name:df[df.Class==cl][FEATURES].median().to_dict() for name,cl in [("fraud",1),("legitimate",0)]}
    return summary,charts,reference_quantiles,reference_profiles

def main():
    df = load_data(); X = df[FEATURES]; y = df.Class
    eda_summary,eda_charts,reference_quantiles,reference_profiles=eda_snapshot(df)
    X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=.30, random_state=SEED, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=.50, random_state=SEED, stratify=y_rest)
    scaler = StandardScaler().fit(X_train[["Time", "Amount"]])
    def prep(x):
        z=x.copy(); z[["Time", "Amount"]]=scaler.transform(x[["Time", "Amount"]]); return z
    a,b,c=prep(X_train),prep(X_val),prep(X_test)
    lrpipe=Pipeline([("lr",LogisticRegression(class_weight="balanced",solver="liblinear",max_iter=2000,random_state=SEED))])
    grid=GridSearchCV(lrpipe,{"lr__C":np.logspace(-3,2,6)},scoring="average_precision",cv=StratifiedKFold(5,shuffle=True,random_state=SEED),n_jobs=2)
    grid.fit(a,y_train); base=grid.best_estimator_
    try:
        from sklearn.frozen import FrozenEstimator
        calibrated=CalibratedClassifierCV(FrozenEstimator(base),method="sigmoid")
    except ImportError:
        try: calibrated=CalibratedClassifierCV(estimator=base,method="sigmoid",cv="prefit")
        except TypeError: calibrated=CalibratedClassifierCV(base_estimator=base,method="sigmoid",cv="prefit")
    calibrated.fit(b,y_val)
    lrval=calibrated.predict_proba(b)[:,1]; tlr=choose_threshold(y_val,lrval)
    # A class-balanced bagged tree baseline complements the linear classifier on tabular data.
    forest=RandomForestClassifier(n_estimators=300,max_features="sqrt",min_samples_leaf=2,
        class_weight="balanced_subsample",n_jobs=2,random_state=SEED)
    forest.fit(a,y_train)
    rfval=forest.predict_proba(b)[:,1]; trf=choose_threshold(y_val,rfval)
    best_if=None; best_ap=-1
    for n in (100,200):
        for cont in (.001,.0017,.005,.01):
            model=IsolationForest(n_estimators=n,contamination=cont,random_state=SEED,n_jobs=2).fit(a)
            s=-model.score_samples(b); ap=average_precision_score(y_val,s)
            if ap>best_ap: best_if=model; best_ap=ap; best_n=n; best_cont=cont
    raw_train=-best_if.score_samples(a); norm=MinMaxScaler().fit(raw_train.reshape(-1,1))
    def anomaly(x): return np.clip(norm.transform((-best_if.score_samples(x)).reshape(-1,1)).ravel(),0,1)
    ifval=anomaly(b); tif=choose_threshold(y_val,ifval)
    bestw=.7; bestf=-1
    for w in np.linspace(.1,.9,17):
        rv=w*lrval+(1-w)*ifval; f=f1_score(y_val,rv>=choose_threshold(y_val,rv),zero_division=0)
        if f>bestf: bestf=f;bestw=float(w)
    riskval=bestw*lrval+(1-bestw)*ifval; trisk=choose_threshold(y_val,riskval)
    # Use independently tuned validation thresholds and a conservative consensus rule.
    thresholds={"lr_high":tlr,"if_high":tif,"lr_mod":float(np.quantile(lrval,.99)),"if_mod":float(np.quantile(ifval,.99)),"risk":trisk}
    lrtest=calibrated.predict_proba(c)[:,1]; rftest=forest.predict_proba(c)[:,1]; iftest=anomaly(c); hytest=bestw*lrtest+(1-bestw)*iftest
    hy_pred=np.array([decide(float(p),float(q),float(r),thresholds)["decision"]=="Flag for Review" for p,q,r in zip(lrtest,iftest,hytest)])
    hy_metrics=evaluate(y_test,hytest,0.5) if False else None
    tn,fp,fn,tp=confusion_matrix(y_test,hy_pred,labels=[False,True]).ravel()
    hy_metrics={"precision":float(precision_score(y_test,hy_pred,zero_division=0)),"recall":float(recall_score(y_test,hy_pred,zero_division=0)),"f1":float(f1_score(y_test,hy_pred,zero_division=0)),"roc_auc":float(roc_auc_score(y_test,hytest)),"pr_auc":float(average_precision_score(y_test,hytest)),"confusion_matrix":{"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)}}
    metrics={"trained_at":datetime.now(timezone.utc).isoformat(),"model_version":"1.0.0","dataset_sha256":hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
      "summary":eda_summary,"charts":eda_charts,"reference_profiles":reference_profiles,
      "rows":{"total":len(df),"train":len(a),"validation":len(b),"test":len(c)},"fraud_count":int(y.sum()),"fraud_rate":float(y.mean()),"duplicates":int(df.duplicated().sum()),"missing_values":int(df.isna().sum().sum()),
      "thresholds":thresholds,"weight_lr":bestw,"hyperparameters":{"lr_C":float(grid.best_params_["lr__C"]),"rf_n_estimators":300,"rf_max_features":"sqrt","rf_min_samples_leaf":2,"if_n_estimators":best_n,"if_contamination":best_cont},
      "cv_results":[{"C":float(params["lr__C"]),"mean_pr_auc":float(score),"std_pr_auc":float(std)} for params,score,std in zip(grid.cv_results_["params"],grid.cv_results_["mean_test_score"],grid.cv_results_["std_test_score"])],
      "curves":{"Logistic Regression":curve_points(y_test,lrtest),"Random Forest":curve_points(y_test,rftest),"Isolation Forest":curve_points(y_test,iftest),"Hybrid":curve_points(y_test,hytest)},
      "models":{"Logistic Regression":evaluate(y_test,lrtest,tlr),"Random Forest":evaluate(y_test,rftest,trf),"Isolation Forest":evaluate(y_test,iftest,tif),"Hybrid":hy_metrics},
      "threshold_previews":{},
      "coefficients":sorted([{"feature":f,"value":float(v)} for f,v in zip(FEATURES,base.named_steps["lr"].coef_[0])],key=lambda x:abs(x["value"]),reverse=True)[:20],
      "feature_importance":sorted([{"feature":f,"importance":float(v)} for f,v in zip(FEATURES,forest.feature_importances_)],key=lambda x:x["importance"],reverse=True)[:15]}
    val_scores={"Logistic Regression":lrval,"Random Forest":rfval,"Isolation Forest":ifval,"Hybrid":riskval}
    for name,values in val_scores.items():
        metrics["threshold_previews"][name]={}
        for cutoff in np.linspace(.01,.99,99):
            pred=values>=cutoff; tn,fp,fn,tp=confusion_matrix(y_val,pred,labels=[False,True]).ravel(); key=f"{cutoff:.2f}"
            metrics["threshold_previews"][name][key]={"precision":float(precision_score(y_val,pred,zero_division=0)),
              "recall":float(recall_score(y_val,pred,zero_division=0)),"f1":float(f1_score(y_val,pred,zero_division=0)),
              "confusion_matrix":{"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)}}
    MODEL_PATH.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump({"scaler":scaler,"lr":calibrated,"random_forest":forest,"isolation":best_if,"normalizer":norm,"weight_lr":bestw,"thresholds":thresholds,"features":FEATURES,"reference_quantiles":reference_quantiles},MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics,indent=2)); print(json.dumps(metrics["models"],indent=2))

if __name__ == "__main__": main()
