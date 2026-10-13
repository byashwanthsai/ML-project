from contextlib import asynccontextmanager
import json, sqlite3, uuid, time
from time import perf_counter
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from .config import DATA_PATH, MODEL_PATH, METRICS_PATH, DB_PATH, FEATURES
from .data_loader import load_data
from .schemas import Transaction, Verdict
from .fusion import risk_score, decide

models=None
def init_db():
    DB_PATH.parent.mkdir(exist_ok=True)
    with sqlite3.connect(DB_PATH) as db:
        db.execute("CREATE TABLE IF NOT EXISTS scored(id TEXT PRIMARY KEY, payload TEXT, decision TEXT, risk REAL, created TEXT DEFAULT CURRENT_TIMESTAMP, verdict TEXT, note TEXT)")

@asynccontextmanager
async def lifespan(app):
    global models
    init_db()
    if MODEL_PATH.exists(): models=joblib.load(MODEL_PATH)
    yield

app=FastAPI(title="Credit Card Fraud Detection API",version="1.0.0",lifespan=lifespan)
app.add_middleware(CORSMiddleware,allow_origins=["http://localhost:5173"],allow_methods=["*"],allow_headers=["*"])
FRONTEND_DIST=Path(__file__).resolve().parents[2]/"frontend"/"dist"

def error(code,message): return {"error":{"code":code,"message":message}}
def data():
    try: return load_data()
    except FileNotFoundError as e: raise HTTPException(503,detail=error("DATASET_MISSING",str(e)))
    except ValueError as e: raise HTTPException(422,detail=error("INVALID_DATASET",str(e)))
def cached_metrics():
    return json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
def require_models():
    global models
    if models is None:
        if MODEL_PATH.exists(): models=joblib.load(MODEL_PATH)
        else: raise HTTPException(503,detail=error("MODELS_MISSING","Models not trained. Run `npm run train`."))
def transform(frame):
    require_models(); x=frame[FEATURES].copy(); x[["Time","Amount"]]=models["scaler"].transform(x[["Time","Amount"]]); return x
def score(frame):
    x=transform(frame); p=models["lr"].predict_proba(x)[:,1]
    norm=models["normalizer"]; anomaly=np.clip(norm.transform((-models["isolation"].score_samples(x)).reshape(-1,1)).ravel(),0,1)
    risk=np.array([risk_score(float(a),float(b),models["weight_lr"]) for a,b in zip(p,anomaly)])
    decisions=[decide(float(a),float(b),float(c),models["thresholds"]) for a,b,c in zip(p,anomaly,risk)]
    return x,p,anomaly,risk,decisions
def save_one(row,p,a,r,d):
    tid=str(uuid.uuid4())
    with sqlite3.connect(DB_PATH) as db: db.execute("INSERT INTO scored(id,payload,decision,risk) VALUES(?,?,?,?)",(tid,json.dumps(row),d["decision"],float(r)))
    return {"transaction_id":tid,"scores":{"logistic_regression":float(p),"isolation_forest":float(a),"hybrid":float(r)},**d,"model_version":"1.0.0"}

@app.get("/api/health")
def health():
    available=DATA_PATH.exists() or bool(cached_metrics().get("summary"))
    return {"status":"ok","dataset_available":available,"models_available":MODEL_PATH.exists(),"dataset_path":str(DATA_PATH),"models_path":str(MODEL_PATH),"message":None if available else f"Place creditcard.csv at {DATA_PATH}"}
@app.get("/api/eda/summary")
def summary():
    if not DATA_PATH.exists() and cached_metrics().get("summary"): return cached_metrics()["summary"]
    df=data(); fraud=df[df.Class==1]; q1=float(df.Amount.quantile(.25));q3=float(df.Amount.quantile(.75));iqr=q3-q1
    return {"rows":len(df),"fraud_count":int(df.Class.sum()),"fraud_rate":float(df.Class.mean()),"duplicates":int(df.duplicated().sum()),"missing_values":int(df.isna().sum().sum()),"mean_amount":float(df.Amount.mean()),"median_amount":float(df.Amount.median()),"majority_accuracy":float((df.Class==0).mean()),"class_counts":{"legitimate":int((df.Class==0).sum()),"fraud":int(df.Class.sum())},"amount_by_class":{"legitimate":{"mean":float(df[df.Class==0].Amount.mean()),"median":float(df[df.Class==0].Amount.median())},"fraud":{"mean":float(fraud.Amount.mean()),"median":float(fraud.Amount.median())}},"iqr_amount":{"q1":q1,"q3":q3,"lower_bound":q1-1.5*iqr,"upper_bound":q3+1.5*iqr,"outlier_count":int(((df.Amount<q1-1.5*iqr)|(df.Amount>q3+1.5*iqr)).sum())}}
@app.get("/api/eda/charts")
def charts(bins:int=Query(40,ge=5,le=150),feature:str=Query("V1")):
    if feature not in FEATURES: raise HTTPException(422,detail=error("INVALID_FEATURE","Choose one of the transaction features."))
    if not DATA_PATH.exists():
        snapshot=cached_metrics().get("charts")
        if not snapshot: raise HTTPException(503,detail=error("DATASET_MISSING","No EDA snapshot is available."))
        selected=str(min(80,max(10,bins)))
        return {"amount_histogram":snapshot["amount_histograms"].get(selected,snapshot["amount_histograms"]["40"]),
          "amount_box":snapshot["amount_box"],"time_histogram":snapshot["time_histogram"],"correlation":snapshot["correlation"],
          "class_correlations":snapshot["class_correlations"],"kde":[x for x in snapshot["kde"] if x["feature"]==feature],
          "scatter":snapshot["scatter"],"feature":feature}
    df=data(); cols=[*FEATURES,"Class"]
    amount=[]
    for cls,label in [(0,"legitimate"),(1,"fraud")]:
        h,e=np.histogram(df.loc[df.Class==cls,"Amount"],bins=bins); amount.extend([{"bin":float((e[i]+e[i+1])/2),"count":int(h[i]),"class":label} for i in range(len(h))])
    times,e=np.histogram(df.Time,bins=40)
    corr=df[cols].corr().round(3)
    kde=[]
    from scipy.stats import gaussian_kde
    xx=np.linspace(df[feature].quantile(.005),df[feature].quantile(.995),100)
    for cls,label in [(0,"legitimate"),(1,"fraud")]:
        vals=df.loc[df.Class==cls,feature].sample(min(12000,int((df.Class==cls).sum())),random_state=42)
        dens=gaussian_kde(vals)(xx)
        kde.extend([{"x":float(x),"density":float(y),"class":label} for x,y in zip(xx,dens)])
    amount_box=[]
    for cls,label in [(0,"legitimate"),(1,"fraud")]:
        vals=df.loc[df.Class==cls,"Amount"]; q1,q2,q3=vals.quantile([.25,.5,.75]).tolist(); spread=q3-q1
        low=float(vals[vals>=q1-1.5*spread].min()); high=float(vals[vals<=q3+1.5*spread].max())
        outs=vals[(vals<q1-1.5*spread)|(vals>q3+1.5*spread)]
        amount_box.append({"class":label,"q1":float(q1),"median":float(q2),"q3":float(q3),"lower_whisker":low,"upper_whisker":high,"outlier_count":int(len(outs)),"outliers":outs.sample(min(150,len(outs)),random_state=42).astype(float).tolist() if len(outs) else []})
    scatter=df.loc[df.Class==1,["Time","Amount"]].to_dict(orient="records")
    return {"amount_histogram":amount,"amount_box":amount_box,"time_histogram":[{"bin":float((e[i]+e[i+1])/2),"count":int(times[i])} for i in range(len(times))],"correlation":corr.to_dict(),"class_correlations":[{"feature":c,"value":float(corr.loc[c,"Class"])} for c in FEATURES],"kde":kde,"scatter":scatter,"feature":feature}
@app.get("/api/eda/features")
def features(): return FEATURES
@app.get("/api/models/metrics")
def metrics():
    if not METRICS_PATH.exists(): raise HTTPException(503,detail=error("MODELS_MISSING","Train models to compute metrics."))
    return cached_metrics()
@app.get("/api/models/threshold-preview")
def threshold_preview(model:str="Hybrid",threshold:float=Query(.5,ge=0,le=1)):
    previews=cached_metrics().get("threshold_previews",{})
    if model not in previews: raise HTTPException(422,detail=error("INVALID_MODEL","Choose Logistic Regression, Random Forest, Isolation Forest, or Hybrid."))
    key=f"{round(threshold,2):.2f}"
    if key not in previews[model]: raise HTTPException(422,detail=error("INVALID_THRESHOLD","Threshold preview supports values from 0.01 to 0.99 in 0.01 steps."))
    return {"model":model,"threshold":float(key),**previews[model][key],"split":"validation"}
@app.post("/api/predict")
def predict(tx:Transaction):
    started=perf_counter(); row=tx.model_dump(); frame=pd.DataFrame([row]); x,p,a,r,d=score(frame); result=save_one(row,p[0],a[0],r[0],d[0])
    if "random_forest" in models: result["scores"]["random_forest"]=float(models["random_forest"].predict_proba(x)[:,1][0])
    coefficients=json.loads(METRICS_PATH.read_text()).get("coefficients",[]) if METRICS_PATH.exists() else []
    contributions=sorted([{"feature":item["feature"],"value":float(item["value"]*x.iloc[0][item["feature"]])} for item in coefficients],key=lambda item:abs(item["value"]),reverse=True)[:8]
    return {**result,"latency_ms":round((perf_counter()-started)*1000,2),"top_contributions":contributions}
@app.get("/api/samples")
def samples():
    if not DATA_PATH.exists() and cached_metrics().get("reference_profiles"): return cached_metrics()["reference_profiles"]
    df=data()
    return {key:df[df.Class==cl].sample(1,random_state=42).drop(columns="Class").iloc[0].to_dict() for key,cl in [("fraud",1),("legitimate",0)]}
@app.post("/api/predict/batch")
async def batch(file:UploadFile=File(...)):
    try: df=pd.read_csv(file.file)
    except Exception: raise HTTPException(422,detail=error("INVALID_CSV","Could not read CSV upload."))
    if len(df)>50000: raise HTTPException(413,detail=error("TOO_MANY_ROWS","Upload is limited to 50,000 rows."))
    missing=[f for f in FEATURES if f not in df]
    if missing: raise HTTPException(422,detail=error("MISSING_FIELDS",f"Missing required feature columns: {', '.join(missing)}"))
    if df[FEATURES].isna().any().any() or not all(pd.api.types.is_numeric_dtype(df[f]) for f in FEATURES):
        raise HTTPException(422,detail=error("INVALID_FIELDS","Feature values must be present and numeric."))
    x,p,a,r,dec=score(df); results=[]
    rf_scores=models["random_forest"].predict_proba(x)[:,1] if "random_forest" in models else None
    insert_rows=[]
    for i,(pp,aa,rr,dd) in enumerate(zip(p,a,r,dec)):
        tid=str(uuid.uuid4()); insert_rows.append((tid,json.dumps(df[FEATURES].iloc[i].to_dict()),dd["decision"],float(rr)))
        item={"row":i+1,"transaction_id":tid,"risk":float(rr),"decision":dd["decision"],"rules":dd["rules"],"lr":float(pp),"anomaly":float(aa)}
        if rf_scores is not None: item["random_forest"]=float(rf_scores[i])
        results.append(item)
    with sqlite3.connect(DB_PATH) as db: db.executemany("INSERT INTO scored(id,payload,decision,risk) VALUES(?,?,?,?)",insert_rows)
    flagged=sum(x["decision"]=="Flag for Review" for x in results)
    out={"total":len(df),"flagged":flagged,"approved":len(df)-flagged,"flagged_rate":flagged/max(1,len(df)),"results":results}
    if "Class" in df: out["metrics"]={"precision":float(precision_score(df.Class,[x["decision"]=="Flag for Review" for x in results],zero_division=0)),"recall":float(recall_score(df.Class,[x["decision"]=="Flag for Review" for x in results],zero_division=0)),"f1":float(f1_score(df.Class,[x["decision"]=="Flag for Review" for x in results],zero_division=0))}
    return out
@app.get("/api/review/queue")
def review_queue():
    with sqlite3.connect(DB_PATH) as db:
        db.row_factory=sqlite3.Row
        rows=db.execute("SELECT * FROM scored WHERE decision='Flag for Review' ORDER BY created DESC LIMIT 100").fetchall()
    return [dict(x) for x in rows]
@app.get("/api/review/stats")
def review_stats():
    with sqlite3.connect(DB_PATH) as db:
        pending=db.execute("SELECT count(*) FROM scored WHERE decision='Flag for Review' AND verdict IS NULL").fetchone()[0]
        fraud=db.execute("SELECT count(*) FROM scored WHERE verdict='fraud'").fetchone()[0]
        legit=db.execute("SELECT count(*) FROM scored WHERE verdict='legitimate'").fetchone()[0]
    return {"pending":pending,"confirmed_fraud":fraud,"false_alarms":legit,"false_alarm_rate":legit/max(1,fraud+legit)}
@app.post("/api/review")
def review(v:Verdict):
    with sqlite3.connect(DB_PATH) as db: cur=db.execute("UPDATE scored SET verdict=?,note=? WHERE id=?",(v.verdict,v.note,v.transaction_id))
    if cur.rowcount==0: raise HTTPException(404,detail=error("NOT_FOUND","Transaction not found."))
    return {"ok":True}
@app.get("/api/review/export")
def export():
    with sqlite3.connect(DB_PATH) as db: df=pd.read_sql_query("SELECT id,payload,decision,risk,created,verdict,note FROM scored WHERE verdict IS NOT NULL",db)
    from io import StringIO
    buf=StringIO();df.to_csv(buf,index=False);buf.seek(0);return StreamingResponse(iter([buf.getvalue()]),media_type="text/csv",headers={"Content-Disposition":"attachment; filename=review_labels.csv"})
@app.get("/api/monitoring/drift")
def drift():
    if not MODEL_PATH.exists(): return {"samples":0,"features":[]}
    with sqlite3.connect(DB_PATH) as db: rows=db.execute("SELECT payload,risk FROM scored ORDER BY created DESC LIMIT 2000").fetchall()
    if not rows: return {"samples":0,"features":[]}
    ref=data() if DATA_PATH.exists() else models.get("reference_quantiles",{}); records=[json.loads(r[0]) for r in rows]; feats=[]
    for f in FEATURES:
        base=ref[f].to_numpy() if DATA_PATH.exists() else np.array(ref.get(f,[]),dtype=float); cur=np.array([x[f] for x in records]); edges=np.unique(np.quantile(base,np.linspace(0,1,11)) if DATA_PATH.exists() else base)
        if len(edges)<3: continue
        p=np.histogram(base,bins=edges)[0]/len(base);q=np.histogram(cur,bins=edges)[0]/len(cur);p=np.clip(p,1e-6,None);q=np.clip(q,1e-6,None);psi=float(np.sum((q-p)*np.log(q/p)))
        feats.append({"feature":f,"psi":psi,"status":"drift" if psi>=.25 else "warning" if psi>=.1 else "stable"})
    return {"samples":len(rows),"features":feats,"scores":[float(x[1]) for x in rows]}

@app.get("/{path:path}",include_in_schema=False)
def frontend(path:str):
    if not FRONTEND_DIST.exists(): raise HTTPException(404,detail="Frontend build is unavailable.")
    target=(FRONTEND_DIST/path).resolve()
    if target.is_relative_to(FRONTEND_DIST.resolve()) and target.is_file(): return FileResponse(target)
    return FileResponse(FRONTEND_DIST/"index.html")
