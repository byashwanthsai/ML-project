import csv
import io
import numpy as np
from fastapi.testclient import TestClient
from app import main
from app.config import FEATURES

class FakeScaler:
    def transform(self, values): return values.to_numpy()
class FakeLR:
    def predict_proba(self, values): return np.tile([.1,.9],(len(values),1))
class FakeForest:
    def score_samples(self, values): return np.zeros(len(values))
class FakeNormalizer:
    def transform(self, values): return np.full((len(values),1),.2)

def payload(): return {name:0.0 for name in FEATURES}

def test_health_predict_validation_batch_and_missing_models(tmp_path, monkeypatch):
    monkeypatch.setattr(main,"MODEL_PATH",tmp_path/"missing.joblib")
    monkeypatch.setattr(main,"DB_PATH",tmp_path/"reviews.db")
    monkeypatch.setattr(main,"models",None)
    with TestClient(main.app) as client:
        assert client.get("/api/health").status_code==200
        assert client.post("/api/predict",json={}).status_code==422
        assert client.post("/api/predict",json=payload()).status_code==503
        main.models={"scaler":FakeScaler(),"lr":FakeLR(),"isolation":FakeForest(),"normalizer":FakeNormalizer(),"weight_lr":.7,
            "thresholds":{"lr_high":.8,"if_high":.8,"lr_mod":.5,"if_mod":.5,"risk":.75}}
        result=client.post("/api/predict",json=payload())
        assert result.status_code==200
        assert result.json()["decision"]=="Flag for Review"
        text=io.StringIO();writer=csv.DictWriter(text,fieldnames=FEATURES);writer.writeheader();writer.writerow(payload())
        batch=client.post("/api/predict/batch",files={"file":("sample.csv",text.getvalue(),"text/csv")})
        assert batch.status_code==200 and batch.json()["total"]==1
        main.models=None
        assert client.post("/api/predict",json=payload()).status_code==503
