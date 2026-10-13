from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "creditcard.csv"
MODEL_PATH = ROOT / "models" / "artifacts.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
DB_PATH = ROOT / "models" / "reviews.db"
FEATURES = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
SEED = 42

