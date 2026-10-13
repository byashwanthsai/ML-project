import pandas as pd
from .config import DATA_PATH, FEATURES

_cache = None

def load_data():
    global _cache
    if _cache is not None:
        return _cache
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Put creditcard.csv in backend/data/.")
    frame = pd.read_csv(DATA_PATH)
    expected = [*FEATURES, "Class"]
    if list(frame.columns) != expected:
        raise ValueError(f"Expected exactly {len(expected)} columns in the expected order; found {len(frame.columns)}.")
    _cache = frame
    return frame

