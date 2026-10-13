def risk_score(p_lr: float, anomaly: float, weight_lr: float = 0.7) -> float:
    return weight_lr * p_lr + (1 - weight_lr) * anomaly

def decide(p_lr, anomaly, risk, thresholds):
    rules = []
    if p_lr >= thresholds["lr_high"]: rules.append("Strong supervised signal")
    if anomaly >= thresholds["if_high"]: rules.append("Strong anomaly signal")
    if p_lr >= thresholds["lr_mod"] and anomaly >= thresholds["if_mod"]: rules.append("Both models agree at moderate confidence")
    if risk >= thresholds["risk"]: rules.append("Hybrid risk threshold exceeded")
    return {"decision": "Flag for Review" if rules else "Auto-Approve", "rules": rules}

