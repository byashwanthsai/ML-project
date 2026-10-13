from app.fusion import decide, risk_score

T={"lr_high":.8,"if_high":.8,"lr_mod":.5,"if_mod":.5,"risk":.75}
def test_high_lr_flags(): assert decide(.8,0,.56,T)["decision"]=="Flag for Review"
def test_high_if_flags(): assert decide(0,.8,.24,T)["decision"]=="Flag for Review"
def test_moderate_agreement_flags(): assert decide(.5,.5,.5,T)["decision"]=="Flag for Review"
def test_weighted_risk_flags(): assert decide(.7,.7,.75,T)["decision"]=="Flag for Review"
def test_low_signals_approve(): assert decide(.2,.2,.2,T)["decision"]=="Auto-Approve"
def test_weight_sum_default(): assert abs(risk_score(.5,.5,.7)-.5)<1e-9
