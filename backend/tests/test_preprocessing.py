import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def test_scaler_fit_training_only():
    train=pd.DataFrame({"Time":[0.,10.],"Amount":[5.,15.]}); val=pd.DataFrame({"Time":[1000.],"Amount":[900.]})
    scaler=StandardScaler().fit(train); assert np.allclose(scaler.mean_,[5,10]); assert not np.allclose(scaler.mean_,val.mean())

