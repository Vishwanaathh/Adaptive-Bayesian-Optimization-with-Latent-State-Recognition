import numpy as np
from abo_lsr.gp import ExactRBFGaussianProcess

def test_gp_fit_predict_finite():
    x=np.linspace(0,1,12)[:,None]; y=np.sin(6*x[:,0])
    gp=ExactRBFGaussianProcess(restarts=0).fit(x,y,force=True)
    m,s=gp.predict([[.25],[.75]])
    assert np.isfinite(m).all() and np.isfinite(s).all()
    assert (s>0).all()
