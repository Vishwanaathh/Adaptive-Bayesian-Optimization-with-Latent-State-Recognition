import numpy as np
from abo_lsr.hmm import GaussianHMM

def test_filter_normalizes():
    h=GaussianHMM()
    r=h.filter(1.2)
    assert np.isclose(r.belief.sum(),1)
    assert np.all(r.belief>=0)

def test_em_keeps_transition_rows_stochastic():
    rng=np.random.default_rng(4)
    x=np.r_[rng.normal(1.5,.5,30),rng.normal(0,.5,30),rng.normal(-1,.3,30)]
    h=GaussianHMM()
    assert h.fit_em(x,n_iter=8)
    assert np.allclose(h.A.sum(axis=1),1)
    assert np.all(h.vars>=h.min_variance)

def test_em_preserves_variance_ordered_state_labels():
    rng = np.random.default_rng(12)
    observations = np.r_[rng.normal(0, 1.5, 40), rng.normal(0, .7, 40), rng.normal(0, .25, 40)]
    hmm = GaussianHMM()
    hmm.fit_em(observations, n_iter=10)
    assert hmm.vars[0] >= hmm.vars[1] >= hmm.vars[2]
