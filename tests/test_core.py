import numpy as np
import pcscs.analysis as analysis
from pcscs import PCSCS, compute_similarity_matrix


def test_similarity_matrix():
    f=np.array([[1.,0.],[0.,1.],[1.,0.]])
    s=compute_similarity_matrix(f)
    assert s.shape==(3,3)
    assert np.allclose(np.diag(s),1)
    assert np.isclose(s[0,2],1)
    assert np.isclose(s[0,1],0)


def test_progressive_components_are_nonincreasing_as_threshold_decreases():
    s=np.array([[1,.9,.2,.1],[.9,1,.3,.1],[.2,.3,1,.8],[.1,.1,.8,1.]])
    r=PCSCS().analyze_layer(s,'toy',n_steps=100)
    assert r.num_classes[0]==4
    assert r.num_classes[-1]==1
    assert np.all(np.diff(r.num_classes)<=0)


def test_sigmoid_fallback_interpolation_handles_decreasing_thresholds(monkeypatch):
    def fail(*args,**kwargs): raise RuntimeError('forced')
    monkeypatch.setattr(analysis,'curve_fit',fail)
    thresholds=np.array([1.0,.8,.6,.4])
    counts=np.array([4,3,2,1])
    x,y,params,ok=analysis.fit_sigmoid(thresholds,counts)
    assert not ok and params is None
    assert np.isclose(y[0],4)
    assert np.isclose(y[-1],1)
    assert np.all(np.diff(y)<=1e-12)
