import numpy as np
from pcscs.spectral import SpectralAnalyzer


def test_partial_spectrum_does_not_cap_component_count():
    s=np.eye(6)
    r=SpectralAnalyzer(s).static_spectral_analysis(threshold=.5, k=2)
    assert r.n_components==6
