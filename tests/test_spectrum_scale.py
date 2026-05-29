import numpy as np

from ir_pipeline.preprocess import (
    ensure_absorbance,
    infer_spectrum_y_scale,
    transmittance_to_absorbance,
    validate_absorbance_spectrum,
)


def test_infer_transmittance_near_one():
    t = np.linspace(0.99, 0.7, 200)
    assert infer_spectrum_y_scale(t) == "transmittance"


def test_infer_absorbance_small_positive():
    t = np.linspace(0.99, 0.7, 200)
    a = transmittance_to_absorbance(t)
    assert infer_spectrum_y_scale(a) == "absorbance"


def test_ensure_absorbance_does_not_invert_already_absorbed():
    t = np.linspace(0.99, 0.7, 200)
    a = transmittance_to_absorbance(t)
    out, meta = ensure_absorbance(a, assumed_scale="absorbance")
    assert meta["method"] == "assumed_absorbance"
    assert np.allclose(out, a)
    qc = validate_absorbance_spectrum(out)
    assert qc["ok"] is True
