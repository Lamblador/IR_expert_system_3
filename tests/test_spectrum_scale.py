import numpy as np

from ir_pipeline.preprocess import (
    ensure_absorbance,
    infer_spectrum_y_scale,
    transmittance_to_absorbance,
    validate_absorbance_spectrum,
)
from ir_pipeline.telegram_preprocess import convert_to_absorption_bot, jcamp_xy_to_telegram


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


def test_convert_to_absorption_bot_not_one_minus_t():
    t = np.array([0.98, 0.95, 0.5, 0.98])
    ab, meta = convert_to_absorption_bot(t, yunits="TRANSMITTANCE")
    assert meta["converted"] is True
    assert ab[2] > ab[0]


def test_jcamp_xy_to_telegram_with_absorbance_units():
    x = np.linspace(4000, 400, 100)
    t = 0.85 + 0.1 * np.sin(x / 300.0)
    tg = jcamp_xy_to_telegram(x, t, yunits="TRANSMITTANCE")
    qc = tg.scale_meta.get("absorbance_qc", {})
    assert qc.get("ok") is True
    assert float(np.nanmax(tg.absorption)) > float(np.nanmedian(tg.absorption))
