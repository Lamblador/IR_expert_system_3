import numpy as np

from ir_pipeline.preprocess import preprocess_to_grid, transmittance_to_absorbance, wavenumbers_from_jcamp


def test_transmittance_to_absorbance_percent():
    y = np.array([100.0, 50.0, 10.0])
    a = transmittance_to_absorbance(y)
    assert np.all(np.isfinite(a))
    assert a[0] < a[2]


def test_micrometers_to_wavenumber():
    lam = np.array([4.0, 10.0])  # µm
    nu = wavenumbers_from_jcamp(lam, "MICROMETERS")
    assert np.isclose(nu[0], 2500.0)
    assert np.isclose(nu[1], 1000.0)


def test_preprocess_to_grid_basic():
    x = np.linspace(400, 4000, 80)
    y = np.cos(x / 600.0) + 1.1
    gs = preprocess_to_grid(x, y, grid_step=50.0)
    assert gs.wavenumbers[0] == 400.0
    assert gs.wavenumbers[-1] <= 4000.0
    assert len(gs.wavenumbers) == len(gs.absorbance_normalized)
