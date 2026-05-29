import numpy as np

from ir_pipeline.measurement_context import build_context_matrix, context_column_names
from ir_pipeline.resnet_input import (
    GRID_MAX_CM1,
    GRID_MIN_CM1,
    RESNET_GRID_LEN,
    RESNET_WAVENUMBERS,
    absorbance_grid_to_resnet_tensor,
)


def test_resnet_grid_matches_dataset():
    assert RESNET_GRID_LEN == 1801
    assert float(RESNET_WAVENUMBERS[0]) == GRID_MIN_CM1
    assert float(RESNET_WAVENUMBERS[-1]) == GRID_MAX_CM1


def test_absorbance_on_dataset_grid_no_extra_interp():
    wn = RESNET_WAVENUMBERS.astype(np.float64)
    ab = np.linspace(0.05, 1.0, len(wn))
    rs = absorbance_grid_to_resnet_tensor(wn, ab, peak_threshold=0.1)
    assert rs.tensor_3ch.shape == (3, RESNET_GRID_LEN)
    assert np.allclose(rs.absorption, ab.astype(np.float32), atol=1e-5)


def test_context_one_hot_sums_to_two():
    import pandas as pd

    meta = pd.DataFrame(
        [
            {"spectrum_id": "a", "measurement_mode": "atr", "sample_state": "solid"},
            {"spectrum_id": "b", "measurement_mode": "gas", "sample_state": "gas"},
        ]
    )
    X, cols = build_context_matrix(meta, ["a", "b"])
    assert len(cols) == len(context_column_names())
    assert X[0].sum() == 2.0
    assert X[1].sum() == 2.0
