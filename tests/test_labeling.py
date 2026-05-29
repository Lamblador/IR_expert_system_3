import numpy as np

from ir_pipeline.labeling import REGION_PEAK_MIN_HEIGHT, find_dominant_peak_in_region


def test_peak_rejected_when_region_max_below_threshold():
    wn = np.linspace(2200, 1800, 10)
    y = np.full(10, 0.05)
    y[5] = 0.08
    mask = np.ones(10, dtype=bool)
    pk, conf = find_dominant_peak_in_region(
        wn, y, mask, 1850.0, 2150.0, min_peak_height=REGION_PEAK_MIN_HEIGHT
    )
    assert pk is None
    assert conf == 0.0


def test_peak_accepted_when_region_max_at_threshold():
    wn = np.linspace(2200, 1800, 10)
    y = np.full(10, 0.05)
    y[5] = REGION_PEAK_MIN_HEIGHT
    mask = np.ones(10, dtype=bool)
    pk, conf = find_dominant_peak_in_region(
        wn, y, mask, 1850.0, 2150.0, min_peak_height=REGION_PEAK_MIN_HEIGHT
    )
    assert pk is not None
    assert conf > 0.0
