from ir_pipeline.dataset_split import DEFAULT_FRACTIONS, fractions_from_train_frac


def test_fractions_from_train_frac_matches_default_at_70():
    f = fractions_from_train_frac(0.70)
    assert f == DEFAULT_FRACTIONS


def test_fractions_from_train_frac_honors_train_frac():
    f = fractions_from_train_frac(0.85)
    assert abs(f["train"] - 0.85) < 1e-9
    assert abs(sum(f.values()) - 1.0) < 1e-9
    assert abs(f["val"] / f["test"] - DEFAULT_FRACTIONS["val"] / DEFAULT_FRACTIONS["test"]) < 1e-9
