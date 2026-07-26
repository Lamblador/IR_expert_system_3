from pathlib import Path

import numpy as np

from ir_pipeline.jcamp_loader import extract_jcamp_structure_ids, qc_jcamp_dict, read_jcamp_dict
from ir_pipeline.structure_resolver import mol_from_resolution, resolve_structure_for_record


def test_mislabeled_xpp_xy_pairs(tmp_path: Path):
    """SDBS_extraction пишет (X++(Y..Y)), а в теле — пары X Y."""
    p = tmp_path / "IR-NIDA-demo.jdx"
    xs = np.arange(400.0, 420.0, 2.0)
    ys = np.linspace(0.01, 0.09, xs.size)
    rows = []
    for i in range(0, xs.size, 5):
        chunk = []
        for x, y in zip(xs[i : i + 5], ys[i : i + 5]):
            chunk.append(f"{x:.2f} {y:.6f}")
        rows.append(" ".join(chunk))
    p.write_text(
        "\n".join(
            [
                "##TITLE=demo",
                "##JCAMP-DX=5.01",
                "##DATA TYPE=INFRARED SPECTRUM",
                "##XUNITS=1/CM",
                "##YUNITS=ABSORBANCE",
                "##XFACTOR=1",
                "##YFACTOR=1",
                f"##FIRSTX={xs[0]:.4f}",
                f"##LASTX={xs[-1]:.4f}",
                f"##NPOINTS={xs.size}",
                "##XYDATA=(X++(Y..Y))",
                *rows,
                "##END=",
            ]
        ),
        encoding="utf-8",
    )
    d = read_jcamp_dict(p)
    qc = qc_jcamp_dict(d)
    assert qc.ok, qc.reason
    assert len(d["x"]) == len(d["y"]) == xs.size
    np.testing.assert_allclose(d["x"], xs, atol=1e-6)
    np.testing.assert_allclose(d["y"], ys, atol=1e-6)


def test_jcamp_inchi_used_for_structure_labels(tmp_path: Path):
    """InChI/InChIKey из JCAMP попадают в dict и дают SMILES офлайн."""
    inchi = "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"
    inchikey = "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
    p = tmp_path / "ethanol.jdx"
    p.write_text(
        "\n".join(
            [
                "##TITLE=ethanol",
                "##JCAMP-DX=5.01",
                "##DATA TYPE=INFRARED SPECTRUM",
                f"##INCHI={inchi}",
                f"##INCHIKEY={inchikey}",
                "##XUNITS=1/CM",
                "##YUNITS=ABSORBANCE",
                "##XFACTOR=1",
                "##YFACTOR=1",
                "##FIRSTX=400.0000",
                "##LASTX=410.0000",
                "##NPOINTS=6",
                "##XYDATA=(X++(Y..Y))",
                "400.00 0.1 402.00 0.1 404.00 0.1 406.00 0.1 408.00 0.1 410.00 0.1",
                "##END=",
            ]
        ),
        encoding="utf-8",
    )
    d = read_jcamp_dict(p)
    j_inchi, j_ik = extract_jcamp_structure_ids(d)
    assert j_inchi == inchi
    assert j_ik == inchikey

    cache: dict = {}
    res = resolve_structure_for_record(
        None,
        "ethanol",
        cache,
        sleep_s=0.0,
        allow_network=False,
        inchi=j_inchi,
        inchikey=j_ik,
    )
    assert res.get("smiles"), res
    assert res.get("source") == "jcamp_inchi"
    assert "inchikey:" + inchikey in cache
    mol = mol_from_resolution(res)
    assert mol is not None
