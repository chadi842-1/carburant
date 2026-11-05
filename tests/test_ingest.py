from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

from carburant.config import get_paths
from carburant.geo import departement_from_cp
from carburant.ingest import (
    fetch,
    list_raw,
    parse_coordinate,
    parse_maj,
    parse_price,
    parse_xml,
    raw_path,
    stage,
)
from carburant.synthetic import generate

FIXTURE = Path(__file__).parent / "fixtures" / "official_sample.xml"


# ---- field parsers
@pytest.mark.parametrize("raw,expected", [
    ("1799", 1.799), ("1.799", 1.799), ("1,799", 1.799), ("17.99", 17.99), ("0", 0.0),
    ("", None), (None, None), ("abc", None),
])
def test_parse_price(raw, expected):
    assert parse_price(raw) == (pytest.approx(expected) if expected is not None else None)


def test_parse_coordinate_handles_both_encodings():
    assert parse_coordinate("4620114") == pytest.approx(46.20114)
    assert parse_coordinate("46.20114") == pytest.approx(46.20114)
    assert parse_coordinate("-57900") == pytest.approx(-0.579)
    assert parse_coordinate(None) is None


def test_parse_maj_formats():
    assert parse_maj("2026-06-01 07:53:00") == datetime(2026, 6, 1, 7, 53)
    assert parse_maj("2026-06-01T07:53:00") == datetime(2026, 6, 1, 7, 53)
    assert parse_maj("2026-13-45 99:00:00") is None


def test_departement_from_cp():
    assert departement_from_cp("75011") == "75"
    assert departement_from_cp("06000") == "06"
    assert departement_from_cp("20000") == "2A" and departement_from_cp("20200") == "2B"
    assert departement_from_cp("97400") == "974"
    assert departement_from_cp("") is None and departement_from_cp("7501") is None


# ---- official-format fixture
def test_parse_official_fixture():
    res = parse_xml(FIXTURE)
    assert res.n_pdv == 2 and res.warnings == ["pdv without id skipped"]
    s = res.stations.set_index("pdv_id")
    assert s.loc["1000001", "latitude"] == pytest.approx(46.20114)
    assert s.loc["1000001", "ville"] == "SAINT-DENIS-LèS-BOURG"      # ISO-8859-1 decoded
    assert s.loc["1000002", "pop"] == "A"
    p = res.prix
    assert len(p) == 4
    gazole = p[(p.pdv_id == "1000001") & (p.carburant == "Gazole")].iloc[0]
    assert gazole.prix_eur == pytest.approx(1.799) and gazole.carburant_id == 1
    bad = p[(p.pdv_id == "1000002") & (p.carburant == "Gazole")].iloc[0]
    assert pd.isna(bad.prix_eur) and pd.isna(bad.maj)                 # kept, not dropped
    assert res.ruptures.iloc[0].carburant == "SP98"


# ---- synthetic source
@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    paths = get_paths(tmp_path_factory.mktemp("synth")).ensure()
    run = generate(paths, n_stations=60, days=8, start=date(2026, 6, 1), seed=3)
    return paths, run


def test_synthetic_files_are_partitioned_and_parse(synth):
    paths, run = synth
    files = list_raw(paths)
    assert [d for d, _ in files][:2] == ["2026-06-01", "2026-06-02"] and len(files) == 8
    total = sum(parse_xml(f).n_prix for _, f in files)
    assert total == run.n_prix_elements
    assert raw_path(paths, "2026-06-03").exists()


def test_synthetic_mixes_both_price_encodings(synth):
    paths, _ = synth
    first = parse_xml(list_raw(paths)[0][1]).prix
    last = parse_xml(list_raw(paths)[-1][1]).prix
    assert first["valeur_raw"].dropna().str.contains(r"^\d{4}$").any()
    assert last["valeur_raw"].dropna().str.contains(r"^\d\.\d{3}$").any()


def test_clean_synthetic_prices_are_plausible(synth):
    paths, _ = synth
    prix = pd.concat([parse_xml(f).prix for _, f in list_raw(paths)])
    gaz = prix[(prix.carburant == "Gazole") & prix.prix_eur.between(0.3, 4)]
    assert 1.4 < gaz.prix_eur.median() < 2.1


def test_defect_manifest_matches_the_raw_files(tmp_path):
    rates = {"duplicate": 0.05, "price_typo": 0.03, "zero_price": 0.02, "missing_price": 0.02,
             "bad_timestamp": 0.02, "unknown_fuel": 0.02}
    paths = get_paths(tmp_path).ensure()
    run = generate(paths, n_stations=40, days=5, seed=1, defect_rates=rates)
    counts = run.defects["defect"].value_counts()
    assert set(counts.index) == set(rates)
    prix = pd.concat([parse_xml(f).prix for _, f in list_raw(paths)])
    assert (prix["valeur_raw"] == "").sum() == counts["missing_price"]
    assert (prix["carburant"] == "Superéthanol premium").sum() == counts["unknown_fuel"]
    assert prix["maj"].isna().sum() == counts["bad_timestamp"]
    keys = ["pdv_id", "carburant", "maj_raw", "valeur_raw"]
    assert prix.duplicated(subset=keys).sum() == counts["duplicate"]


def test_generation_is_reproducible(tmp_path):
    a = generate(get_paths(tmp_path / "a"), n_stations=20, days=3, seed=9)
    b = generate(get_paths(tmp_path / "b"), n_stations=20, days=3, seed=9)
    assert a.n_prix_elements == b.n_prix_elements
    pd.testing.assert_frame_equal(a.defects, b.defects)


# ---- staging and fetch
def test_stage_writes_parquet(synth):
    paths, run = synth
    out = stage(paths)
    assert out["files"] == 8 and out["prix_rows"] == run.n_prix_elements
    assert (paths.staging / "prix.parquet").exists()


def test_fetch_from_file_url_and_zip(tmp_path):
    import zipfile

    paths = get_paths(tmp_path).ensure()
    dest = fetch(FIXTURE.resolve().as_uri(), paths, "2026-06-01")
    assert parse_xml(dest).n_pdv == 2
    z = tmp_path / "PrixCarburants_quotidien.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.write(FIXTURE, "PrixCarburants_quotidien_20260602.xml")
    dest2 = fetch(z.resolve().as_uri(), paths, "2026-06-02")
    assert parse_xml(dest2).n_prix == 4


def test_stage_without_raw_files_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="no raw files"):
        stage(get_paths(tmp_path))
