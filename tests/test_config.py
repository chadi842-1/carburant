from carburant.config import FUEL_IDS, FUELS, PRICE_MAX, PRICE_MIN, get_paths


def test_fuel_catalogue_is_the_official_one():
    assert set(FUELS.values()) == {"Gazole", "SP95", "SP98", "E10", "E85", "GPLc"}
    assert all(FUEL_IDS[name] == i for i, name in FUELS.items())


def test_price_range_is_sane():
    assert 0 < PRICE_MIN < 1 < 2 < PRICE_MAX


def test_paths_layout(tmp_path, monkeypatch):
    p = get_paths(tmp_path).ensure()
    assert p.raw.is_dir() and p.staging.is_dir() and p.reports.is_dir()
    assert p.warehouse.name == "carburant.duckdb"
    monkeypatch.setenv("CARBURANT_HOME", str(tmp_path))
    assert get_paths().root == tmp_path.resolve()
