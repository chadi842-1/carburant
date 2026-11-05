import json

import numpy as np
import pytest

from carburant import anomaly as A
from carburant.warehouse import connect


@pytest.fixture(scope="module")
def daily(built):
    paths, _ = built
    with connect(paths, read_only=True) as con:
        return con.execute("select * from mart_prix_journalier").df()


def test_injection_rate_magnitude_and_labels(daily):
    inj = A.inject_anomalies(daily, rate=0.02, seed=3)
    n = int(inj["is_anomaly"].sum())
    assert n == round(0.02 * len(daily))
    rel = (inj.loc[inj.is_anomaly, "prix_eur"] / inj.loc[inj.is_anomaly, "prix_origine"] - 1).abs()
    assert rel.between(0.029, 0.201).all()
    assert (inj.loc[~inj.is_anomaly, "prix_eur"] == inj.loc[~inj.is_anomaly, "prix_origine"]).all()


def test_features_are_finite_and_unbiased_on_clean_data(daily):
    d = A.build_features(daily.assign(is_anomaly=False, magnitude=0.0))
    assert np.isfinite(d[A.FEATURES].to_numpy()).all()
    assert abs(d["dev_peers"].median()) < 0.01           # peers are the right reference
    assert d["has_history"].mean() > 0.5


def test_a_single_spike_is_scored_far_above_normal_rows(daily):
    d0 = daily.copy().reset_index(drop=True)
    i = d0.index[len(d0) // 2]
    d0.loc[i, "prix_eur"] *= 1.25
    d = A.build_features(d0)
    s = A.rule_score(d)
    same = ((d["pdv_id"] == d0.loc[i, "pdv_id"]) & (d["carburant"] == d0.loc[i, "carburant"])
            & (d["date_prix"] == d0.loc[i, "date_prix"]))
    spiked = d.index[same][0]
    assert s[spiked] > np.quantile(s, 0.999)


def test_evaluation_protocol_and_quality(daily):
    d = A.build_features(A.inject_anomalies(daily, rate=0.02, seed=5))
    res = A.evaluate(d, n_boot=100, seed=5)
    sp = res["split"]
    assert sp["fit_rows"] + sp["threshold_rows"] + sp["test_rows"] == len(d)
    base_rate = sp["test_anomalies"] / sp["test_rows"]
    for name, m in res["detectors"].items():
        assert m["pr_auc"] > 10 * base_rate, name              # far better than chance
        lo, hi = m["ci95"]["recall"]
        assert lo <= m["recall"] <= hi
        big = m["recall_by_magnitude"]["10-20%"]["recall"]
        assert big is None or big >= 0.8
    json.dumps(A.public(res))                                   # serialisable


def test_time_split_is_ordered(daily):
    tr, va, te = A.time_split(daily)
    assert daily.loc[tr, "date_prix"].max() < daily.loc[va, "date_prix"].min()
    assert daily.loc[va, "date_prix"].max() < daily.loc[te, "date_prix"].min()


def test_best_threshold_on_separable_scores():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    assert A.best_threshold(y, s) == pytest.approx(0.8)


def test_run_anomaly_writes_reports_and_table(built):
    paths, _ = built
    out = A.run_anomaly(paths, n_boot=50)
    assert (paths.reports / "anomaly_metrics.json").exists()
    assert (paths.reports / "anomaly_eval.png").exists()
    with connect(paths, read_only=True) as con:
        n = con.execute("select count(*) from ml.anomalies_prix").fetchone()[0]
    assert n == out["clean_data_flags"]["flagged"]
    assert out["clean_data_flags"]["false_alarm_rate"] < 0.02
