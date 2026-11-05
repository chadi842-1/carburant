"""Price-anomaly detection, evaluated on anomalies injected on purpose.

Setting: a "suspect price" is a station's daily price that departs from both its own recent
history and its local peers (same département, fuel and road type). Legitimate market moves shift
every station together, so they move neither signal much.

Evaluation protocol (labels exist only because anomalies are injected):
- a share of daily prices is multiplied by (1 +/- m), m ~ U(3%, 20%), recorded as labels;
- days are split in time: the first 50% fit the detector, the next 25% choose its threshold
  (F1-optimal), the last 25% are scored once;
- precision / recall / F1 with 95% bootstrap intervals, PR-AUC, and recall by magnitude band.

Two detectors, same features:
- robust rule: score = min(|z_history|, |z_peers|) with MAD-scaled z-scores (no training);
- Isolation Forest on [dev_history, dev_peers, jump, z_history, z_peers].
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_curve

from .config import SEED

FEATURES = ["dev_history", "dev_peers", "jump", "z_history", "z_peers"]
MAG_BANDS = [(0.03, 0.05), (0.05, 0.10), (0.10, 0.20)]
HISTORY_WINDOW = 7


# ------------------------------------------------------------------------- injection
def inject_anomalies(
    df: pd.DataFrame, rate: float = 0.01, mag_range: tuple[float, float] = (0.03, 0.20),
    seed: int = SEED,
) -> pd.DataFrame:
    """Copy of `df` with `is_anomaly`, `magnitude` and a perturbed `prix_eur`."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    n = len(out)
    idx = rng.choice(n, size=max(1, int(round(rate * n))), replace=False)
    mag = rng.uniform(*mag_range, size=len(idx))
    sign = rng.choice([-1.0, 1.0], size=len(idx))
    out["is_anomaly"] = False
    out["magnitude"] = 0.0
    out["prix_origine"] = out["prix_eur"]
    pos = out.columns.get_loc
    out.iloc[idx, pos("prix_eur")] = (out["prix_eur"].to_numpy()[idx] * (1 + sign * mag)).round(3)
    out.iloc[idx, pos("is_anomaly")] = True
    out.iloc[idx, pos("magnitude")] = mag
    return out


# -------------------------------------------------------------------------- features
def _mad_z(x: pd.Series) -> pd.Series:
    med = x.median()
    mad = (x - med).abs().median()
    scale = 1.4826 * mad if mad > 0 else (x.std() or 1.0)
    return (x - med) / scale


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add relative deviations from the station's history and from local peers on the day."""
    d = df.sort_values(["pdv_id", "carburant", "date_prix"]).reset_index(drop=True)
    g = d.groupby(["pdv_id", "carburant"])["prix_eur"]
    hist = g.transform(lambda s: s.shift(1).rolling(HISTORY_WINDOW, min_periods=2).median())
    prev = g.shift(1)
    peer_keys = ["date_prix", "dep", "carburant", "autoroute"]
    peers = d.groupby(peer_keys)["prix_eur"].transform("median")
    n_peers = d.groupby(peer_keys)["prix_eur"].transform("size")
    nat = d.groupby(["date_prix", "carburant", "autoroute"])["prix_eur"].transform("median")
    peers = peers.where(n_peers >= 3, nat)                  # thin groups fall back to national
    d["dev_history"] = (d["prix_eur"] / hist - 1).fillna(0.0)
    d["dev_peers"] = d["prix_eur"] / peers - 1
    d["jump"] = (d["prix_eur"] / prev - 1).fillna(0.0)
    # z-scores per fuel so that cheap fuels (E85, GPLc) are on the same footing
    d["z_history"] = d.groupby("carburant")["dev_history"].transform(_mad_z)
    d["z_peers"] = d.groupby("carburant")["dev_peers"].transform(_mad_z)
    d["has_history"] = hist.notna()
    return d


def rule_score(d: pd.DataFrame) -> np.ndarray:
    zh, zp = d["z_history"].abs().to_numpy(), d["z_peers"].abs().to_numpy()
    return np.where(d["has_history"].to_numpy(), np.minimum(zh, zp), zp)


# ------------------------------------------------------------------------- evaluation
def time_split(d: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    days = np.sort(d["date_prix"].unique())
    a, b = days[int(len(days) * 0.50)], days[int(len(days) * 0.75)]
    return d["date_prix"] < a, (d["date_prix"] >= a) & (d["date_prix"] < b), d["date_prix"] >= b


def best_threshold(y: np.ndarray, s: np.ndarray) -> float:
    p, r, t = precision_recall_curve(y, s)
    f1 = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0)[:-1]
    return float(t[int(np.argmax(f1))])


def _prf(y: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    tp = float((y & pred).sum())
    prec = tp / pred.sum() if pred.sum() else 0.0
    rec = tp / y.sum() if y.sum() else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def _bootstrap(y, pred, n_boot: int, seed: int) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    stats = np.array([_prf(y[i], pred[i])
                      for i in rng.integers(0, len(y), size=(n_boot, len(y)))])
    q = np.quantile(stats, [0.025, 0.975], axis=0)
    return {k: [round(float(q[0, j]), 4), round(float(q[1, j]), 4)]
            for j, k in enumerate(["precision", "recall", "f1"])}


@dataclass
class DetectorResult:
    name: str
    threshold: float
    metrics: dict
    test_scores: np.ndarray


def evaluate(d: pd.DataFrame, n_boot: int = 1000, seed: int = SEED) -> dict:
    """Fit/threshold/test both detectors on an injected, feature-enriched frame."""
    tr, va, te = time_split(d)
    y = d["is_anomaly"].to_numpy()
    X = d[FEATURES].to_numpy()

    iforest = IsolationForest(n_estimators=300, contamination="auto", random_state=seed)
    iforest.fit(X[tr.to_numpy()])
    scores = {"robust_rule": rule_score(d), "isolation_forest": -iforest.score_samples(X)}

    out = {"split": {"fit_rows": int(tr.sum()), "threshold_rows": int(va.sum()),
                     "test_rows": int(te.sum()), "test_anomalies": int(y[te.to_numpy()].sum())},
           "detectors": {}}
    te_np, va_np = te.to_numpy(), va.to_numpy()
    for name, s in scores.items():
        t = best_threshold(y[va_np], s[va_np])
        yt, st = y[te_np], s[te_np]
        pred = st >= t
        prec, rec, f1 = _prf(yt, pred)
        mags = d.loc[te, "magnitude"].to_numpy()
        by_band = {}
        for lo, hi in MAG_BANDS:
            m = yt & (mags >= lo) & (mags < hi)
            by_band[f"{int(lo * 100)}-{int(hi * 100)}%"] = {
                "n": int(m.sum()), "recall": round(float(pred[m].mean()), 4) if m.sum() else None}
        out["detectors"][name] = {
            "threshold": round(float(t), 5),
            "precision": round(float(prec), 4), "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            "pr_auc": round(float(average_precision_score(yt, st)), 4),
            "flagged": int(pred.sum()),
            "ci95": _bootstrap(yt, pred, n_boot, seed),
            "recall_by_magnitude": by_band,
        }
        out.setdefault("_scores", {})[name] = (yt, st, t)
    return out


def false_alarms_on_clean(d_clean: pd.DataFrame, rule_threshold: float) -> dict:
    """Rule on data with NO injected anomaly: every flag is a false alarm by construction."""
    s = rule_score(d_clean)
    flagged = int((s >= rule_threshold).sum())
    return {"rows": int(len(d_clean)), "flagged": flagged,
            "false_alarm_rate": round(flagged / len(d_clean), 5)}


def plot(result: dict, out_path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .viz import COLORS, INK, INK_2, apply_style, title

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    for k, (name, (yt, st, _)) in enumerate(result["_scores"].items()):
        p, r, _ = precision_recall_curve(yt, st)
        m = result["detectors"][name]
        ax.plot(r, p, color=COLORS[name], lw=2)
        ax.plot(m["recall"], m["precision"], "o", color=COLORS[name], ms=8, mec="white", mew=1.5)
        y0 = 0.30 - 0.09 * k                      # key in the empty lower-left corner
        ax.plot([0.05, 0.11], [y0, y0], color=COLORS[name], lw=2)
        ax.text(0.13, y0, f"{LABELS[name]}  PR-AUC {m['pr_auc']:.2f}, "
                f"P {m['precision']:.2f} / R {m['recall']:.2f}",
                va="center", fontsize=9, color=INK)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    title(ax, "Precision-recall on the test period", "dot = threshold chosen on validation")

    ax = axes[1]
    bands = list(next(iter(result["detectors"].values()))["recall_by_magnitude"])
    x = np.arange(len(bands))
    w = 0.36
    for k, name in enumerate(result["detectors"]):
        vals = [result["detectors"][name]["recall_by_magnitude"][b]["recall"] or 0 for b in bands]
        bars = ax.bar(x + (k - 0.5) * w, vals, w - 0.04, color=COLORS[name], label=LABELS[name])
        for b, v in zip(bars, vals, strict=True):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center",
                    fontsize=8, color=INK_2)
    ax.set_xticks(x)
    ax.set_xticklabels(bands)
    ax.set_xlabel("Size of the injected price change")
    ax.set_ylim(0, 1.12)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    title(ax, "Recall by anomaly size", "small anomalies are the hard ones")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


LABELS = {"robust_rule": "Robust rule", "isolation_forest": "Isolation Forest"}


def public(result: dict) -> dict:
    """Result without the raw score arrays (JSON-serialisable)."""
    return {k: v for k, v in result.items() if not k.startswith("_")}


def run_anomaly(paths, rate: float = 0.01, n_boot: int = 1000, seed: int = SEED) -> dict:
    """Evaluate on injected anomalies, then flag suspect prices in the real (un-injected) data.

    Writes reports/anomaly_metrics.json, reports/anomaly_eval.png and the table ml.anomalies_prix
    (rule flags with their scores), which the API and the agent read.
    """
    import json

    from .warehouse import connect

    with connect(paths, read_only=True) as con:
        df = con.execute("select * from mart_prix_journalier").df()
    res = evaluate(build_features(inject_anomalies(df, rate=rate, seed=seed)), n_boot, seed)
    t_rule = res["detectors"]["robust_rule"]["threshold"]
    clean = build_features(df.assign(is_anomaly=False, magnitude=0.0))
    clean["score"] = rule_score(clean)
    res["clean_data_flags"] = false_alarms_on_clean(clean, t_rule)
    res["settings"] = {"injection_rate": rate, "magnitude_range": [0.03, 0.20],
                       "history_window": HISTORY_WINDOW, "n_boot": n_boot, "seed": seed}
    flags = clean.loc[clean["score"] >= t_rule,
                      ["date_prix", "pdv_id", "ville", "dep", "carburant", "prix_eur",
                       "dev_history", "dev_peers", "score"]]
    with connect(paths) as con:
        con.execute("create schema if not exists ml")
        con.register("flags_df", flags)
        con.execute("create or replace table ml.anomalies_prix as select * from flags_df")
    paths.reports.mkdir(parents=True, exist_ok=True)
    plot(res, paths.reports / "anomaly_eval.png")
    out = public(res)
    (paths.reports / "anomaly_metrics.json").write_text(json.dumps(out, indent=2, default=str))
    return out
