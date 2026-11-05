"""Command line: `carburant <step>` or `carburant run` for the whole pipeline.

    carburant generate [--stations N --days D]    synthetic files into the raw zone
    carburant ingest --url URL --date YYYY-MM-DD  one real file into the raw zone
    carburant stage | load | dbt | anomalies | eval | report
    carburant run [--stations N --days D]         everything, then reports/results.md
    carburant ask "question"                      one question to the agent
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import date

from .config import SEED, Paths, get_paths


def step_generate(paths: Paths, stations: int, days: int, seed: int) -> dict:
    from .synthetic import generate

    run = generate(paths, n_stations=stations, days=days, seed=seed)
    return {"stations": run.n_stations, "days": run.days, "prix_elements": run.n_prix_elements,
            "injected_defects": int(len(run.defects))}


def step_stage(paths: Paths) -> dict:
    from .ingest import stage

    return stage(paths)


def step_load(paths: Paths) -> dict:
    from .warehouse import load_raw

    return load_raw(paths)


def step_dbt(paths: Paths) -> dict:
    from .warehouse import quality_report, run_dbt

    res = run_dbt(paths)
    if not res["success"]:
        raise SystemExit(f"dbt build failed: {res['failures']}")
    out = {"dbt": res, "quality": quality_report(paths)}
    (paths.reports / "quality.json").write_text(json.dumps(out, indent=2, default=str))
    return out


def step_anomalies(paths: Paths, seed: int) -> dict:
    from .anomaly import run_anomaly

    return run_anomaly(paths, seed=seed)


def step_eval(paths: Paths) -> dict:
    from .evaluate import compare_routers

    return compare_routers(paths)


def step_report(paths: Paths) -> str:
    from .dashboard import plot_dashboard
    from .report import render_results

    plot_dashboard(paths)
    md = render_results(paths)
    (paths.reports / "results.md").write_text(md, encoding="utf-8")
    return str(paths.reports / "results.md")


def run_all(paths: Paths, stations: int, days: int, seed: int) -> None:
    paths.ensure()
    steps = [
        ("generate", lambda: step_generate(paths, stations, days, seed)),
        ("stage", lambda: step_stage(paths)),
        ("load", lambda: step_load(paths)),
        ("dbt build + quality", lambda: step_dbt(paths)),
        ("anomalies", lambda: step_anomalies(paths, seed)),
        ("agent eval", lambda: step_eval(paths)),
        ("report", lambda: step_report(paths)),
    ]
    meta = {"stations": stations, "days": days, "seed": seed, "started": str(date.today())}
    (paths.reports / "run_config.json").write_text(json.dumps(meta, indent=2))
    for i, (name, fn) in enumerate(steps, 1):
        print(f"[{i}/{len(steps)}] {name}", flush=True)
        fn()
    print(f"done: {paths.reports / 'results.md'}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="carburant", description=__doc__.splitlines()[0])
    ap.add_argument("--root", help="working directory (default $CARBURANT_HOME or cwd)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("generate", "run"):
        p = sub.add_parser(name)
        p.add_argument("--stations", type=int, default=1000)
        p.add_argument("--days", type=int, default=120)
        p.add_argument("--seed", type=int, default=SEED)
    p = sub.add_parser("ingest")
    p.add_argument("--url", required=True)
    p.add_argument("--date", required=True)
    for name in ("stage", "load", "dbt", "anomalies", "eval", "report"):
        sub.add_parser(name)
    p = sub.add_parser("ask")
    p.add_argument("question")
    a = ap.parse_args(argv)
    paths = get_paths(a.root).ensure()

    if a.cmd == "run":
        run_all(paths, a.stations, a.days, a.seed)
        return
    if a.cmd == "generate":
        out = step_generate(paths, a.stations, a.days, a.seed)
    elif a.cmd == "ingest":
        from .ingest import fetch

        out = {"written": str(fetch(a.url, paths, a.date))}
    elif a.cmd == "stage":
        out = step_stage(paths)
    elif a.cmd == "load":
        out = step_load(paths)
    elif a.cmd == "dbt":
        out = step_dbt(paths)["quality"]
    elif a.cmd == "anomalies":
        out = {k: v for k, v in step_anomalies(paths, SEED).items() if k != "detectors"}
    elif a.cmd == "eval":
        out = {k: {m: v[m] for m in ("accuracy", "accuracy_by_kind", "latency_ms")}
               for k, v in step_eval(paths).items()}
    elif a.cmd == "report":
        out = {"written": step_report(paths)}
    else:
        from .agent import Agent

        ans = Agent(paths).ask(a.question)
        out = {"answer": ans.answer, "sources": [asdict_safe(s) for s in ans.sources]}
    print(json.dumps(out, indent=2, ensure_ascii=False, default=str))


def asdict_safe(obj):
    return obj.model_dump() if hasattr(obj, "model_dump") else asdict(obj)


if __name__ == "__main__":
    main()
