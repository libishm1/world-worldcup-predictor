#!/usr/bin/env python3
"""Improved Spain vs Argentina predictor using leakage-safe rolling features,
score-based Poisson models, and a calibrated outcome ensemble.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict, deque
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import poisson
from sklearn.metrics import accuracy_score, log_loss


RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"


def ensure_results_file(path: Path) -> Path:
    """Download the public international-results dataset when it is not cached."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Historical results not found; downloading to {path} ...")
    try:
        urlretrieve(RESULTS_URL, path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not download historical results from {RESULTS_URL}. "
            f"Download results.csv manually and pass it with --data. Details: {exc}"
        ) from exc
    return path

NAME_MAP = {
    "USA": "United States", "Korea Republic": "South Korea",
    "Republic of Ireland": "Ireland", "Türkiye": "Turkey",
    "Cape Verde": "Cabo Verde", "Côte d'Ivoire": "Ivory Coast",
    "Czechia": "Czech Republic", "Curaçao": "Curacao",
    "Congo DR": "DR Congo", "Congo": "Republic of the Congo",
}

FEATURES = [
    "neutral", "importance", "home_elo", "away_elo", "elo_diff",
    "home_form5", "away_form5", "home_gd5", "away_gd5",
    "home_gf5", "away_gf5", "home_ga5", "away_ga5",
    "home_form10", "away_form10", "home_gd10", "away_gd10",
    "home_gf10", "away_gf10", "home_ga10", "away_ga10",
    "home_rest", "away_rest", "h2h_n", "h2h_home_form", "h2h_home_gd",
    "is_world_cup",
]


def normalize_team(name: str) -> str:
    return NAME_MAP.get(name, name)


def tournament_importance(name: str) -> float:
    t = str(name).lower()
    if "fifa world cup" in t and "qualif" not in t:
        return 4.0
    if any(s in t for s in ["uefa euro", "copa am", "africa cup", "asian cup", "gold cup"]):
        return 3.0
    if "qualif" in t:
        return 2.5
    if "nations league" in t:
        return 2.0
    if "friendly" in t:
        return 1.0
    return 1.5


def recent_stats(history: deque, n: int) -> tuple[float, float, float, float]:
    if not history:
        return 0.5, 0.0, 1.2, 1.2
    vals = list(history)[-n:]
    result = np.array([v[0] for v in vals], dtype=float)
    gf = np.array([v[1] for v in vals], dtype=float)
    ga = np.array([v[2] for v in vals], dtype=float)
    return float(result.mean()), float((gf-ga).mean()), float(gf.mean()), float(ga.mean())


def feature_row(home: str, away: str, date: pd.Timestamp, neutral: int, tournament: str,
                elo, history, last_date, h2h) -> dict:
    home = normalize_team(home); away = normalize_team(away)
    h5 = recent_stats(history[home], 5); a5 = recent_stats(history[away], 5)
    h10 = recent_stats(history[home], 10); a10 = recent_stats(history[away], 10)
    rh = (date-last_date[home]).days if home in last_date else 30
    ra = (date-last_date[away]).days if away in last_date else 30
    pair = list(h2h[(home, away)])
    return {
        "date": date, "home_team": home, "away_team": away,
        "neutral": int(neutral), "importance": tournament_importance(tournament),
        "home_elo": float(elo[home]), "away_elo": float(elo[away]),
        "elo_diff": float(elo[home]-elo[away]),
        "home_form5": h5[0], "away_form5": a5[0],
        "home_gd5": h5[1], "away_gd5": a5[1],
        "home_gf5": h5[2], "away_gf5": a5[2],
        "home_ga5": h5[3], "away_ga5": a5[3],
        "home_form10": h10[0], "away_form10": a10[0],
        "home_gd10": h10[1], "away_gd10": a10[1],
        "home_gf10": h10[2], "away_gf10": a10[2],
        "home_ga10": h10[3], "away_ga10": a10[3],
        "home_rest": float(min(max(rh, 0), 60)), "away_rest": float(min(max(ra, 0), 60)),
        "h2h_n": float(len(pair)),
        "h2h_home_form": float(np.mean([x[0] for x in pair])) if pair else 0.5,
        "h2h_home_gd": float(np.mean([x[1] for x in pair])) if pair else 0.0,
        "is_world_cup": int("fifa world cup" in tournament.lower() and "qualif" not in tournament.lower()),
    }


def build_dataset(raw: pd.DataFrame):
    raw = raw.copy()
    raw["home_team"] = raw["home_team"].map(normalize_team)
    raw["away_team"] = raw["away_team"].map(normalize_team)
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.sort_values("date").reset_index(drop=True)

    elo = defaultdict(lambda: 1500.0)
    history = defaultdict(lambda: deque(maxlen=20))
    last_date = {}
    h2h = defaultdict(lambda: deque(maxlen=10))
    rows = []

    for r in raw.itertuples(index=False):
        if pd.isna(r.home_score) or pd.isna(r.away_score):
            continue
        home, away = normalize_team(r.home_team), normalize_team(r.away_team)
        date = pd.Timestamp(r.date)
        neutral = int(str(r.neutral).upper() == "TRUE")
        f = feature_row(home, away, date, neutral, str(r.tournament), elo, history, last_date, h2h)
        hs, ass = int(r.home_score), int(r.away_score)
        f.update({
            "home_score": hs, "away_score": ass,
            "label": 0 if hs > ass else (1 if hs == ass else 2),
        })
        rows.append(f)

        he, ae = elo[home], elo[away]
        bonus = 0.0 if neutral else 55.0
        expected = 1.0 / (1.0 + 10 ** (-((he + bonus) - ae) / 400.0))
        actual = 1.0 if hs > ass else (0.5 if hs == ass else 0.0)
        margin = abs(hs-ass)
        mov = math.log(margin+1.0) * (2.2/(0.001*abs(he-ae)+2.2)) if margin else 1.0
        k = 18.0 + 5.0*tournament_importance(str(r.tournament))
        delta = k*mov*(actual-expected)
        elo[home] += delta; elo[away] -= delta

        hr = 1.0 if hs > ass else (0.5 if hs == ass else 0.0)
        ar = 1.0-hr
        history[home].append((hr, hs, ass)); history[away].append((ar, ass, hs))
        h2h[(home, away)].append((hr, hs-ass)); h2h[(away, home)].append((ar, ass-hs))
        last_date[home] = date; last_date[away] = date

    return pd.DataFrame(rows), elo, history, last_date, h2h


def outcome_probs(lam_home: float, lam_away: float, rho: float = -0.08, max_goals: int = 10):
    goals = np.arange(max_goals+1)
    matrix = np.outer(poisson.pmf(goals, lam_home), poisson.pmf(goals, lam_away))
    matrix[0, 0] *= 1-lam_home*lam_away*rho
    matrix[0, 1] *= 1+lam_home*rho
    matrix[1, 0] *= 1+lam_away*rho
    matrix[1, 1] *= 1-rho
    matrix /= matrix.sum()
    return np.array([np.tril(matrix, -1).sum(), np.trace(matrix), np.triu(matrix, 1).sum()]), matrix


def fit_models(train: pd.DataFrame):
    X = train[FEATURES].astype(float)
    age_years = (train["date"].max()-train["date"]).dt.days/365.25
    weights = np.exp(-np.log(2)*age_years/5.0) * (0.85+0.15*train["importance"])

    common = dict(
        n_estimators=420, learning_rate=0.035, max_depth=3,
        min_child_weight=8, subsample=0.85, colsample_bytree=0.85,
        reg_lambda=5.0, reg_alpha=0.15, tree_method="hist", n_jobs=4,
    )
    classifier = xgb.XGBClassifier(
        objective="multi:softprob", num_class=3, eval_metric="mlogloss",
        random_state=42, **common,
    )
    classifier.fit(X, train["label"].astype(int), sample_weight=weights, verbose=False)

    home_goals = xgb.XGBRegressor(
        objective="count:poisson", eval_metric="poisson-nloglik", random_state=43, **common,
    )
    away_goals = xgb.XGBRegressor(
        objective="count:poisson", eval_metric="poisson-nloglik", random_state=44, **common,
    )
    home_goals.fit(X, train["home_score"].astype(float), sample_weight=weights, verbose=False)
    away_goals.fit(X, train["away_score"].astype(float), sample_weight=weights, verbose=False)
    return classifier, home_goals, away_goals


def predict_components(models, frame: pd.DataFrame):
    clf, hg, ag = models
    X = frame[FEATURES].astype(float)
    cls = clf.predict_proba(X)
    lam_h = np.clip(hg.predict(X), 0.10, 5.0)
    lam_a = np.clip(ag.predict(X), 0.10, 5.0)
    pois = np.vstack([outcome_probs(h, a)[0] for h, a in zip(lam_h, lam_a)])
    return cls, pois, lam_h, lam_a


def score_metrics(y, probs):
    onehot = np.eye(3)[np.asarray(y, dtype=int)]
    return {
        "log_loss": float(log_loss(y, probs, labels=[0,1,2])),
        "accuracy": float(accuracy_score(y, np.argmax(probs, axis=1))),
        "brier": float(np.mean(np.sum((probs-onehot)**2, axis=1))),
    }


def temperature_scale(probs, temperature):
    probs = np.clip(np.asarray(probs, dtype=float), 1e-9, 1.0)
    scaled = probs ** (1.0/temperature)
    return scaled / scaled.sum(axis=1, keepdims=True)

def tune_blend(y, cls, pois):
    best = None
    for alpha in np.linspace(0.0, 1.0, 51):
        raw = alpha*cls+(1-alpha)*pois
        raw = raw/raw.sum(axis=1, keepdims=True)
        for temperature in np.linspace(0.80, 1.35, 56):
            p = temperature_scale(raw, temperature)
            ll = log_loss(y, p, labels=[0,1,2])
            if best is None or ll < best[0]:
                best = (ll, float(alpha), float(temperature))
    return best[1], best[2]


def symmetric_final_prediction(models, row_ab: pd.DataFrame, row_ba: pd.DataFrame, alpha: float, temperature: float):
    c1, p1, lh1, la1 = predict_components(models, row_ab)
    c2, p2, lh2, la2 = predict_components(models, row_ba)
    c = np.array([(c1[0,0]+c2[0,2])/2, (c1[0,1]+c2[0,1])/2, (c1[0,2]+c2[0,0])/2])
    p = np.array([(p1[0,0]+p2[0,2])/2, (p1[0,1]+p2[0,1])/2, (p1[0,2]+p2[0,0])/2])
    c /= c.sum(); p /= p.sum()
    final = alpha*c+(1-alpha)*p
    final = temperature_scale(final.reshape(1,-1), temperature)[0]
    lam_home = float((lh1[0]+la2[0])/2)
    lam_away = float((la1[0]+lh2[0])/2)
    _, matrix = outcome_probs(lam_home, lam_away)
    return final, c, p, lam_home, lam_away, matrix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("home", nargs="?", default="Spain")
    ap.add_argument("away", nargs="?", default="Argentina")
    ap.add_argument("--data", default="data_cache/results.csv")
    ap.add_argument("--date", default="2026-07-19")
    ap.add_argument("--output", default=None, help="JSON output path (default: predictions/<date>/<teams>.json)")
    args = ap.parse_args()

    data_path = ensure_results_file(Path(args.data))
    if args.output is None:
        slug = f"{args.home}_vs_{args.away}".lower().replace(" ", "_")
        output_path = Path("predictions") / args.date / f"{slug}.json"
    else:
        output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(data_path)
    dataset, elo, history, last_date, h2h = build_dataset(raw)
    dataset = dataset[dataset["date"] >= pd.Timestamp("2006-01-01")].reset_index(drop=True)

    train = dataset[dataset["date"] < pd.Timestamp("2022-01-01")]
    validation = dataset[(dataset["date"] >= pd.Timestamp("2022-01-01")) & (dataset["date"] < pd.Timestamp("2024-01-01"))]
    test = dataset[(dataset["date"] >= pd.Timestamp("2024-01-01")) & (dataset["date"] < pd.Timestamp(args.date))]

    models = fit_models(train)
    vc, vp, _, _ = predict_components(models, validation)
    alpha, temperature = tune_blend(validation["label"], vc, vp)
    tc, tp, _, _ = predict_components(models, test)
    te = temperature_scale(alpha*tc+(1-alpha)*tp, temperature)

    validation_metrics = {
        "classifier": score_metrics(validation["label"], vc),
        "poisson": score_metrics(validation["label"], vp),
        "ensemble": score_metrics(validation["label"], temperature_scale(alpha*vc+(1-alpha)*vp, temperature)),
    }
    test_metrics = {
        "classifier": score_metrics(test["label"], tc),
        "poisson": score_metrics(test["label"], tp),
        "ensemble": score_metrics(test["label"], te),
    }

    # Refit on all completed matches before the final.
    final_train = dataset[dataset["date"] < pd.Timestamp(args.date)]
    final_models = fit_models(final_train)
    date = pd.Timestamp(args.date)
    ab = pd.DataFrame([feature_row(args.home, args.away, date, 1, "FIFA World Cup", elo, history, last_date, h2h)])
    ba = pd.DataFrame([feature_row(args.away, args.home, date, 1, "FIFA World Cup", elo, history, last_date, h2h)])
    probs, cls_probs, pois_probs, lam_h, lam_a, matrix = symmetric_final_prediction(final_models, ab, ba, alpha, temperature)

    scorelines = []
    for h in range(matrix.shape[0]):
        for a in range(matrix.shape[1]):
            scorelines.append((float(matrix[h,a]), h, a))
    scorelines.sort(reverse=True)

    result = {
        "match": f"{args.home} vs {args.away}",
        "date": args.date,
        "training_matches": int(len(final_train)),
        "blend_weight_classifier": alpha,
        "calibration_temperature": temperature,
        "validation": validation_metrics,
        "test_2024_to_2026_07_18": test_metrics,
        "elo": {args.home: float(elo[normalize_team(args.home)]), args.away: float(elo[normalize_team(args.away)])},
        "expected_goals": {args.home: lam_h, args.away: lam_a},
        "probabilities_90_minutes": {args.home: float(probs[0]), "Draw": float(probs[1]), args.away: float(probs[2])},
        "components": {
            "enhanced_xgboost": {args.home: float(cls_probs[0]), "Draw": float(cls_probs[1]), args.away: float(cls_probs[2])},
            "dixon_coles_poisson": {args.home: float(pois_probs[0]), "Draw": float(pois_probs[1]), args.away: float(pois_probs[2])},
        },
        "top_scorelines": [
            {"score": f"{args.home} {h}-{a} {args.away}", "probability": p}
            for p,h,a in scorelines[:8]
        ],
        "pick_90_minutes": [args.home, "Draw", args.away][int(np.argmax(probs))],
    }
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"\nSaved prediction to: {output_path}")

if __name__ == "__main__":
    main()
