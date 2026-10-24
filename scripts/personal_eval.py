"""Personalization A, measured: does the owner's taste profile (recsys.taste) help on their own later listening?

    python -m recsys.personal plays --export <zip> --bundle bundles/<bundle> --out data/personal   # first
    python scripts/personal_eval.py --bundle bundles/<bundle> --plays data/personal --out runs/personal

The profile is built from plays before the cutoff (recsys.personal.split_times). Cases: every catalog song in
a session of the tuning weeks (then the test weeks) with a catalog song before it in the session; the input is
up to input_length catalog songs before it, as a request. Each case is answered by the serving code
(recsys.recommender, the bundle's own settings) without a profile (the base) and with the profile under each
setting of a grid (w: taste blend; alpha: artist bonus; beta: skip penalty, both in units of the base's typical
score gap between ranks 1 and 10). Per answer:
  next NDCG@10, hit@10   the session's next song
  later                  share of the 10 the session goes on to play
  artists                different artists in the 10
  new                    share of the 10 never listened to before the period
  finds per 100          new songs in the 10 that the owner then listened to (30 s+, not skipped) in the period
The setting with the best 0.7 x finds / base finds + 0.3 x NDCG / base NDCG on the tuning weeks is then
compared with the base on the test weeks, with 95% ranges from resampling whole sessions (bootstrap).
Writes only totals (no songs or artists) to <out>/eval.json.
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recsys.personal import (build_profile, day, load_plays, session_sequences,  # noqa: E402
                             split_times)
from recsys.recommender import Recommender  # noqa: E402

FINDS, NDCG = 0.7, 0.3  # the owner's weights: new songs they end up liking, and their very next song
W = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7]
ALPHA = [0.0, 0.25, 0.5, 1.0]  # x the typical score gap between ranks 1 and 10
BETA = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]  # 1.0 was the best of [0, 0.5, 1]: the edge of the first grid
METRICS = ["next_ndcg@10", "next_hit@10", "later", "artists", "new", "finds_per_100"]


def make_cases(plays, start, stop, input_length):
    """The period's cases [(session number, input song IDs, next song ID, the rest of the session)], plus the
    songs listened to before it and the songs liked (30 s+, not skipped) during it."""
    cases = []
    for s, (songs, _) in enumerate(session_sequences(plays, start, stop)):
        for i in range(1, len(songs)):
            cases.append((s, songs[max(0, i - input_length):i].tolist(), int(songs[i]), set(songs[i:].tolist())))
    ids, t = plays["song"], plays["seconds"]
    known = set(ids[plays["listened"] & (ids >= 0) & (t < start)].tolist())
    liked = set(ids[plays["listened"] & ~plays["skipped"] & (ids >= 0) & (t >= start) & (t < stop)].tolist())
    return cases, known, liked


def answer(rec, case, profile):
    out = rec.recommend([rec.uris[j] for j in case[1]], profile=profile)["recommendations"]
    return [rec.song_id[r["uri"]] for r in out], [r["score"] for r in out]


def case_metrics(rec, cases, known, liked, profile):
    """(per-case metrics (cases, METRICS), each answer's score gap between ranks 1 and 10)."""
    rows, gaps = [], []
    for case in cases:
        top, scores = answer(rec, case, profile)
        _, _, target, later = case
        new = [j not in known for j in top]
        rows.append([1 / np.log2(top.index(target) + 2) if target in top else 0.0, target in top,
                     np.mean([j in later for j in top]), len({rec.song_artist[j] for j in top}), np.mean(new),
                     100 * np.mean([n and j in liked for n, j in zip(new, top)])])
        gaps.append(scores[0] - scores[-1])
    return np.array(rows, dtype=np.float64), np.array(gaps)


def score(m, base):
    """The owner's 70/30 score: 1.0 = as good as the base."""
    return FINDS * m["finds_per_100"] / max(base["finds_per_100"], 1e-9) + NDCG * m["next_ndcg@10"] / base["next_ndcg@10"]


def means(rows):
    return dict(zip(METRICS, rows.mean(0).round(4).tolist()))


def bootstrap(rows_base, rows_best, sessions, n=1000, seed=0):
    """95% ranges of the best setting's score and metrics minus the base's, resampling whole sessions."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(sessions)
    by = {s: np.flatnonzero(sessions == s) for s in uniq}
    scores, diffs = [], []
    for _ in range(n):
        idx = np.concatenate([by[s] for s in rng.choice(uniq, len(uniq))])
        b, m = means(rows_base[idx]), means(rows_best[idx])
        scores.append(score(m, b))
        diffs.append([m[k] - b[k] for k in METRICS])
    lo, hi = np.percentile(scores, [2.5, 97.5])
    d = np.percentile(np.array(diffs), [2.5, 97.5], axis=0)
    return {"score": [round(float(lo), 3), round(float(hi), 3)],
            **{f"{k} change": [round(float(d[0, i]), 4), round(float(d[1, i]), 4)] for i, k in enumerate(METRICS)}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--plays", default="data/personal")
    ap.add_argument("--out", default="runs/personal")
    args = ap.parse_args()
    began = time.perf_counter()
    plays = load_plays(args.plays, args.bundle)
    cutoff, test_start, end = split_times(plays["seconds"])
    profile, info = build_profile(plays, args.bundle, until=cutoff, key="eval")
    rec = Recommender(args.bundle, "cpu")
    L = rec.settings["input_length"]
    parts = {name: make_cases(plays, a, b, L) for name, (a, b) in
             {"tune": (cutoff, test_start), "test": (test_start, end)}.items()}
    result = {"bundle": rec.version, "profile_until": day(cutoff), "tune": [day(cutoff), day(test_start)],
              "test": [day(test_start), day(end - 1)], "profile": info,
              "cases": {k: len(v[0]) for k, v in parts.items()},
              "sessions": {k: len({c[0] for c in v[0]}) for k, v in parts.items()},
              "serving": {"max_per_artist": rec.max_per_artist, "mix": rec.mix},
              "weights": {"finds": FINDS, "next_ndcg": NDCG}}
    print(json.dumps(result), flush=True)
    # tuning weeks: the base, then the grid
    cases, known, liked = parts["tune"]
    base_rows, gaps = case_metrics(rec, cases, known, liked, None)
    gap = float(np.median(gaps))
    base = means(base_rows)
    grid = []
    for w, a, b in itertools.product(W, ALPHA, BETA):
        rec.profiles = {"eval": profile.with_settings(w, a * gap, b * gap)}
        m = means(case_metrics(rec, cases, known, liked, "eval")[0])
        grid.append({"w": w, "alpha": a, "beta": b, **m, "score": round(score(m, base), 4)})
        print(json.dumps(grid[-1]), flush=True)
    best = max(grid, key=lambda g: g["score"])
    result["tune_results"] = {"score_gap": round(gap, 4), "base": base, "best": best,
                              "grid": sorted(grid, key=lambda g: -g["score"])}
    # test weeks: the base and the chosen setting, with bootstrap ranges
    cases, known, liked = parts["test"]
    sessions = np.array([c[0] for c in cases])
    rec.profiles = {"eval": profile.with_settings(best["w"], best["alpha"] * gap, best["beta"] * gap)}
    rows_base, _ = case_metrics(rec, cases, known, liked, None)
    rows_best, _ = case_metrics(rec, cases, known, liked, "eval")
    b, m = means(rows_base), means(rows_best)
    result["test_results"] = {"base": b, "chosen": {**m, "score": round(score(m, b), 4)},
                              "ranges_95": bootstrap(rows_base, rows_best, sessions)}
    rec.close()
    result["seconds"] = round(time.perf_counter() - began)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("test_results", "seconds")}, indent=2))


if __name__ == "__main__":
    main()
