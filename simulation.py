"""B.A.S.I.C. final model - the 10,000 model stress test from the video.

Runs on the saved 2025-26 stats in data/, so with the default seed you get
the exact same rankings as the video every time.
"""
import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
DEFAULT_SEED = 20260914
RUN_COUNT = 10_000
THRESHOLDS = [250, 500, 750, 1000]
CENTERS = ["mean", "median", "trimmed"]
SCALES = ["std", "mad"]
DISTANCES = ["euclidean", "manhattan"]

STYLE = [
    "shot_0_3_freq", "shot_3_10_freq", "shot_10_16_freq", "shot_16_3p_freq",
    "shot_3p_freq", "ast_2p_rate", "ast_3p_rate", "dunk_fga_rate",
    "corner3_share", "adv_ftr",
]
CATEGORIES = {
    "scoring_volume": ["p100_pts"],
    "efficiency": ["adv_tspct"],
    "playmaking_security": ["adv_astpct", "adv_tovpct"],
    "rebounding": ["adv_orbpct", "adv_drbpct"],
    "defense_activity": ["adv_stlpct", "adv_blkpct"],
    "role": ["adv_usgpct", "minutes_per_game", "pos_index"],
    "style": STYLE,
    "physical": ["height_in", "weight_lb", "age"],
    "impact": ["adv_bpm"],
}


def fix_encoding(name):
    # some names with accents got mangled when scraping, this undoes that
    text = str(name)
    try:
        if any(char in text for char in ("Ã", "Ä", "Å", "â")):
            return text.encode("latin-1").decode("utf-8")
    except UnicodeError:
        pass
    return text


def normalized_name(name):
    # "Luka Dončić" -> "lukadoncic" so names match between files
    text = unicodedata.normalize("NFKD", fix_encoding(name))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", "", text.casefold())


def prepare_data(master_path, bio_path):
    players = pd.read_csv(master_path)
    bios = pd.read_csv(bio_path)

    # fill in height/weight for the players NBA.com bios missed

    players["_name_key"] = players["Player"].map(normalized_name)
    bios["_name_key"] = bios["PLAYER_NAME"].map(normalized_name)
    bio_lookup = bios.drop_duplicates("_name_key").set_index("_name_key")
    for stat, bio_col in (("height_in", "PLAYER_HEIGHT_INCHES"), ("weight_lb", "PLAYER_WEIGHT")):
        missing = players[stat].isna()
        players.loc[missing, stat] = players.loc[missing, "_name_key"].map(bio_lookup[bio_col])

    # turn the 5 position % columns into one number, 1 = PG ... 5 = C
    # (my first version had these as 5 separate stats, which basically counted position 5 times)
    positional = ["pos_pgpct", "pos_sgpct", "pos_sfpct", "pos_pfpct", "pos_cpct"]
    denominator = players[positional].sum(axis=1).replace(0, np.nan)
    players["pos_index"] = sum(
        (i+1) * players[col] for i, col in enumerate(positional)
    ) / denominator
    return players


def center_and_scale(matrix, center, scaling):
    if center == "mean":
        reference = np.nanmean(matrix, axis=0)
    elif center == "median":
        reference = np.nanmedian(matrix, axis=0)
    else:
        reference = []
        for j in range(matrix.shape[1]):
            values = np.sort(matrix[~np.isnan(matrix[:, j]), j])
            lo = int(len(values) * .1)
            hi = max(int(len(values) * .9), lo + 1)
            reference.append(values[lo:hi].mean())
        reference = np.array(reference)
    if scaling == "std":
        divisor = np.nanstd(matrix, axis=0)
    else:
        median = np.nanmedian(matrix, axis=0)
        divisor = np.array([
            np.nanmedian(np.abs(matrix[:, j]-median[j])) * 1.4826
            for j in range(matrix.shape[1])
        ])
    divisor[(divisor == 0) | np.isnan(divisor)] = np.nan
    return reference, divisor


def precompute(players):
    # there are only 4 x 3 x 2 = 24 cutoff/center/scaling combos, so scale
    # everything once up front instead of 10,000 times.
    # to qualify you need at least 80% of the stats in every category
    # (my first version let players with a missing category score 0 there, which helped them)
    cache = {}
    for threshold in THRESHOLDS:
        pool = players[players.minutes >= threshold].copy().reset_index(drop=True)
        valid = np.ones(len(pool), dtype=bool)
        available = {}
        for category, features in CATEGORIES.items():
            present = [f for f in features if f in pool and pool[f].notna().mean() >= .5]
            if not present:
                raise ValueError(f"Missing entire category {category} at {threshold} minutes")
            available[category] = present
            valid &= (pool[present].notna().mean(axis=1).values >= .8)
        pool = pool.loc[valid].reset_index(drop=True)
        features = sorted(set(f for group in available.values() for f in group))
        prepared = pool[features].astype(float).copy()
        prepared = prepared.fillna(prepared.median())
        indices = {f: i for i, f in enumerate(features)}
        matrix = prepared.to_numpy()
        for center in CENTERS:
            for scaling in SCALES:
                reference, divisor = center_and_scale(matrix, center, scaling)
                normalized = (matrix-reference) / divisor
                cache[(threshold, center, scaling)] = {
                    "pool": pool, "normalized": normalized,
                    "feature_index": indices, "available": available,
                }
    return cache


def simulate(players, runs=RUN_COUNT, seed=DEFAULT_SEED, include_run_ranks=False):
    rng = np.random.default_rng(seed)
    cache = precompute(players)
    records = []
    for run in range(runs):
        threshold = int(rng.choice(THRESHOLDS))
        center = str(rng.choice(CENTERS, p=[.40, .40, .20]))
        scale = str(rng.choice(SCALES, p=[.55, .45]))
        distance = str(rng.choice(DISTANCES))
        entry = cache[(threshold, center, scale)]
        pool = entry["pool"]
        normalized = entry["normalized"]
        feature_index = entry["feature_index"]
        distances = []
        for category, features in entry["available"].items():
            retained = rng.random(len(features)) < .85
            if not retained.any():
                retained[rng.integers(0, len(features))] = True
            indices = [feature_index[f] for f, keep in zip(features, retained) if keep]
            values = normalized[:, indices]
            if distance == "euclidean":
                per_player = np.sqrt(np.mean(values*values, axis=1))
            else:
                per_player = np.mean(np.abs(values), axis=1)
            distances.append(per_player)
        category_scores = np.column_stack(distances)
        category_weights = rng.dirichlet(np.repeat(8.0, category_scores.shape[1]))
        overall_scores = category_scores @ category_weights
        sorted_idx = np.argsort(overall_scores)
        n = len(sorted_idx)
        ranks = np.empty(n, dtype=int)
        ranks[sorted_idx] = np.arange(1, n+1)
        rank_pct = (ranks-1)/max(n-1,1)
        record = pd.DataFrame({
            "run": run,
            "Player": pool.Player.values,
            "rank": ranks,
            "rank_pct": rank_pct,
        })
        if include_run_ranks:
            record["score"] = overall_scores
            record["threshold"] = threshold
        records.append(record)
    run_records = pd.concat(records, ignore_index=True)
    completed = run_records.run.nunique()
    summary = run_records.groupby("Player").agg(
        runs_eligible=("run", "nunique"),
        average_rank=("rank", "mean"),
        median_rank=("rank", "median"),
        average_rank_pct=("rank_pct", "mean"),
        median_rank_pct=("rank_pct", "median"),
        rank_sd=("rank", "std"),
        best_rank=("rank", "min"),
        worst_rank=("rank", "max"),
        wins=("rank", lambda values: int((values == 1).sum())),
        top5=("rank", lambda values: int((values <= 5).sum())),
        top10=("rank", lambda values: int((values <= 10).sum())),
        top10pct=("rank_pct", lambda values: int((values <= .10).sum())),
    ).reset_index()
    summary["eligibility_rate"] = summary.runs_eligible / completed
    for field in ("wins", "top5", "top10", "top10pct"):
        summary[f"{field}_rate_when_eligible"] = summary[field] / summary.runs_eligible
    summary = summary.merge(players[["Player", "team", "pos", "minutes"]], on="Player", how="left")
    finalists = summary[summary.eligibility_rate >= .999].copy()
    finalists = finalists.sort_values(["median_rank_pct", "average_rank_pct", "rank_sd"])
    finalists.insert(0, "FinalistRank", np.arange(1, len(finalists)+1))
    return finalists, run_records


def main():
    parser = argparse.ArgumentParser(description="Run the B.A.S.I.C. 10,000-model simulation.")
    parser.add_argument("--master", type=Path, default=DATA / "model_input.csv",
                        help="Player stats (default: data/model_input.csv)")
    parser.add_argument("--bios", type=Path, default=DATA / "bios.csv",
                        help="Height/weight fill-ins (default: data/bios.csv)")
    parser.add_argument("--out", type=Path, default=Path("results"),
                        help="Output folder (default: results/)")
    parser.add_argument("--runs", type=int, default=RUN_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--save-run-ranks", action="store_true",
                        help="Save each run's player ranks to a compressed CSV")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    players = prepare_data(args.master, args.bios)
    finalists, run_records = simulate(players, args.runs, args.seed, args.save_run_ranks)
    finalists.to_csv(args.out / "finalists.csv", index=False)
    if args.save_run_ranks:
        run_records.to_csv(args.out / "run_ranks.csv.gz", index=False, compression="gzip")
    print(finalists.head(10)[["FinalistRank", "Player", "average_rank", "median_rank", "wins"]].to_string(index=False))
    print(f"Saved {len(finalists)} finalists to {args.out/'finalists.csv'}")


if __name__ == "__main__":
    main()
