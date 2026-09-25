"""Phases 1-5 of B.A.S.I.C. (the build-up before the final model).

Pulls 2025-26 stats from Basketball-Reference and player bios from NBA.com,
then runs each phase and saves the leaderboards to phase_results/.

Heads up: this scrapes live, so if the sites change or block you it'll break,
and the numbers can drift a bit from data/phases/ as stats get updated.
Downloads get cached in phase_results/raw/ so you only hit the sites once.
"""
import re
import time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

SEASON = "2025-26"
YEAR = 2026
OUT = Path("phase_results")
RAW = OUT / "raw"

BREF = "https://www.basketball-reference.com/leagues/NBA_{}_{}.html"
PAGES = {
    "per_game": "per_game",
    "per100": "per_poss",
    "advanced": "advanced",
    "shooting": "shooting",
    "pbp": "play-by-play",
}
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/140 Safari/537.36"}


# ---------- getting the data ----------

def clean_name(name):
    return re.sub(r"\*+$", "", str(name).strip()).replace("\xa0", " ")


def flatten(df):
    # bref tables have 2-row headers, e.g. ("% of FGA by Distance", "0-3")
    if isinstance(df.columns, pd.MultiIndex):
        cols = []
        for parts in df.columns:
            parts = [str(p).strip() for p in parts]
            parts = [p for p in parts if p and not p.startswith("Unnamed")]
            cols.append("__".join(dict.fromkeys(parts)) or "col")
        df.columns = cols
    else:
        df.columns = [str(c).strip() for c in df.columns]
    # some tables repeat column names
    seen = {}
    new_cols = []
    for c in df.columns:
        new_cols.append(c if c not in seen else f"{c}__dup{seen[c]}")
        seen[c] = seen.get(c, 0) + 1
    df.columns = new_cols
    return df


def find_col(df, *names):
    for name in names:
        if name in df.columns:
            return name
    for name in names:
        hits = [c for c in df.columns if c.split("__")[-1] == name]
        if len(hits) == 1:
            return hits[0]
    return None


def get_bref(name, page):
    cached = RAW / f"{name}.csv"
    if cached.exists():
        return pd.read_csv(cached)

    url = BREF.format(YEAR, page)
    print("downloading", url)
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    for table in pd.read_html(StringIO(r.text)):
        table = flatten(table)
        player_col = find_col(table, "Player")
        if player_col:
            break
    else:
        raise RuntimeError(f"couldn't find the player table on {url}")

    table = table.rename(columns={player_col: "Player"})
    table["Player"] = table["Player"].map(clean_name)
    table = table[~table["Player"].isin(["Player", "League Average"])]
    table.to_csv(cached, index=False)
    time.sleep(3)  # bref rate limits you if you go too fast
    return table


def one_row_per_player(df):
    # traded players have a row per team plus a combined "2TM" row, keep the combined one
    team = find_col(df, "Team", "Tm")
    if team is None:
        return df.drop_duplicates("Player")
    combined = df[team].astype(str).str.match(r"^\d+TM$")
    df = pd.concat([df[combined], df[~combined]])
    return df.drop_duplicates("Player").reset_index(drop=True)


def height_to_inches(h):
    m = re.match(r"^(\d+)\s*[-']\s*(\d+)", str(h))
    if m:
        return int(m.group(1)) * 12 + int(m.group(2))
    try:
        h = float(h)
        return h if h > 50 else np.nan
    except ValueError:
        return np.nan


def get_bios():
    cached = RAW / "nba_bios.csv"
    if cached.exists():
        bios = pd.read_csv(cached)
    else:
        try:
            from nba_api.stats.endpoints import leaguedashplayerbiostats
            print("downloading NBA.com bios")
            bios = leaguedashplayerbiostats.LeagueDashPlayerBioStats(
                season=SEASON, per_mode_simple="PerGame", timeout=60
            ).get_data_frames()[0]
            bios.to_csv(cached, index=False)
        except Exception as e:
            print("couldn't get NBA.com bios, skipping phase 3:", e)
            return None

    return pd.DataFrame({
        "Player": bios["PLAYER_NAME"].map(clean_name),
        "height_in": bios["PLAYER_HEIGHT"].map(height_to_inches),
        "weight_lb": pd.to_numeric(bios["PLAYER_WEIGHT"], errors="coerce"),
    }).drop_duplicates("Player")


def build_dataset():
    t = {name: one_row_per_player(get_bref(name, page)) for name, page in PAGES.items()}

    def stat(table, *names):
        col = find_col(t[table], *names)
        if col is None:
            return pd.Series(np.nan, index=t[table].index)
        return pd.to_numeric(t[table][col], errors="coerce")

    def add(df, table, cols):
        extra = pd.DataFrame({"Player": t[table]["Player"]})
        for new, names in cols.items():
            extra[new] = stat(table, *names)
        return df.merge(extra, on="Player", how="left")

    pg = t["per_game"]
    df = pd.DataFrame({
        "Player": pg["Player"],
        "team": pg[find_col(pg, "Team", "Tm")].astype(str),
        "pos": pg[find_col(pg, "Pos")].astype(str),
    })
    df = add(df, "per_game", {
        "age": ["Age"], "games": ["G"], "minutes_per_game": ["MP"],
        "pg_pts": ["PTS"], "pg_trb": ["TRB"], "pg_ast": ["AST"],
    })
    # per game MP is minutes per game, the advanced table has season totals
    df = add(df, "advanced", {"minutes": ["MP"]})
    df = add(df, "per100", {f"p100_{s.lower()}": [s] for s in ["PTS", "TRB", "AST", "STL", "BLK", "TOV"]})
    df = add(df, "advanced", {
        "adv_" + s.lower().replace("%", "pct"): [s]
        for s in ["TS%", "3PAr", "FTr", "ORB%", "DRB%", "AST%", "STL%", "BLK%", "TOV%", "USG%", "BPM"]
    })
    df = add(df, "pbp", {"pos_" + s.lower().replace("%", "pct"): [s] for s in ["PG%", "SG%", "SF%", "PF%", "C%"]})
    df = add(df, "shooting", {
        "shot_dist": ["Dist."],
        "shot_0_3_freq": ["% of FGA by Distance__0-3"],
        "shot_3_10_freq": ["% of FGA by Distance__3-10"],
        "shot_10_16_freq": ["% of FGA by Distance__10-16"],
        "shot_16_3p_freq": ["% of FGA by Distance__16-3P"],
        "shot_3p_freq": ["% of FGA by Distance__3P"],
        "ast_2p_rate": ["% of FG Ast'd__2P"],
        "ast_3p_rate": ["% of FG Ast'd__3P"],
        "dunk_fga_rate": ["Dunks__%FGA"],
        "corner3_share": ["Corner 3s__%3PA"],
    })

    bios = get_bios()
    if bios is not None:
        df = df.merge(bios, on="Player", how="left")
    return df.drop_duplicates("Player").reset_index(drop=True)


# ---------- the actual model ----------

def distance(X):
    # z-score every stat, then distance from the league average (0)
    z = (X - X.mean()) / X.std(ddof=0).replace(0, np.nan)
    return np.sqrt((z ** 2).mean(axis=1))


def save(df, score, filename, cols):
    board = df.sort_values(score).copy()
    board.insert(0, "Rank", range(1, len(board) + 1))
    cols = ["Rank", "Player", score] + [c for c in cols if c in board]
    board[cols].to_csv(OUT / filename, index=False)
    print(f"{filename}: {board.iloc[0]['Player']} ({board.iloc[0][score]:.4f})")


def fill_missing(df, stats, min_share):
    # drop players missing too much, fill small gaps with the median
    df = df[df[stats].notna().mean(axis=1) >= min_share].copy()
    df[stats] = df[stats].fillna(df[stats].median())
    return df


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    df = build_dataset()
    print(f"{len(df)} players\n")
    info = ["team", "pos", "minutes", "minutes_per_game", "games"]

    # Phase 1: just points, rebounds, assists per game, everyone counts
    stats = ["pg_pts", "pg_trb", "pg_ast"]
    d = df.dropna(subset=stats).copy()
    d["basic_v1"] = distance(d[stats])
    save(d, "basic_v1", "01_phase1_caveman_average.csv", info)

    # Phase 2: per 100 possessions, with a minutes cutoff
    per100 = ["p100_pts", "p100_trb", "p100_ast"]
    for cutoff in [250, 500, 750, 1000]:
        d = df[df.minutes >= cutoff].dropna(subset=per100).copy()
        d["basic_v2"] = distance(d[per100])
        save(d, "basic_v2", f"02_phase2_per100_min{cutoff}.csv", info)

    # everything from here on uses 500+ minutes
    pool = df[df.minutes >= 500].copy()

    # Phase 3: height, weight, age. Then half that, half phase 2
    body = [c for c in ["height_in", "weight_lb", "age"] if c in pool and pool[c].notna().sum() > 50]
    if len(body) >= 2:
        d = pool.dropna(subset=body).copy()
        d["physical_distance"] = distance(d[body])
        save(d, "physical_distance", "03a_phase3_physical_only.csv", ["team", "pos"] + body)

        d = pool.dropna(subset=per100 + body).copy()
        d["basic_v3"] = (distance(d[per100]) + distance(d[body])) / 2
        save(d, "basic_v3", "03b_phase3_production_plus_physical.csv", info + body)
    else:
        print("no height/weight data, skipping phase 3")

    # Phase 4: a bunch more box score + advanced stats, all weighted equally
    full = [
        "p100_pts", "p100_trb", "p100_ast", "p100_stl", "p100_blk", "p100_tov",
        "adv_tspct", "adv_orbpct", "adv_drbpct", "adv_astpct", "adv_stlpct",
        "adv_blkpct", "adv_tovpct", "adv_usgpct",
    ]
    d = fill_missing(pool, full, 0.9)
    d["basic_v4"] = distance(d[full])
    save(d, "basic_v4", "04_phase4_full_equal_features.csv", info)

    # Phase 5: add shot profile / play style
    style = [
        "shot_dist", "shot_0_3_freq", "shot_3_10_freq", "shot_10_16_freq",
        "shot_16_3p_freq", "shot_3p_freq", "ast_2p_rate", "ast_3p_rate",
        "dunk_fga_rate", "corner3_share", "adv_3par",
    ]
    style = [s for s in style if pool[s].notna().sum() >= len(pool) / 2]
    d = fill_missing(pool, full + style, 0.85)
    d["basic_v5"] = distance(d[full + style])
    save(d, "basic_v5", "05_phase5_add_play_style.csv", info)

    # phases 6-8 = the final model, see simulation.py


if __name__ == "__main__":
    main()
