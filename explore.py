"""Look up players in the B.A.S.I.C. final rankings. No installs needed."""
import argparse
import csv
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "data" / "final_rankings.csv"


def load_results(path=RESULTS):
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("final_rankings.csv is empty")
    for row in rows:
        for key in ("FinalistRank", "runs_eligible", "wins", "top5", "top10"):
            row[key] = int(row[key])
        for key in ("average_rank", "median_rank", "rank_sd", "eligibility_rate"):
            row[key] = float(row[key])
    return rows


def fmt(row):
    runs = row["runs_eligible"]
    return (f'{row["FinalistRank"]:>3}. {row["Player"]:<25} '
            f'avg rank {row["average_rank"]:>7.2f}  '
            f'median {row["median_rank"]:>5.0f}  '
            f'#1 {row["wins"]/runs:>6.1%}  '
            f'top 5 {row["top5"]/runs:>6.1%}  '
            f'top 10 {row["top10"]/runs:>6.1%}')


def match_player(rows, query):
    # exact name first, then any unique partial match ("tyson" works)
    q = query.casefold().strip()
    exact = [r for r in rows if r["Player"].casefold() == q]
    if exact:
        return exact[0]
    matches = [r for r in rows if q in r["Player"].casefold()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"couldn't find {query!r}")
    raise ValueError("more than one match: " + ", ".join(r["Player"] for r in matches))


def run():
    parser = argparse.ArgumentParser(
        description="Look up players in the B.A.S.I.C. final rankings."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    leaders = sub.add_parser("leaderboard", help="show the top players")
    leaders.add_argument("--top", type=int, default=10, help="how many (default 10)")
    p = sub.add_parser("player", help="look up one player")
    p.add_argument("name")
    compare = sub.add_parser("compare", help="compare players side by side")
    compare.add_argument("names", nargs="+")
    args = parser.parse_args()
    rows = load_results()
    if args.command == "leaderboard":
        for row in rows[:max(0, args.top)]:
            print(fmt(row))
    elif args.command == "player":
        player = match_player(rows, args.name)
        print(fmt(player))
        print(f'Team: {player["team"]} | Position: {player["pos"]} | '
              f'Season minutes: {float(player["minutes"]):,.0f}')
        print(f'Rank SD: {player["rank_sd"]:.2f} | '
              f'Eligible runs: {player["runs_eligible"]:,} | '
              f'Finalist position: {player["FinalistRank"]}')
    elif args.command == "compare":
        for name in args.names:
            print(fmt(match_player(rows, name)))


if __name__ == "__main__":
    try:
        run()
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"Error: {exc}")
