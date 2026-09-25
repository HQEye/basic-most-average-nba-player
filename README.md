# B.A.S.I.C. - Finding the Most Average NBA Player

**B**asketball **A**verageness **S**imilarity **I**ndex **C**alculation

This is the code and data from my video where I tried to find the most average player in the NBA (2025-26 season). A lot of you asked for the files, so here they are.

Quick note on what "average" means here: I'm not looking for a guy who's average at everything. I'm looking for the player whose overall statistical profile sits closest to the center of the league. You can be great at one thing and bad at another and still end up there. It's not a ranking of who's good.

## The result

| | Player | Avg rank | #1 finishes | Top 5 finishes |
|---:|---|---:|---:|---:|
| 1 | Jaylon Tyson | 4.12 | 2,719 | 7,446 |
| 2 | Tristan da Silva | 5.40 | 3,253 | 6,505 |
| 3 | Quentin Grimes | 6.39 | 205 | 5,152 |

Out of 10,000 models. Da Silva actually won more individual runs than Tyson, but Tyson was more consistently near the top no matter how you defined "average", and consistency is what the final ranking is based on.

## Look up a player

No installs needed, just Python:

```bash
python explore.py leaderboard --top 10
python explore.py player "Jaylon Tyson"
python explore.py compare "Jaylon Tyson" "Tristan da Silva" "Kyle Kuzma"
```

(On Windows you might need `py` instead of `python`.)

Or just open `data/final_rankings.csv` in Excel / Google Sheets.

## Run the model yourself

```bash
pip install -r requirements.txt
python simulation.py
```

Takes about 10 seconds. It runs all 10,000 models and saves the rankings to `results/finalists.csv`, which should match `data/final_rankings.csv` exactly. It runs on the saved stats in `data/`, so you'll get the same answer every time.

Some options if you want to mess around:

- `--save-run-ranks` saves every player's rank in every single run (big file), if you want to try your own way of combining them
- `--runs 1000` for a smaller run
- `--seed 123` for a different random seed (the video used the default, 20260914)

Or just edit the categories and settings at the top of `simulation.py` and see what changes.

## How it works

I built it up in phases. The phase 1-5 leaderboards from the video are in `data/phases/`, and `phases.py` is the code for them:

1. **Points, rebounds, assists per game.** Gui Santos.
2. **Same thing per 100 possessions, with minutes cutoffs.** Gui Santos at 250-750 minutes, Kyle Kuzma at 1000.
3. **Height, weight, and age (alone and combined with phase 2).** Tyrese Martin both times.
4. **A lot more stats: efficiency, rebounding %, usage, etc.** Jaden Ivey.
5. **Add shot profile (where the shots come from).** Kyle Kuzma.

Phases 6-8 are the final model in `simulation.py`. For that, stats are grouped into 9 categories (scoring, efficiency, playmaking, rebounding, defense, role, shot style, physical, and impact/BPM) so one category with a ton of stats can't dominate everything. Each player gets a distance from the league's center in each category, and those get weighted and added up. Lowest total = most average.

The problem is there are a bunch of reasonable ways to define "center" and "distance", and each one can give a different winner. So instead of picking one, I ran 10,000 models, each with randomly chosen:

- minutes cutoff (250 / 500 / 750 / 1000)
- center (mean / median / trimmed mean)
- scaling (standard deviation / MAD)
- distance (Euclidean / Manhattan)
- which stats are included (each one has an 85% chance)
- category weights (random, but around equal)

Players who qualified in basically every run were then ranked by their median rank, with average rank as the tiebreaker.

### Running the phases

```bash
python phases.py
```

This one pulls the stats live from Basketball-Reference and NBA.com, so it needs internet, and it might break if those sites change or block you. Stats also get updated during the season, so your numbers might be slightly off from `data/phases/`. Results go in `phase_results/`.

## Files

```
simulation.py         the final model
phases.py             phases 1-5
explore.py            look up players in the results
data/
  final_rankings.csv  the final 272-player ranking
  model_input.csv     the stats the model runs on
  bios.csv            height/weight for players missing them
  phases/             leaderboards from phases 1-5
tests/                checks that the results match
```

## Credits

Stats are from [Basketball-Reference](https://www.basketball-reference.com/leagues/NBA_2026.html) and NBA.com. I used AI in VS Code to help debug the code. The idea, 75% of code, the statistical methodology, and all the decisions about what counts as "average" are mine.
