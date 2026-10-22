# LLM4Rec Midterm — SWEET.TV TV program recommender

Top-5 TV programs per test user for the Kaggle competition
[sweettv-tv-program-recommender](https://www.kaggle.com/competitions/sweettv-tv-program-recommender) (metric: MAP@5).
Author: Шпак Ангелина Алексеевна (Kaggle: shpakangelina).

## Interaction definition
A user–show interaction is a **full watch**: one viewing session covers ≥50% of an airing (logs × schedule joined on channel and time).
Airing with `tv_show_id = 0` are excluded (irrelevant per the competition data page).

## Methods
| Method | File / function | Role |
|---|---|---|
| Popularity (reach) | `recsys.popularity_scores` | reference baseline |
| Content-based filtering (TF-IDF over category, genres, channel, director, actors, title) | `recsys.cbf_scores` | required method 1 |
| Matrix factorization (implicit ALS, 16 factors, alpha 0.5) | `recsys.als_scores` | required method 2 |
| Personal re-watch signal (count of full watches) | `recsys.history_scores` | hybrid component |
| **Final: Hybrid** = Pop + 1.5·History + 0.5·CBF (+0·ALS), per-user min-max, no time decay; weights tuned on weeks 19–24 | `validate.py` | Kaggle submission (`results/submission_hybrid.csv`, uploaded as `fw_hybrid_nodecay.csv`: private MAP@5 0.40332, public 0.41903) |

## Reproduce
```bash
pip install -r requirements.txt
# put dataset11-30.csv and export_arh_*.csv into data/
python src/build_interactions.py data data          # logs x schedule -> matches.parquet (~40 s)
python src/validate.py data data/submission.csv results   # validation table + submissions (~1.5 min)
# DECAY14=1 python src/validate.py ...   -> variant with 14-day time decay (fw_hybrid.csv, private 0.38892)
```
Outputs in `results/`: `validation_results.csv`, `hybrid_grid_tuning_w19-24.csv`, `config.json`, `val_top5_all_methods.csv`,
`two_users_top5.csv`, `submission_hybrid.csv` (**final Kaggle submission**), `submission_popularity.csv`, `submission_als.csv`, `submission_cbf.csv`.

## Validation protocol
Train weeks 11–24, validate weeks 25–30; truth = full watches in weeks 25–30; Kaggle test users with ≥1 full watch in validation (n=1238);
candidate pool = shows with an ID airing in weeks 25–30 (5013); same users/pool/metric for all methods.
Hybrid weights are tuned on a separate earlier split (train weeks 11–18, tune on 19–24), so weeks 25–30 are a clean hold-out.
Final model retrained on weeks 11–30; candidates = shows airing in weeks 31–42 (6642).

## Exploration history (Appendix B / Table 4 of the report)
`src/experiments/` holds the scripts behind the exploratory Kaggle submissions (run from the repo root with `data/` prepared;
they import `src/recsys.py`). The first submissions used a lenient rule (any ≥5-min overlap, no full-watch filter):
popularity / history / hybrid v2–v3 / channel×time-slot model all scored ≈0.25–0.28 private MAP@5.
Switching to full watches (`fullwatch.py`, `mk_full.py`) raised the score to ≈0.40, and the final pipeline (`validate.py`) uses it.
