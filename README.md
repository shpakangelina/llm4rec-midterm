# LLM4Rec Midterm — SWEET.TV TV program recommender

Top-5 TV programs per test user for the Kaggle competition
[sweettv-tv-program-recommender](https://www.kaggle.com/competitions/sweettv-tv-program-recommender) (metric: MAP@5).
Author: Шпак Ангелина Алексеевна (Kaggle: shpakangelina).

## Methods
| Method | File / function | Role |
|---|---|---|
| Popularity (reach) | `recsys.popularity_scores` | reference baseline |
| Content-based filtering (TF-IDF over category, genres, channel, director, actors, title) | `recsys.cbf_scores` | required method 1 |
| Matrix factorization (implicit ALS, 16 factors, alpha 0.5) | `recsys.als_scores` | required method 2 |
| Personal re-watch signal (decayed minutes already watched) | `recsys.history_scores` | hybrid component |
| **Final: Hybrid** = Pop + 1.5·History + 1.0·ALS (+0·CBF), per-user min-max; weights tuned on weeks 19–24 | `validate.py` | Kaggle submission |

## Reproduce
```bash
pip install -r requirements.txt
# put dataset11-30.csv and export_arh_*.csv into data/
python src/build_interactions.py data data          # logs x schedule -> matches.parquet (~40 s)
python src/validate.py data data/submission.csv results   # validation table + submissions (~1 min)
```
Outputs in `results/`: `validation_results.csv`, `hybrid_grid_tuning_w19-24.csv`, `config.json`, `val_top5_all_methods.csv`,
`two_users_top5.csv`, `submission_hybrid.csv` (**final Kaggle submission**), `submission_popularity.csv`, `submission_als.csv`, `submission_cbf.csv`.

## Validation protocol
Train weeks 11–24, validate weeks 25–30; Kaggle test users with ≥1 identified show in validation (n=1244);
candidate pool = shows with an ID airing in weeks 25–30 (5013); same users/pool/metric for all methods.
Hybrid weights are tuned on a separate earlier split (train weeks 11–18, tune on 19–24), so weeks 25–30 are a clean hold-out.
Final model retrained on weeks 11–30; candidates = shows airing in weeks 31–42 (6642).
