"""Temporal validation + Kaggle submission.

Validation: train on weeks 11-24 (2020-03-09..06-14), validate on weeks 25-30 (06-15..07-26).
Hybrid weights: tuned on a separate earlier split (train 11-18, tune 19-24).
  - users: Kaggle test users who watched >=1 identified show in weeks 25-30
  - candidate pool: shows with an ID airing in weeks 25-30 (from the schedule = metadata, known in advance)
  - truth: shows the user fully watched (>=50% of an airing in one session) in weeks 25-30
  - same users / pool / metrics for every method
Submission: same pipeline retrained on weeks 11-30; candidates = shows airing in weeks 31-42.

Usage: python src/validate.py <data_dir> <submission_template.csv> <out_dir>
"""
import sys, time, json, itertools
import numpy as np
import pandas as pd
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import recsys as R

data_dir, sub_path, out_dir = sys.argv[1:4]
m = pd.read_parquet(f"{data_dir}/matches.parquet")
sched = pd.read_parquet(f"{data_dir}/schedule.parquet")
# relevance / interaction = "full watch": the session covered >= 50% of the airing (FULL_WATCH);
# each full watch counts 1 (sec := 1), so user_item() returns full-watch counts.
FULL_WATCH = 0.5
m = m.merge(sched[["channel_id", "start", "duration"]].rename(columns={"start": "ts"}), on=["channel_id", "ts"], how="left")
m = m[(m.tv_show_id != 0) & (m.sec >= FULL_WATCH * m.duration)].copy()
m["sec"] = 1.0
m["week"] = m.ts.dt.isocalendar().week.astype(int)
sched["week"] = sched.start.dt.isocalendar().week.astype(int)
template = pd.read_csv(sub_path)
test_users = np.array(template.user_id.astype("uint64").values, dtype=np.uint64)
import os
HL = 14 if os.environ.get("DECAY14") else None  # time decay for personal signals: none (final); DECAY14=1 -> half-life 14 days (fw_hybrid.csv variant)


def pool(w0, w1):
    return np.array(sorted(sched[(sched.week >= w0) & (sched.week <= w1) & (sched.tv_show_id != 0)].tv_show_id.unique()))


def all_scores(train, users, cands, meta):
    ui, uid = R.user_item(train), R.user_item(train, HL)
    out, cost = {}, {}
    for name, fn in [("Popularity", lambda: R.popularity_scores(ui, users, cands)),
                     ("History", lambda: R.history_scores(uid, users, cands)),
                     ("CBF", lambda: R.cbf_scores(uid, users, cands, meta)),
                     ("ALS", lambda: R.als_scores(uid, users, cands))]:
        t = time.time(); out[name] = fn(); cost[name] = time.time() - t
    return out, cost


def hybrid(S, wh, wa, wc):
    return R.hybrid_scores([S["Popularity"], S["History"], S["ALS"], S["CBF"]], [1, wh, wa, wc])


if __name__ == "__main__":
    # ---------------- validation
    SPLIT = 25
    train, val = m[m.week < SPLIT], m[m.week >= SPLIT]
    truth = val[val.user_id.isin(set(test_users))].groupby("user_id").tv_show_id.apply(set).to_dict()
    users = np.array(sorted(truth), dtype=np.uint64)
    truth = {np.uint64(k): v for k, v in truth.items()}
    cands = pool(SPLIT, 30)
    meta = R.show_meta(sched[sched.week <= 30])
    print(f"val users {len(users)}, candidates {len(cands)}")
    S, cost = all_scores(train, users, cands, meta)
    results, recs = {}, {}
    for name in S:
        recs[name] = dict(zip(users, R.topk(S[name], cands)))
        results[name] = {**R.evaluate(recs[name], truth), "sec": cost[name]}
    # weights are tuned on an EARLIER, disjoint split (train weeks 11-18, tune on 19-24),
    # so weeks 25-30 stay a clean hold-out for the hybrid as well.
    TUNE = 19
    t_train, t_val = m[m.week < TUNE], m[(m.week >= TUNE) & (m.week < SPLIT)]
    t_truth = t_val[t_val.user_id.isin(set(test_users))].groupby("user_id").tv_show_id.apply(set).to_dict()
    t_users = np.array(sorted(t_truth), dtype=np.uint64)
    t_truth = {np.uint64(k): v for k, v in t_truth.items()}
    t_cands = pool(TUNE, SPLIT - 1)
    St_, _ = all_scores(t_train, t_users, t_cands, meta)
    grid = []
    for wh, wa, wc in itertools.product([0, 0.25, 0.5, 0.75, 1, 1.5], [0, 0.25, 0.5, 1, 1.5, 2], [0, 0.1, 0.25, 0.5]):
        r = R.evaluate(dict(zip(t_users, R.topk(hybrid(St_, wh, wa, wc), t_cands))), t_truth)
        grid.append({"w_hist": wh, "w_als": wa, "w_cbf": wc, **r})
    grid = pd.DataFrame(grid).sort_values("MAP5", ascending=False)
    grid.to_csv(f"{out_dir}/hybrid_grid_tuning_w19-24.csv", index=False)
    print(f"tuning split: users {len(t_users)}, candidates {len(t_cands)}")
    b = grid.iloc[0]; W = (float(b.w_hist), float(b.w_als), float(b.w_cbf))
    # for reference only: what the best weights on the hold-out itself would give (optimistic)
    oracle = max(R.evaluate(dict(zip(users, R.topk(hybrid(S, wh, wa, wc), cands))), truth)["MAP5"]
                 for wh, wa, wc in itertools.product([0, 0.25, 0.5, 0.75, 1, 1.5], [0, 0.25, 0.5, 1, 1.5, 2], [0, 0.1, 0.25, 0.5]))
    print("oracle (weights tuned on hold-out, optimistic):", round(oracle, 4))
    recs["Hybrid"] = dict(zip(users, R.topk(hybrid(S, *W), cands)))
    results["Hybrid"] = {**R.evaluate(recs["Hybrid"], truth), "sec": sum(cost.values())}
    # ablations of the hybrid
    for nm, w in [("Pop+History", (W[0], 0, 0)), ("Pop+ALS", (0, W[1], 0)), ("Pop+CBF(w=0.25)", (0, 0, 0.25))]:
        results[nm] = {**R.evaluate(dict(zip(users, R.topk(hybrid(S, *w), cands))), truth), "sec": 0}
    res = pd.DataFrame(results).T
    print(res.round(4)); print("hybrid weights (hist, als, cbf):", W)
    res.to_csv(f"{out_dir}/validation_results.csv")
    pd.DataFrame([(n, u, r, i) for n, rec in recs.items() for u, lst in rec.items() for r, i in enumerate(lst, 1)],
                 columns=["method", "user_id", "rank", "tv_show_id"]).to_csv(f"{out_dir}/val_top5_all_methods.csv", index=False)
    json.dump({"weights": dict(zip(["w_hist", "w_als", "w_cbf"], map(float, W))), "oracle_MAP5_on_holdout": float(oracle)}, open(f"{out_dir}/config.json", "w"))

    # ---------------- submission: retrain on weeks 11-30, recommend shows airing in weeks 31-42
    cands_t = pool(31, 42)
    meta_t = R.show_meta(sched)
    St, _ = all_scores(m, test_users, cands_t, meta_t)
    for name, Sx in [("hybrid", hybrid(St, *W)), ("popularity", St["Popularity"]), ("als", St["ALS"]), ("cbf", St["CBF"])]:
        top = R.topk(Sx, cands_t)
        sub = pd.DataFrame({"user_id": test_users.astype(str), "tv_show_id": [" ".join(map(str, r)) for r in top]})
        sub.to_csv(f"{out_dir}/submission_{name}.csv", index=False)
        print(name, "submission written", sub.shape, "test candidates", len(cands_t))
