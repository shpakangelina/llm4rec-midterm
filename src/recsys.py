"""Recommenders for SWEET.TV: popularity baseline, content-based filtering (CBF),
matrix factorization (implicit ALS) and a score-level hybrid.

Final hybrid = Pop + 1.5*History + 1.0*ALS + 0*CBF (min-max per user); weights tuned on weeks 19-24 (see validate.py).

All methods share the same inputs:
  train  - matched interactions (user_id, tv_show_id, sec, ts) with tv_show_id != 0
  cands  - array of tv_show_id airing in the target period (same candidate pool for all)
and return {user_id: [top-5 tv_show_id]} for the requested users.
"""
import re
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

K = 5


def user_item(train, half_life_days=None):
    """Aggregate watch seconds per (user, show); optional exponential time decay."""
    w = train.sec.astype(float)
    if half_life_days:
        age = (train.ts.max() - train.ts).dt.total_seconds() / 86400
        w = w * 0.5 ** (age / half_life_days)
    return train.assign(w=w).groupby(["user_id", "tv_show_id"]).w.sum().reset_index()


def topk(scores, items, k=K):
    idx = np.argpartition(-scores, k, axis=1)[:, :k]
    part = np.take_along_axis(scores, idx, 1)
    order = np.argsort(-part, axis=1)
    return items[np.take_along_axis(idx, order, 1)]


# ---------------------------------------------------------------- popularity
def popularity_scores(ui, users, cands):
    """Reach popularity: number of distinct users who watched the show (>= matching threshold)."""
    pop = ui.groupby("tv_show_id").user_id.nunique().reindex(cands).fillna(0).values.astype(float)
    return np.tile(pop / (pop.max() or 1), (len(users), 1))


# ---------------------------------------------------------------- personal re-watch signal
def history_scores(ui, users, cands):
    """log(1 + decayed minutes) the user already spent on each candidate show (0 if never watched)."""
    pos = pd.Series(np.arange(len(cands)), index=cands); upos = pd.Series(np.arange(len(users)), index=users)
    u = ui[ui.user_id.isin(users) & ui.tv_show_id.isin(cands)]
    H = np.zeros((len(users), len(cands)))
    H[upos[u.user_id].values, pos[u.tv_show_id].values] = np.log1p(u.w.values / 60)
    return H


# ---------------------------------------------------------------- content-based
def item_text(meta):
    """Bag of metadata tokens per show: category, genres, channel, director, actors, title words."""
    def tok(x, p):
        return " ".join(f"{p}_{t}" for t in re.findall(r"\w+", str(x).lower())) if pd.notna(x) else ""
    t = ("cat_" + meta.tv_show_category.fillna("na").str.replace(" ", "_") + " "
         + meta.tv_show_genre_1.fillna("").map(lambda x: tok(x, "g")) + " "
         + meta.tv_show_genre_2.fillna("").map(lambda x: tok(x, "g")) + " "
         + meta.tv_show_genre_3.fillna("").map(lambda x: tok(x, "g")) + " "
         + "ch_" + meta.channel_id.astype(str) + " "
         + meta.director.map(lambda x: tok(x, "dir")) + " "
         + meta.actors.map(lambda x: tok(x, "act")) + " "
         + meta.tv_show_title.map(lambda x: tok(x, "t")))
    return t


def show_meta(schedule):
    """One metadata row per show id (most frequent channel; first non-null attributes)."""
    s = schedule[schedule.tv_show_id != 0]
    ch = s.groupby("tv_show_id").channel_id.agg(lambda x: x.value_counts().index[0])
    meta = s.groupby("tv_show_id").first()[["tv_show_title", "tv_show_category", "tv_show_genre_1",
                                            "tv_show_genre_2", "tv_show_genre_3", "director", "actors"]]
    meta["channel_id"] = ch
    return meta


def cbf_scores(ui, users, cands, meta):
    items = meta.index.values
    vec = TfidfVectorizer(token_pattern=r"\S+", sublinear_tf=True, min_df=2)
    X = normalize(vec.fit_transform(item_text(meta)))          # items x features
    pos = pd.Series(np.arange(len(items)), index=items)
    u = ui[ui.user_id.isin(users) & ui.tv_show_id.isin(items)]
    upos = pd.Series(np.arange(len(users)), index=users)
    R = sp.csr_matrix((np.log1p(u.w.values / 60), (upos[u.user_id].values, pos[u.tv_show_id].values)),
                      shape=(len(users), len(items)))
    P = normalize(R @ X)                                        # user profiles
    C = X[pos.reindex(cands).fillna(-1).astype(int).clip(lower=0).values]
    S = (P @ C.T).toarray()
    S[:, ~np.isin(cands, items)] = 0
    return S


# ---------------------------------------------------------------- ALS (matrix factorization)
def als_scores(ui, users, cands, factors=16, reg=0.05, iters=15, alpha=0.5, seed=42):
    from implicit.als import AlternatingLeastSquares
    uids = ui.user_id.unique(); iids = ui.tv_show_id.unique()
    upos = pd.Series(np.arange(len(uids)), index=uids); ipos = pd.Series(np.arange(len(iids)), index=iids)
    conf = alpha * np.log1p(ui.w.values / 60)
    R = sp.csr_matrix((conf.astype(np.float32), (upos[ui.user_id].values, ipos[ui.tv_show_id].values)),
                      shape=(len(uids), len(iids)))
    model = AlternatingLeastSquares(factors=factors, regularization=reg, iterations=iters,
                                    random_state=seed, use_gpu=False)
    model.fit(R, show_progress=False)
    U = np.zeros((len(users), factors), np.float32)
    known = np.isin(users, uids)
    U[known] = model.user_factors[upos[users[known]].values]
    V = np.zeros((len(cands), factors), np.float32)
    kc = np.isin(cands, iids)
    V[kc] = model.item_factors[ipos[cands[kc]].values]
    return U @ V.T


# ---------------------------------------------------------------- hybrid
def minmax_rows(S):
    lo = S.min(1, keepdims=True); hi = S.max(1, keepdims=True)
    return (S - lo) / np.where(hi - lo > 0, hi - lo, 1)


def hybrid_scores(parts, weights):
    return sum(w * minmax_rows(p) for p, w in zip(parts, weights))


# ---------------------------------------------------------------- metrics
def evaluate(rec, truth):
    """MAP@5, Precision@5, Recall@5, HitRate@5 averaged over users with non-empty truth."""
    ap, pr, rc, hr = [], [], [], []
    for u, t in truth.items():
        r = rec[u]
        hits = np.array([x in t for x in r], float)
        prec_at = np.cumsum(hits) / np.arange(1, len(r) + 1)
        ap.append((prec_at * hits).sum() / min(len(t), K))
        pr.append(hits.sum() / K); rc.append(hits.sum() / len(t)); hr.append(float(hits.any()))
    return dict(MAP5=np.mean(ap), P5=np.mean(pr), R5=np.mean(rc), HR5=np.mean(hr), n=len(ap))
