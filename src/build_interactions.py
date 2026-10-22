"""Join viewing logs (dataset11-30.csv) with the EPG schedule (export_arh_*.csv)
to build user-program interactions.

Matching rule: a log session (user, channel, [start, stop)) is matched to every
schedule item on the same channel whose airing interval [s_start, s_start+duration)
overlaps it. The match is kept if the overlap >= min(300 s, 0.5 * programme duration),
i.e. the user watched at least 5 minutes, or at least half of a short programme.
Missing programme IDs (tv_show_id == 0, ~89% of schedule rows) cannot be recommended
(no ID in the submission space), so they are excluded from the interaction matrix,
but their watch time is kept per category for the content-based user profile.

Usage: python src/build_interactions.py <data_dir> <out_dir>
"""
import sys
import numpy as np
import pandas as pd

MIN_OVERLAP = 300  # seconds


def load(data_dir):
    logs = pd.read_csv(f"{data_dir}/dataset11-30.csv")
    logs.columns = ["user_id", "channel_id", "start", "stop", "dur"]
    logs["user_id"] = logs.user_id.astype("uint64")
    logs["start"] = pd.to_datetime(logs.start)
    logs["stop"] = pd.to_datetime(logs.stop)
    files = ["export_arh_11-20-final.csv", "export_arh_21-30-final.csv", "export_arh_31-42-final.csv"]
    s = pd.concat([pd.read_csv(f"{data_dir}/{f}", low_memory=False) for f in files])
    s["start"] = pd.to_datetime(s.start_time, format="%d.%m.%Y %H:%M:%S")
    s = s.drop_duplicates(["channel_id", "start"]).sort_values(["channel_id", "start"]).reset_index(drop=True)
    s["end"] = s.start + pd.to_timedelta(s.duration, unit="s")
    return logs, s


def join(logs, s):
    out = []
    ls = logs.start.values.astype("datetime64[s]").astype(np.int64)
    le = logs.stop.values.astype("datetime64[s]").astype(np.int64)
    ss_all = s.start.values.astype("datetime64[s]").astype(np.int64)
    se_all = s.end.values.astype("datetime64[s]").astype(np.int64)
    for ch, idx in s.groupby("channel_id").indices.items():
        lmask = np.where(logs.channel_id.values == ch)[0]
        if len(lmask) == 0:
            continue
        ss, se = ss_all[idx], se_all[idx]
        a, b = ls[lmask], le[lmask]
        i0 = np.searchsorted(ss, a, side="right") - 1      # item airing at session start
        i1 = np.searchsorted(ss, b, side="left") - 1       # last item starting before session end
        i0 = np.clip(i0, 0, None)
        n = np.clip(i1 - i0 + 1, 0, None)
        rep = np.repeat(np.arange(len(lmask)), n)
        offs = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
        item = i0[rep] + offs
        ov = np.minimum(b[rep], se[item]) - np.maximum(a[rep], ss[item])
        progdur = se[item] - ss[item]
        keep = ov >= np.minimum(MIN_OVERLAP, 0.5 * progdur)
        out.append(pd.DataFrame({"log_idx": lmask[rep][keep], "sched_idx": idx[item][keep], "sec": ov[keep]}))
    m = pd.concat(out, ignore_index=True)
    m["user_id"] = logs.user_id.values[m.log_idx]
    m["ts"] = s.start.values[m.sched_idx]
    for c in ["tv_show_id", "tv_show_category", "tv_show_title", "channel_id"]:
        m[c] = s[c].values[m.sched_idx]
    return m


if __name__ == "__main__":
    data_dir, out_dir = sys.argv[1], sys.argv[2]
    logs, s = load(data_dir)
    m = join(logs, s)
    m.drop(columns=["log_idx"]).to_parquet(f"{out_dir}/matches.parquet")
    s.to_parquet(f"{out_dir}/schedule.parquet")
    print("log sessions:", len(logs), "matched rows:", len(m),
          "sessions matched:", m.log_idx.nunique(), f"({m.log_idx.nunique()/len(logs):.1%})",
          "rows with tv_show_id==0:", f"{(m.tv_show_id==0).mean():.1%}")
