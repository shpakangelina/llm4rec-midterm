import sys; sys.path.insert(0,'repo/src')
import numpy as np, pandas as pd, recsys as R, warnings; warnings.filterwarnings('ignore')
m=pd.read_parquet('data/matches.parquet'); s=pd.read_parquet('data/schedule.parquet'); s['week']=s.start.dt.isocalendar().week.astype(int)
m=m.merge(s[['channel_id','start','duration']].rename(columns={'start':'ts'}),on=['channel_id','ts'],how='left')
m['frac']=m.sec/m.duration
m=m[m.tv_show_id!=0].copy(); m['week']=m.ts.dt.isocalendar().week.astype(int)
m.to_parquet('data/matches_frac.parquet')
print(m.frac.describe())
tu=pd.read_csv('/mnt/user-data/uploads/sweettv-tv-program-recommender/submission.csv'); users_all=np.array(tu.user_id.astype('uint64').values,dtype=np.uint64)
def counts(df,thr): return df[df.frac>=thr].groupby(['user_id','tv_show_id']).size().rename('w').reset_index()
def score(tr,users,cands,thr,pop_w=1e-3):
    c=counts(tr,thr)
    pos=pd.Series(np.arange(len(cands)),index=cands); upos=pd.Series(np.arange(len(users)),index=users)
    u=c[c.user_id.isin(users)&c.tv_show_id.isin(cands)]
    H=np.zeros((len(users),len(cands))); H[upos[u.user_id].values,pos[u.tv_show_id].values]=u.w.values
    pop=c.groupby('tv_show_id').user_id.nunique().reindex(cands).fillna(0).values
    return H+pop_w*pop/pop.max(), H, pop
# validation: train <25, truth = full-watch(thr) in 25-30
for thr in [0.5,0.8]:
    tr,va=m[m.week<25],m[m.week>=25]
    v=counts(va[va.user_id.isin(set(users_all))],thr)
    truth={np.uint64(k):set(g.tv_show_id) for k,g in v.groupby('user_id')}
    users=np.array(sorted(truth),dtype=np.uint64)
    cands=np.array(sorted(s[(s.week>=25)&(s.week<=30)&(s.tv_show_id!=0)].tv_show_id.unique()))
    S,H,pop=score(tr,users,cands,thr)
    P=np.tile(pop.astype(float),(len(users),1))
    ev=lambda X: round(R.evaluate(dict(zip(users,R.topk(X,cands))),truth)['MAP5'],4)
    print(f'thr {thr}: users {len(users)} | count+popfill {ev(S)} | pop {ev(P)}')
