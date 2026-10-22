import sys; sys.path.insert(0,'repo/src')
import numpy as np, pandas as pd, recsys as R, warnings, itertools; warnings.filterwarnings('ignore')
m=pd.read_parquet('data/matches.parquet'); sched=pd.read_parquet('data/schedule.parquet')
m=m[m.tv_show_id!=0].copy(); m['week']=m.ts.dt.isocalendar().week.astype(int); sched['week']=sched.start.dt.isocalendar().week.astype(int)
tr,va=m[m.week<25],m[m.week>=25]
tu=set(pd.read_csv('/mnt/user-data/uploads/sweettv-tv-program-recommender/submission.csv').user_id.astype('uint64'))
v=va[va.user_id.isin(tu)].groupby(['user_id','tv_show_id']).sec.sum().reset_index()
users=np.array(sorted(v.user_id.unique()),dtype=np.uint64)
cands=np.array(sorted(sched[(sched.week>=25)&(sched.week<=30)&(sched.tv_show_id!=0)].tv_show_id.unique()))
meta=R.show_meta(sched[sched.week<=30])
truths={}
truths['any>=5min']={np.uint64(u):set(g.tv_show_id) for u,g in v.groupby('user_id')}
truths['top5_by_time']={np.uint64(u):set(g.nlargest(5,'sec').tv_show_id) for u,g in v.groupby('user_id')}
truths['top10_by_time']={np.uint64(u):set(g.nlargest(10,'sec').tv_show_id) for u,g in v.groupby('user_id')}
truths['>=1h']={np.uint64(u):set(g[g.sec>=3600].tv_show_id) for u,g in v.groupby('user_id') if (g.sec>=3600).any()}
ui=R.user_item(tr); uid=R.user_item(tr,14)
S={'Pop':R.popularity_scores(ui,users,cands),'Hist':R.history_scores(uid,users,cands),'HistNoDecay':R.history_scores(ui,users,cands),
   'CBF':R.cbf_scores(uid,users,cands,meta),'ALS':R.als_scores(uid,users,cands)}
# watch-time popularity
pt=ui.groupby('tv_show_id').w.sum().reindex(cands).fillna(0).values; S['PopTime']=np.tile(pt,(len(users),1))
S['Hybrid_v2']=R.hybrid_scores([S['Pop'],S['Hist'],S['ALS']],[1,1.5,1])
for tn,t in truths.items():
    us=np.array([u for u in users if u in t],dtype=np.uint64); idx=np.isin(users,us)
    row={k:round(R.evaluate(dict(zip(users[idx],R.topk(x[idx],cands))),t)['MAP5'],3) for k,x in S.items()}
    print(tn,len(us),row)
import pickle; pickle.dump((users,cands,S,truths),open('exp4.pkl','wb'))
