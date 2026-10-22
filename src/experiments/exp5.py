import sys; sys.path.insert(0,'repo/src')
import numpy as np, pandas as pd, recsys as R, warnings, itertools; warnings.filterwarnings('ignore')
m=pd.read_parquet('data/matches.parquet'); sched=pd.read_parquet('data/schedule.parquet')
m=m[m.tv_show_id!=0].copy(); m['week']=m.ts.dt.isocalendar().week.astype(int); sched['week']=sched.start.dt.isocalendar().week.astype(int)
tu=set(pd.read_csv('/mnt/user-data/uploads/sweettv-tv-program-recommender/submission.csv').user_id.astype('uint64'))
meta=R.show_meta(sched[sched.week<=30])
def setup(w_tr_end,w_va0,w_va1):
    tr=m[m.week<w_tr_end]; va=m[(m.week>=w_va0)&(m.week<=w_va1)]
    v=va[va.user_id.isin(tu)].groupby(['user_id','tv_show_id']).sec.sum().reset_index()
    users=np.array(sorted(v.user_id.unique()),dtype=np.uint64)
    cands=np.array(sorted(sched[(sched.week>=w_va0)&(sched.week<=w_va1)&(sched.tv_show_id!=0)].tv_show_id.unique()))
    T={'any':{np.uint64(u):set(g.tv_show_id) for u,g in v.groupby('user_id')},
       'top5':{np.uint64(u):set(g.nlargest(5,'sec').tv_show_id) for u,g in v.groupby('user_id')},
       'top10':{np.uint64(u):set(g.nlargest(10,'sec').tv_show_id) for u,g in v.groupby('user_id')}}
    ui=R.user_item(tr)
    S={'Pop':R.popularity_scores(ui,users,cands),'H':R.history_scores(ui,users,cands),'H14':R.history_scores(R.user_item(tr,14),users,cands),
       'ALS':R.als_scores(R.user_item(tr,14),users,cands),'CBF':R.cbf_scores(R.user_item(tr,14),users,cands,meta)}
    return users,cands,S,T
def ev(users,cands,Sx,t): return R.evaluate(dict(zip(users,R.topk(Sx,cands))),t)['MAP5']
# tuning split 11-18 -> 19-24
u1,c1,S1,T1=setup(19,19,24)
grid=list(itertools.product([0.1,0.25,0.5,1],[1,2,4],[0,0.5,1],[0,0.25]))   # pop, H(no decay), ALS, CBF  (H14 off)
for tn in ['any','top5','top10']:
    best=max(grid,key=lambda w: ev(u1,c1,R.hybrid_scores([S1['Pop'],S1['H'],S1['ALS'],S1['CBF']],w),T1[tn]))
    print('tuned on',tn,best, round(ev(u1,c1,R.hybrid_scores([S1['Pop'],S1['H'],S1['ALS'],S1['CBF']],best),T1[tn]),3))
u2,c2,S2,T2=setup(25,25,30)
for name,w in [('HistOnly+popfallback',(0.01,1,0,0)),('tuned_top5',None),('tuned_top10',None)]:
    pass
import pickle; pickle.dump((u2,c2,S2,T2),open('exp5.pkl','wb'))
