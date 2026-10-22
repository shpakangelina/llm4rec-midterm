import sys; sys.path.insert(0,'repo/src')
import numpy as np, pandas as pd, scipy.sparse as sp, recsys as R, warnings; warnings.filterwarnings('ignore')
L=pd.read_parquet('data/logs.parquet'); sched=pd.read_parquet('data/schedule.parquet')
sched['week']=sched.start.dt.isocalendar().week.astype(int)
L['week']=L.start.dt.isocalendar().week.astype(int)
NS=405*0+1
chs=np.array(sorted(set(sched.channel_id)|set(L.channel_id))); cpos=pd.Series(np.arange(len(chs)),index=chs)
def slot_minutes(start, stop, ch):
    """explode intervals into hourly slots (channel, dow, hour) with minutes; returns DataFrame idx, slot, min"""
    a=start.values.astype('datetime64[s]').astype(np.int64); b=stop.values.astype('datetime64[s]').astype(np.int64)
    h0=a//3600; h1=(b-1)//3600; n=np.clip(h1-h0+1,1,24*7)
    rep=np.repeat(np.arange(len(a)),n); hh=h0[rep]+(np.arange(n.sum())-np.repeat(np.cumsum(n)-n,n))
    ov=(np.minimum(b[rep],(hh+1)*3600)-np.maximum(a[rep],hh*3600))/60
    dow=((hh//24)+3)%7   # 1970-01-01 was Thursday -> 0=Mon
    hod=hh%24
    slot=cpos[ch.values[rep]].values*168+dow*24+hod
    return rep, slot, np.clip(ov,0,None)
def user_profile(Lt, users):
    upos=pd.Series(np.arange(len(users)),index=users); Lt=Lt[Lt.user_id.isin(users)]
    rep,slot,mins=slot_minutes(Lt.start,Lt.stop,Lt.channel_id)
    U=sp.csr_matrix((mins,(upos[Lt.user_id.values[rep]].values,slot)),shape=(len(users),len(chs)*168))
    return U
def show_slots(S, cands, weeks):
    s=S[(S.tv_show_id.isin(cands))]
    rep,slot,mins=slot_minutes(s.start,s.end,s.channel_id)
    ipos=pd.Series(np.arange(len(cands)),index=cands)
    A=sp.csr_matrix((mins/weeks,(ipos[s.tv_show_id.values[rep]].values,slot)),shape=(len(cands),len(chs)*168))
    return A
def slot_scores(Lt, users, S, cands, ntrain_weeks, ntest_weeks):
    U=user_profile(Lt,users)/ntrain_weeks/60.0   # avg minutes per week in slot / 60 = prob watching
    A=show_slots(S,cands,ntest_weeks)            # minutes per week the show occupies the slot
    return (U@A.T).toarray()
if __name__=='__main__':
    m=pd.read_parquet('data/matches.parquet'); m=m[m.tv_show_id!=0].copy(); m['week']=m.ts.dt.isocalendar().week.astype(int)
    tu=set(pd.read_csv('/mnt/user-data/uploads/sweettv-tv-program-recommender/submission.csv').user_id.astype('uint64'))
    va=m[(m.week>=25)]; v=va[va.user_id.isin(tu)].groupby(['user_id','tv_show_id']).sec.sum().reset_index()
    users=np.array(sorted(v.user_id.unique()),dtype=np.uint64)
    cands=np.array(sorted(sched[(sched.week>=25)&(sched.week<=30)&(sched.tv_show_id!=0)].tv_show_id.unique()))
    T={'any':{np.uint64(u):set(g.tv_show_id) for u,g in v.groupby('user_id')},
       'top5':{np.uint64(u):set(g.nlargest(5,'sec').tv_show_id) for u,g in v.groupby('user_id')},
       'top10':{np.uint64(u):set(g.nlargest(10,'sec').tv_show_id) for u,g in v.groupby('user_id')}}
    Ls=L[L.week<25]
    Sv=sched[(sched.week>=25)&(sched.week<=30)]
    SL=slot_scores(Ls,users,Sv,cands,14,6)
    ui=R.user_item(m[m.week<25]); P=R.popularity_scores(ui,users,cands); H=R.history_scores(ui,users,cands)
    ev=lambda Sx,t: round(R.evaluate(dict(zip(users,R.topk(Sx,cands))),t)['MAP5'],3)
    for name,Sx in [('Pop',P),('Hist',H),('Slot',SL),('Slot+0.3Pop',R.hybrid_scores([SL,P],[1,0.3])),('Slot+0.3Hist',R.hybrid_scores([SL,H],[1,0.3]))]:
        print(name,{t:ev(Sx,T[t]) for t in T})
    np.save('SLval.npy',SL)
