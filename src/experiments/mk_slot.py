import sys; sys.path.insert(0,'repo/src'); sys.path.insert(0,'.')
from slot import *
m=pd.read_parquet('data/matches.parquet'); m=m[m.tv_show_id!=0].copy()
t=pd.read_csv('/mnt/user-data/uploads/sweettv-tv-program-recommender/submission.csv')
users=np.array(t.user_id.astype('uint64').values,dtype=np.uint64)
St=sched[sched.week>=31]
cands=np.array(sorted(St[St.tv_show_id!=0].tv_show_id.unique()))
SL=slot_scores(L,users,St,cands,20,12); np.save('SLtest.npy',SL)
ui=R.user_item(m); P=R.popularity_scores(ui,users,cands); H=R.history_scores(ui,users,cands)
valid=set(cands.astype(str))
for name,S in [('slot',SL),('slot_pop',R.hybrid_scores([SL,P],[1,0.3])),('slot_hist',R.hybrid_scores([SL,H],[1,0.3]))]:
    top=R.topk(S,cands)
    sub=pd.DataFrame({'user_id':t.user_id.astype(str),'tv_show_id':[' '.join(map(str,r)) for r in top]})
    ids=sub.tv_show_id.str.split(' '); assert ids.map(lambda r:len(set(r))==5 and set(r)<=valid).all() and len(sub)==1260
    sub.to_csv(f'/mnt/user-data/outputs/submission_{name}.csv',index=False); print(name,'ok', 'zero-score users',(S.max(1)==0).sum())
