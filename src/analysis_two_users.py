import sys; sys.path.insert(0,__file__.rsplit('/',1)[0])
import numpy as np, pandas as pd, recsys as R
pd.set_option('display.width',250); pd.set_option('display.max_colwidth',45)
m=pd.read_parquet('data/matches.parquet'); sched=pd.read_parquet('data/schedule.parquet')
mm=m[m.tv_show_id!=0].copy(); mm['week']=mm.ts.dt.isocalendar().week.astype(int)
tr,va=mm[mm.week<25],mm[mm.week>=25]
meta=R.show_meta(sched)
rec=pd.read_csv('results/val_top5_all_methods.csv',dtype={'user_id':'uint64'})
P=pd.read_csv('peruser.csv',dtype={'user_id':'uint64'}).set_index('user_id')
out=[]
for u in [12684833855980372679,1150084824203709165]:
    u=np.uint64(u); print('\n######## USER',u); print(P.loc[u].to_dict())
    h=tr[tr.user_id==u]
    print('train: sessions-matched rows',len(h),'shows',h.tv_show_id.nunique(),'hours',round(h.sec.sum()/3600,1))
    print('category share:',(h.groupby('tv_show_category').sec.sum()/h.sec.sum()).round(2).sort_values(ascending=False).head(5).to_dict())
    hs=h.groupby('tv_show_id').sec.sum().sort_values(ascending=False).head(6)
    print(meta.loc[hs.index,['tv_show_title','tv_show_category','tv_show_genre_1']].assign(hours=(hs/3600).round(1)))
    t=set(va[va.user_id==u].tv_show_id)
    vs=va[va.user_id==u].groupby('tv_show_id').sec.sum().sort_values(ascending=False).head(5)
    print('VAL top watched:'); print(meta.loc[vs.index,['tv_show_title','tv_show_category']].assign(hours=(vs/3600).round(1)))
    for mt in ['Popularity','CBF','ALS','Hybrid']:
        r=rec[(rec.user_id==u)&(rec.method==mt)].sort_values('rank')
        x=meta.loc[r.tv_show_id,['tv_show_title','tv_show_category']].copy(); x['hit']=[i in t for i in r.tv_show_id]; x['in_train_hist']=x.index.isin(h.tv_show_id)
        print('--',mt,'AP=',round(P.loc[u,mt],3)); print(x)
        for i,(sid,row) in enumerate(x.iterrows(),1): out.append((str(u),mt,i,sid,row.tv_show_title,row.tv_show_category,row.hit,row.in_train_hist))
pd.DataFrame(out,columns=['user_id','method','rank','tv_show_id','title','category','hit_in_val','in_train_history']).to_csv('results/two_users_top5.csv',index=False)
