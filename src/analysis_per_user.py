import sys; sys.path.insert(0,__file__.rsplit('/',1)[0])
import numpy as np, pandas as pd, recsys as R
m=pd.read_parquet('data/matches.parquet'); sched=pd.read_parquet('data/schedule.parquet')
m=m.merge(sched[['channel_id','start','duration']].rename(columns={'start':'ts'}),on=['channel_id','ts'],how='left')
m=m[m.sec>=0.5*m.duration]   # full watch (>=50% of airing)
L=pd.read_parquet('data/logs.parquet')
print('matched watch-seconds share on id0:', round(m[m.tv_show_id==0].sec.sum()/m.sec.sum(),3))
print('id0 categories in matches:\n', m[m.tv_show_id==0].groupby('tv_show_category').sec.sum().div(m[m.tv_show_id==0].sec.sum()).round(3).sort_values(ascending=False).head(6))
print('sessions <300s share', round((L.dur<300).mean(),3))
mm=m[m.tv_show_id!=0].copy(); mm['week']=mm.ts.dt.isocalendar().week.astype(int)
tr,va=mm[mm.week<25],mm[mm.week>=25]
meta=R.show_meta(sched)
rec=pd.read_csv('results/val_top5_all_methods.csv',dtype={'user_id':'uint64'})
tu=set(rec.user_id)
truth=va[va.user_id.isin(tu)].groupby('user_id').tv_show_id.apply(set)
# per-user AP
def ap(r,t):
    h=[x in t for x in r]; s=0;c=0
    for i,x in enumerate(h,1):
        if x: c+=1; s+=c/i
    return s/min(len(t),5)
g=rec.sort_values('rank').groupby(['method','user_id']).tv_show_id.apply(list).unstack(0)
A=pd.DataFrame({mth:[ap(g.loc[u,mth],truth[u]) for u in g.index] for mth in g.columns},index=g.index)
print(A.mean().round(4))
# coverage
print('unique shows in top5 across users:', rec.groupby('method').tv_show_id.nunique().to_dict())
# category mix per method
rc=rec.merge(meta[['tv_show_category']],left_on='tv_show_id',right_index=True)
print(pd.crosstab(rc.method,rc.tv_show_category,normalize='index').round(2))
# user stats
cat_share=tr.groupby(['user_id','tv_show_category']).sec.sum().unstack(fill_value=0); cat_share=cat_share.div(cat_share.sum(1),axis=0)
A['n_train_shows']=tr[tr.user_id.isin(tu)].groupby('user_id').tv_show_id.nunique()
A['top_cat']=cat_share.idxmax(1); A['top_cat_share']=cat_share.max(1)
A['n_truth']=truth.map(len)
A.to_csv('peruser.csv')
print('popularity AP=0 users:', (A.Popularity==0).sum(), ' hybrid AP=0:', (A.Hybrid==0).sum())
print(A.groupby('top_cat')[['Popularity','CBF','ALS','Hybrid']].mean().round(3).assign(n=A.groupby('top_cat').size()))
# candidates: niche user where pop fails, hybrid wins; typical user
c1=A[(A.Popularity<=0.2)&(A.Hybrid>=0.8)&(A.top_cat_share>0.35)].sort_values('top_cat_share',ascending=False).head(8)
print(c1)
c2=A[(A.top_cat.isin(['Инфо','Познавательное','Сериалы']))&(A.Hybrid<A.Popularity)].head(8); print(c2)
