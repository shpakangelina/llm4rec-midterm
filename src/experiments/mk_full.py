exec(open('fullwatch.py').read().split('# validation')[0].replace("m.to_parquet('data/matches_frac.parquet')",""))
cands=np.array(sorted(s[(s.week>=31)&(s.tv_show_id!=0)].tv_show_id.unique())); valid=set(cands.astype(str))
for thr in [0.5,0.8]:
    S,H,pop=score(m,users_all,cands,thr)
    top=R.topk(S,cands)
    sub=pd.DataFrame({'user_id':tu.user_id.astype(str),'tv_show_id':[' '.join(map(str,r)) for r in top]})
    ids=sub.tv_show_id.str.split(' '); assert ids.map(lambda r:len(set(r))==5 and set(r)<=valid).all() and len(sub)==1260
    name=f'fullwatch{int(thr*100)}'; sub.to_csv(f'/mnt/user-data/outputs/submission_{name}.csv',index=False); print(name,'ok')
