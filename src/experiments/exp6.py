exec(open('exp5.py').read().split('# tuning split')[0])
import pickle
u1,c1,S1,T1=setup(19,19,24)
grid=list(itertools.product([0.1,0.25,0.5],[2,4,8,16],[0,0.25,0.5,1]))
res=sorted(((ev(u1,c1,R.hybrid_scores([S1['Pop'],S1['H'],S1['ALS']],w),T1['top10']),w) for w in grid),reverse=True)[:3]; print('tune top10',res)
W=res[0][1]
u2,c2,S2,T2=pickle.load(open('exp5.pkl','rb'))
for name,w in [('v2 (Pop+1.5H14+ALS)',None),('HistOnly+fallback',(0.01,1,0)),('v3 tuned',W)]:
    Sx=R.hybrid_scores([S2['Pop'],S2['H14'],S2['ALS']],(1,1.5,1)) if w is None else R.hybrid_scores([S2['Pop'],S2['H'],S2['ALS']],w)
    print(name,{t:round(ev(u2,c2,Sx,T2[t]),3) for t in ['any','top5','top10']})
pickle.dump(W,open('W3.pkl','wb'))
