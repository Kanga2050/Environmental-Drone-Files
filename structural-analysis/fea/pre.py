import meshio, numpy as np, trimesh, pickle
from scipy.spatial import cKDTree
m=meshio.read("manta_3.0.inp")
X=m.points
E={"can":m.cells[0].data,"fus":m.cells[1].data}
off={"can":1,"fus":1+len(E["can"])}
FC=[(0,1,2),(0,3,1),(1,3,2),(2,3,0)]
def bfaces(T):
    f=np.concatenate([T[:,list(c)] for c in FC]); s=np.sort(f,1)
    _,inv,cnt=np.unique(s,axis=0,return_inverse=True,return_counts=True)
    b=cnt[inv.ravel()]==1
    eid=np.tile(np.arange(len(T)),4)[b]; fid=np.repeat(np.arange(4),len(T))[b]
    return eid,fid,f[b]
B={k:bfaces(E[k]) for k in E}
for k in B: print(k,"boundary faces",len(B[k][0]))
cen=lambda k:X[B[k][2]].mean(1)
area=lambda k:0.5*np.linalg.norm(np.cross(X[B[k][2][:,1]]-X[B[k][2][:,0]],X[B[k][2][:,2]]-X[B[k][2][:,0]]),axis=1)
nrm=lambda k:np.cross(X[B[k][2][:,1]]-X[B[k][2][:,0]],X[B[k][2][:,2]]-X[B[k][2][:,0]])
tf=cKDTree(X[np.unique(B["fus"][2])]); tc=cKDTree(X[np.unique(B["can"][2])])
dc,_=tf.query(cen("can")); df,_=tc.query(cen("fus"))
for tol in (0.3,0.6,1.0,2.0): print("tie tol",tol,"canopy faces",(dc<tol).sum(),"fus faces",(df<tol).sum())
env=trimesh.load("../cfd/envelope.stl")
pts,_=trimesh.sample.sample_surface_even(env,1500000)
te=cKDTree(pts)
wet={k:te.query(cen(k))[0] for k in B}
for k in wet:
    for t in (1,2,3): print(k,"wetted faces within",t,"mm:",(wet[k]<t).sum(),"area cm2",round(area(k)[wet[k]<t].sum()/100,1))
mot=[(0,0,0.0),(200,0,0.0),(0,200,250.0),(200,200,250.0)]
mn=[]
nb=np.unique(B["fus"][2])
for (x,y,z) in mot:
    r=np.hypot(X[nb,0]-x,X[nb,1]-y)
    s=nb[(r<14)&(np.abs(X[nb,2]-z)<0.6)]
    mn.append(s); print("motor",(x,y,z),"pad nodes",len(s), "z", X[s,2].min() if len(s) else None)
pickle.dump(dict(X=X,E=E,off=off,B=B,dc=dc,df=df,wet=wet,mn=mn),open("pre.pkl","wb"))
