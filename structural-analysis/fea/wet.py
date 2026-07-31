import pickle, numpy as np, trimesh
from scipy.spatial import cKDTree
d=pickle.load(open("pre.pkl","rb")); X=d["X"]; E=d["E"]; B=d["B"]
FC=[(0,1,2),(0,3,1),(1,3,2),(2,3,0)]; opp=[3,2,0,1]
env=trimesh.load("../cfd/envelope.stl")
pts,fi=trimesh.sample.sample_surface_even(env,2000000); en=env.face_normals[fi]
te=cKDTree(pts)
W={}
for k in B:
    eid,fid,f=B[k]
    c=X[f].mean(1); n=np.cross(X[f[:,1]]-X[f[:,0]],X[f[:,2]]-X[f[:,0]])
    x4=X[E[k][eid,np.array(opp)[fid]]]
    s=np.sign(np.einsum('ij,ij->i',n,c-x4)); n=n*s[:,None]
    a=0.5*np.linalg.norm(n,axis=1); u=n/(2*a[:,None]+1e-30)
    dd,j=te.query(c)
    ok=(dd<3.0)&(np.einsum('ij,ij->i',u,en[j])>0.3)
    W[k]=dict(ok=ok,u=u,a=a,c=c,flip=s<0)
    print(k,"outward wetted faces",ok.sum(),"area cm2",round(a[ok].sum()/100,1))
d["W"]=W; pickle.dump(d,open("pre.pkl","wb"))
