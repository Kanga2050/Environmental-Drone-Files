import trimesh, numpy as np, scipy.ndimage as nd, pymeshlab as ml, sys
from skimage import measure
p=float(sys.argv[1]); rc=float(sys.argv[2])
m=trimesh.load("body_outer.stl")
lo=m.bounds[0]-10*p; hi=m.bounds[1]+10*p
sh=np.ceil((hi-lo)/p).astype(int)+1
g=np.zeros(sh,bool)
T=m.triangles
for i in range(0,len(T),20000):
    t=T[i:i+20000]
    L=np.linalg.norm(t[:,[1,2,0]]-t,axis=2).max(1)
    k=np.clip(np.ceil(L/(0.4*p)).astype(int),1,200)
    for kk in np.unique(k):
        s=t[k==kk]
        a,b=np.meshgrid(np.arange(kk+1),np.arange(kk+1),indexing='ij'); msk=a+b<=kk
        u=a[msk]/kk; w=b[msk]/kk
        P=s[:,0,None]*(1-u-w)[None,:,None]+s[:,1,None]*u[None,:,None]+s[:,2,None]*w[None,:,None]
        idx=np.floor((P.reshape(-1,3)-lo)/p+0.5).astype(int)
        g[idx[:,0],idx[:,1],idx[:,2]]=True
solid=nd.binary_fill_holes(g)
print("grid",sh,"solid cm3",round(solid.sum()*p**3/1e3,1))
r=max(1,int(round(rc/p)))
bl=np.array(np.meshgrid(*[np.arange(-r,r+1)]*3,indexing='ij'));bl=(bl**2).sum(0)<=r*r
cl=nd.binary_dilation(solid,bl); cl=nd.binary_fill_holes(cl); cl=nd.binary_erosion(cl,bl)
cl=nd.binary_fill_holes(cl|solid)
lab,n=nd.label(cl); sz=nd.sum(cl,lab,range(1,n+1)); cl=lab==(np.argmax(sz)+1)
print("envelope cm3",round(cl.sum()*p**3/1e3,1),"comps",n)
vs,fs,_,_=measure.marching_cubes(np.pad(cl,1).astype(np.float32),0.5)
vs=(vs-1)*p+lo
out=trimesh.Trimesh(vs,fs); trimesh.smoothing.filter_taubin(out,iterations=20); out.export("env_raw.stl")
ms=ml.MeshSet(); ms.load_new_mesh("env_raw.stl"); ms.meshing_remove_duplicate_vertices()
ms.meshing_isotropic_explicit_remeshing(iterations=8,targetlen=ml.PureValue(3.0),featuredeg=60)
ms.compute_selection_by_self_intersections_per_face(); print("selfint",ms.current_mesh().selected_face_number())
ms.save_current_mesh("envelope.stl")
e=trimesh.load("envelope.stl"); print("env faces",len(e.faces),"wt",e.is_watertight,"vol cm3",round(e.volume/1e3,1),"area cm2",round(e.area/100,1),e.bounds.round(1).tolist())
