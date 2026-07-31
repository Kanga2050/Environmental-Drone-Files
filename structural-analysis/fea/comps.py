import gmsh, numpy as np
gmsh.initialize(); gmsh.option.setNumber("General.Terminal",0)
occ=gmsh.model.occ
occ.importShapes("../cfd/manta.step"); occ.synchronize()
want={"battery":[180],"fc":[189],"vtx":[176,179],"cam":[1],"rx":[74]}
keep=sum(want.values(),[])
occ.remove([(3,t) for d,t in gmsh.model.getEntities(3) if t not in keep],recursive=True); occ.synchronize()
gmsh.option.setNumber("Mesh.MeshSizeMax",2.0); gmsh.option.setNumber("Mesh.MeshSizeMin",0.5)
gmsh.model.mesh.generate(2)
out={}
for k,v in want.items():
    P=[]
    for t in v:
        for d,s in gmsh.model.getBoundary([(3,t)],oriented=False):
            _,c,_=gmsh.model.mesh.getNodes(2,s,includeBoundary=True); P.append(c.reshape(-1,3))
    out[k]=np.vstack(P); print(k,out[k].shape,out[k].min(0).round(1),out[k].max(0).round(1))
np.savez("comps.npz",**out)
