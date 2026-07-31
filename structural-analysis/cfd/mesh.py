import gmsh, sys, time, math
hb=float(sys.argv[1]); out=sys.argv[2]; hf=float(sys.argv[3]) if len(sys.argv)>3 else 220
t0=time.time()
gmsh.initialize(); gmsh.option.setNumber("General.Terminal",1); gmsh.option.setNumber("General.Verbosity",2)
gmsh.option.setNumber("General.NumThreads",24)
gmsh.merge(sys.argv[4] if len(sys.argv)>4 else "body_r3.stl")
bod=[e[1] for e in gmsh.model.getEntities(2)]
g=gmsh.model.geo
lb=g.addSurfaceLoop(bod)
L=2000.0; c=(100,111,133)
xs=[c[0]-L,c[0]+L]; ys=[c[1]-L,c[1]+L]; zs=[c[2]-L,c[2]+L]
pt={}
for i in range(2):
    for j in range(2):
        for k in range(2): pt[i,j,k]=g.addPoint(xs[i],ys[j],zs[k])
def ln(a,b): return g.addLine(pt[a],pt[b])
def face(a,b,cc,d):
    l=[ln(a,b),ln(b,cc),ln(cc,d),ln(d,a)]; return g.addPlaneSurface([g.addCurveLoop(l)])
far=[face((0,0,0),(1,0,0),(1,1,0),(0,1,0)),face((0,0,1),(1,0,1),(1,1,1),(0,1,1)),
     face((0,0,0),(1,0,0),(1,0,1),(0,0,1)),face((0,1,0),(1,1,0),(1,1,1),(0,1,1)),
     face((0,0,0),(0,1,0),(0,1,1),(0,0,1)),face((1,0,0),(1,1,0),(1,1,1),(1,0,1))]
g.removeAllDuplicates()
D=[(0,0,21.0,61.0,7.0),(200,0,21.0,61.0,7.0),(0,200,271.0,63.5,7.0),(200,200,271.0,63.5,7.0)]
dsk=[]
for (x,y,z,R,t) in D:
    p0=g.addPoint(x,y,z); q=[g.addPoint(x+R,y,z),g.addPoint(x,y+R,z),g.addPoint(x-R,y,z),g.addPoint(x,y-R,z)]
    a=[g.addCircleArc(q[i],p0,q[(i+1)%4]) for i in range(4)]
    s=g.addPlaneSurface([g.addCurveLoop(a)])
    ex=g.extrude([(2,s)],0,0,t)
    top=ex[0][1]; vol=ex[1][1]; sides=[e[1] for e in ex[2:]]
    dsk.append((s,top,sides,vol))
g.synchronize()
for s,top,sides,vol in dsk: g.remove([(3,vol)])
fl_far=g.addSurfaceLoop(far)
loops=[fl_far,lb]+[g.addSurfaceLoop([s,top]+sides) for s,top,sides,vol in dsk]
V=g.addVolume(loops)
g.synchronize()
gmsh.model.addPhysicalGroup(3,[V],name="fluid")
gmsh.model.addPhysicalGroup(2,far,name="farfield")
gmsh.model.addPhysicalGroup(2,bod,name="body")
for i,(s,top,sides,vol) in enumerate(dsk):
    gmsh.model.addPhysicalGroup(2,[top],name=f"dup{i+1}")
    gmsh.model.addPhysicalGroup(2,[s],name=f"ddn{i+1}")
    gmsh.model.addPhysicalGroup(2,sides,name=f"dsd{i+1}")
f=gmsh.model.mesh.field
f.add("Distance",1); f.setNumbers(1,"SurfacesList",bod); f.setNumber(1,"Sampling",20)
f.add("Threshold",2); f.setNumber(2,"InField",1); f.setNumber(2,"SizeMin",hb); f.setNumber(2,"SizeMax",hf); f.setNumber(2,"DistMin",2); f.setNumber(2,"DistMax",2000)
f.add("Distance",3); f.setNumbers(3,"SurfacesList",sum([[s,top]+sides for s,top,sides,vol in dsk],[])); f.setNumber(3,"Sampling",20)
f.add("Threshold",4); f.setNumber(4,"InField",3); f.setNumber(4,"SizeMin",hb*1.2); f.setNumber(4,"SizeMax",hf); f.setNumber(4,"DistMin",2); f.setNumber(4,"DistMax",2000)
f.add("Box",5); f.setNumber(5,"VIn",hb*3); f.setNumber(5,"VOut",hf)
for k,v in zip(["XMin","XMax","YMin","YMax","ZMin","ZMax","Thickness"],[-200,400,-200,430,-250,450,250]): f.setNumber(5,k,v)
f.add("Min",6); f.setNumbers(6,"FieldsList",[2,4,5]); f.setAsBackgroundMesh(6)
for k,v in [("MeshSizeExtendFromBoundary",0),("MeshSizeFromPoints",0),("MeshSizeFromCurvature",0),("Algorithm",6),("Algorithm3D",int(sys.argv[5]) if len(sys.argv)>5 else 1),("ScalingFactor",0.001)]:
    gmsh.option.setNumber("Mesh."+k,v)
gmsh.model.mesh.generate(3)
_,et,_=gmsh.model.mesh.getElements(3); _,ets,_=gmsh.model.mesh.getElements(2)
print("tets",sum(len(e) for e in et),"tris",sum(len(e) for e in ets),"time",round(time.time()-t0))
gmsh.write(out)
gmsh.finalize()
