import gmsh, sys, time, numpy as np
h=float(sys.argv[1]); t0=time.time()
gmsh.initialize(); gmsh.option.setNumber("General.Terminal",1); gmsh.option.setNumber("General.Verbosity",2)
gmsh.option.setNumber("General.NumThreads",1)
gmsh.model.occ.importShapes("two.brep"); gmsh.model.occ.synchronize()
vs=gmsh.model.getEntities(3); print(vs,[gmsh.model.occ.getMass(*v) for v in vs])
for k,v in [("MeshSizeMax",h),("MeshSizeMin",h*0.3),("MeshSizeFromCurvature",0),("Algorithm",6),("Algorithm3D",1),("SecondOrderLinear",1),("HighOrderOptimize",0)]:
    gmsh.option.setNumber("Mesh."+k,v)
gmsh.model.mesh.generate(3); gmsh.model.mesh.setOrder(2)
for d,t in vs:
    gmsh.model.addPhysicalGroup(3,[t],name=f"V{t}")
et,tags,nodes=gmsh.model.mesh.getElements(3)
print("elements",[len(x) for x in tags],"time",round(time.time()-t0))
gmsh.option.setNumber("Mesh.SaveGroupsOfElements",1); gmsh.option.setNumber("Mesh.SaveGroupsOfNodes",1)
gmsh.write(f"manta_{h}.inp")
gmsh.finalize()
