import gmsh, sys
gmsh.initialize(); gmsh.option.setNumber("General.Terminal",0)
gmsh.option.setNumber("Geometry.OCCImportLabels",1)
gmsh.model.occ.importShapes("manta.step"); gmsh.model.occ.synchronize()
for d,t in gmsh.model.getEntities(3):
    n=gmsh.model.getEntityName(d,t)
    bb=gmsh.model.getBoundingBox(d,t)
    m=gmsh.model.occ.getMass(d,t); c=gmsh.model.occ.getCenterOfMass(d,t)
    print(t, n[-60:], "bb", [round(x,1) for x in bb], "vol", round(m,1), "c", [round(x,1) for x in c])
gmsh.finalize()
