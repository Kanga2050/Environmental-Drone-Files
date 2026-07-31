import gmsh
gmsh.initialize(); gmsh.option.setNumber("General.Terminal",0)
occ=gmsh.model.occ
occ.importShapes("../cfd/manta.step"); occ.synchronize()
occ.remove([(3,t) for d,t in gmsh.model.getEntities(3) if t not in (194,195)],recursive=True); occ.synchronize()
for t in (194,195):
    print(t, occ.getMass(3,t), [round(x,2) for x in gmsh.model.getBoundingBox(3,t)])
gmsh.write("two.brep")
