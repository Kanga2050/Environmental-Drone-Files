import json, math, sys, os, subprocess, numpy as np
rho=1.225; g=9.81
M=json.load(open("mass.json"))
it=[list(x) for x in M["items"]]
it[0][1]=426.7; it[0][2]=[99.999,107.388,123.001]
it[1][1]=83.1;  it[1][2]=[99.980,75.270,153.025]
m=sum(x[1] for x in it)/1000; cg=np.sum([np.array(x[2])*x[1] for x in it],0)/sum(x[1] for x in it)
W=m*g
D=[(0,0,0.061),(200,0,0.061),(0,200,0.0635),(200,200,0.0635)]
def split(T,dT=0.0):
    yf=cg[1]/200; Tf=T*yf+dT; Tr=T-Tf
    return [Tr/2,Tr/2,Tf/2,Tf/2]
def cfg(name,u,Ts,mesh="m3.su2",iters=2500,thr=None):
    u=np.array(u,float); V=np.linalg.norm(u)
    vin=[];mdot=[]
    for (x,y,R),T in zip(D,Ts):
        A=math.pi*R*R; Vax=-u[2]; Vt=math.hypot(u[0],u[1])
        lo,hi=0.0,60.0
        for _ in range(80):
            vi=(lo+hi)/2
            if 2*rho*A*vi*math.hypot(Vt,Vax+vi)>T: hi=vi
            else: lo=vi
        vi=(lo+hi)/2
        un=math.sqrt(max(Vax+vi,0)*max(Vax+2*vi,0)) if T>0 else Vax
        vec=np.array([u[0],u[1],-un]); mag=np.linalg.norm(vec)
        vin.append((mag,vec/mag)); mdot.append(rho*A*(Vax+vi))
    inl=", ".join(f"ddn{i+1}, 288.15, {mag:.5f}, {d[0]:.6f}, {d[1]:.6f}, {d[2]:.6f}" for i,(mag,d) in enumerate(vin))
    out=", ".join(f"dup{i+1}, {md:.6f}" for i,md in enumerate(mdot))
    s=f"""SOLVER= INC_RANS
KIND_TURB_MODEL= SST
MATH_PROBLEM= DIRECT
RESTART_SOL= NO
INC_NONDIM= DIMENSIONAL
INC_DENSITY_MODEL= CONSTANT
INC_DENSITY_INIT= {rho}
INC_VELOCITY_INIT= ({u[0]:.6f}, {u[1]:.6f}, {u[2]:.6f})
INC_TEMPERATURE_INIT= 288.15
FREESTREAM_TURBULENCEINTENSITY= 0.02
FREESTREAM_TURB2LAMVISCRATIO= 5.0
FLUID_MODEL= CONSTANT_DENSITY
VISCOSITY_MODEL= CONSTANT_VISCOSITY
MU_CONSTANT= 1.81e-5
REF_ORIGIN_MOMENT_X= {cg[0]/1000:.5f}
REF_ORIGIN_MOMENT_Y= {cg[1]/1000:.5f}
REF_ORIGIN_MOMENT_Z= {cg[2]/1000:.5f}
REF_LENGTH= 1.0
REF_AREA= 1.0
MARKER_HEATFLUX= ( body, 0.0 )
MARKER_WALL_FUNCTIONS= ( body, STANDARD_WALL_FUNCTION )
MARKER_EULER= ( dsd1, dsd2, dsd3, dsd4 )
MARKER_FAR= ( farfield )
INC_INLET_TYPE= VELOCITY_INLET, VELOCITY_INLET, VELOCITY_INLET, VELOCITY_INLET
MARKER_INLET= ( {inl} )
INC_OUTLET_TYPE= MASS_FLOW_OUTLET, MASS_FLOW_OUTLET, MASS_FLOW_OUTLET, MASS_FLOW_OUTLET
MARKER_OUTLET= ( {out} )
MARKER_PLOTTING= ( body )
MARKER_MONITORING= ( body )
NUM_METHOD_GRAD= GREEN_GAUSS
CFL_NUMBER= 10
CFL_ADAPT= YES
CFL_ADAPT_PARAM= ( 0.5, 1.2, 5.0, 100.0 )
LINEAR_SOLVER= FGMRES
LINEAR_SOLVER_PREC= ILU
LINEAR_SOLVER_ERROR= 0.05
LINEAR_SOLVER_ITER= 10
CONV_NUM_METHOD_FLOW= FDS
MUSCL_FLOW= YES
SLOPE_LIMITER_FLOW= VENKATAKRISHNAN
VENKAT_LIMITER_COEFF= 0.05
CONV_NUM_METHOD_TURB= SCALAR_UPWIND
MUSCL_TURB= NO
ITER= {iters}
CONV_FIELD= RMS_PRESSURE
CONV_RESIDUAL_MINVAL= -12
MESH_FILENAME= {mesh}
MESH_FORMAT= SU2
SCREEN_OUTPUT= (INNER_ITER, RMS_PRESSURE, RMS_TKE, CFX, CFY, CFZ, CMX, AVG_CFL)
HISTORY_OUTPUT= (ITER, RMS_RES, AERO_COEFF)
OUTPUT_FILES= (RESTART, PARAVIEW, SURFACE_PARAVIEW, SURFACE_CSV)
VOLUME_OUTPUT= (COORDINATES, SOLUTION, PRIMITIVE)
OUTPUT_WRT_FREQ= 500
"""
    os.makedirs(name,exist_ok=True)
    open(f"{name}/case.cfg","w").write(s)
    json.dump({"u":u.tolist(),"V":V,"T":list(Ts),"mdot":mdot,"jet":[v[0] for v in vin],"mass":m,"cg":cg.tolist(),"W":W},open(f"{name}/meta.json","w"),indent=1)
if __name__=="__main__":
    print("mass kg",round(m,4),"W N",round(W,3),"cg",cg.round(1))
