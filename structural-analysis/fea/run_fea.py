import deck, json, sys, os, subprocess, numpy as np, time
from scipy.spatial import cKDTree
C=os.path.abspath("../cfd/ccx/Library")
env=dict(os.environ,PATH=C+r"\bin;"+C+r"\mingw-w64\bin;"+os.environ["PATH"],OMP_NUM_THREADS=sys.argv[3] if len(sys.argv)>3 else "8")
def pmap(case,k=1.0):
    a=np.loadtxt(f"../cfd/{case}/surface.csv",delimiter=",",skiprows=1,usecols=(1,2,3,4))
    t=cKDTree(a[:,:3]*1000.0); p=a[:,3]*k
    return lambda c: p[t.query(c)[1]]
def run(name):
    t=time.time()
    r=subprocess.run([C+r"\bin\ccx.exe","-i",name],cwd=name,env=env,stdout=open(f"{name}/log","w"),stderr=subprocess.STDOUT)
    print(name,"rc",r.returncode,int(time.time()-t),"s",flush=True)
if __name__=="__main__":
    name=sys.argv[1]; spec=json.loads(sys.argv[2])
    k=spec.get("scale",1.0)
    pf=pmap(spec["cfd"],k) if spec.get("cfd") else None
    if spec.get("cfd") and "T" not in spec:
        spec["T"]=[t*k for t in json.load(open(f"../cfd/{spec['cfd']}/meta.json"))["T"]]
    meta=deck.write(name,T=spec.get("T"),pfun=pf,fixed=spec.get("fixed"),accel=spec.get("accel"),extra=spec.get("extra"),modal=spec.get("modal",False))
    print(json.dumps({k:meta.get(k) for k in ("T","F_aero","a","g_eff","m_total_g")},default=float))
    run(name)
