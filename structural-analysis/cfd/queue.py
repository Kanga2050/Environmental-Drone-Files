import subprocess, sys, time, threading, os
q=[l.strip() for l in open(sys.argv[1]) if l.strip()]; th=sys.argv[2]; nw=int(sys.argv[3])
lk=threading.Lock()
def w():
    while True:
        with lk:
            if not q: return
            c=q.pop(0)
        s=time.time()
        r=subprocess.run([os.path.abspath("su2/bin/SU2_CFD.exe"),"-t",th,"case.cfg"],cwd=c,stdout=open(os.path.join(c,"log.txt"),"w"),stderr=subprocess.STDOUT)
        with lk: open("done.txt","a").write(f"{c} rc={r.returncode} secs={int(time.time()-s)}\n")
ts=[threading.Thread(target=w) for _ in range(nw)]
for t in ts: t.start(); time.sleep(3)
for t in ts: t.join()
