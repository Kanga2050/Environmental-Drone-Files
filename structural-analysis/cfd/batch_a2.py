import run, numpy as np
W=run.W; sp=run.split
cases=[("hover",[0,0,-0.05],sp(W)),("climb5",[0,0,-5.0],sp(1.1*W)),("vto_max",[0,0,-8.0],sp(4*13.73)),("gust_side10",[-10.0,0,-0.05],sp(W)),("gust_rear10",[0,10.0,-0.05],sp(W))]
for n,u,T in cases:
    run.cfg(n,u,T,mesh="../m45.su2",iters=1500)
    m=__import__("json").load(open(f"{n}/meta.json")); print(n,np.round(m["jet"],2),np.round(m["mdot"],4))
