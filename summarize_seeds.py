import sys
import json
import os
import numpy as np

runs=[]
for d in sys.argv[1:]:
    p=os.path.join(d,"results.json")
    if os.path.exists(p):
        runs.append(json.load(open(p)))

if not runs:
    print("no results.json found")
    sys.exit(0)

print(f"{'seed':>4} {'eps':>5} {'first':>6} {'last':>6} {'best':>6} {'score':>7}")
for r in runs:
    f=lambda x:f"{x:6.2f}" if x is not None else "   n/a"
    print(f"{r['seed']:>4} {r['episodes']:>5} {f(r['first_fifth'])} {f(r['last_fifth'])} {f(r['best'])} {r['crafter_score']:>6.2f}%")

for k in ["first_fifth","last_fifth","best","crafter_score"]:
    v=np.array([r[k] for r in runs if r[k] is not None])
    if len(v)==0:
        continue
    print(f"{k:14s} {v.mean():.2f} ± {v.std(ddof=1) if len(v)>1 else 0:.2f}  (n={len(v)})")
