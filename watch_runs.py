import os
import re
import sys
import glob
import json
import time

STEPS=int(os.environ.get("STEPS",50000))
pat=re.compile(r"([a-z_0-9]+) ([-+]?\d+\.?\d*)")


def last_upd(log):
    line=None
    with open(log,errors="ignore") as f:
        for l in f:
            if l.startswith("upd "):
                line=l
    if line is None:
        return None
    return {k:float(v) for k,v in pat.findall(line)}


def flags(m):
    out=[]
    if m["upd"]>=200 and m["klraw"]<1.0:
        out.append("posterior ignoring obs (klraw<1)")
    if m["erank"]<50:
        out.append("low erank")
    if m["upd"]>=500 and m["skill"]<0.05:
        out.append("cos~base (predicting mean)")
    if m["gnorm"]>1000:
        out.append("gnorm spike")
    return out


def fmt_eta(sec):
    h,r=divmod(int(sec),3600)
    return f"{h}h{r//60:02d}m"


def report():
    runs=sorted(glob.glob("checkpoint_v3arch_seed*"))
    print(time.strftime("%b %d %H:%M:%S"))
    if not runs:
        print("no runs yet")
        return
    for d in runs:
        res=os.path.join(d,"results.json")
        log=os.path.join(d,"train.log")
        name=os.path.basename(d).replace("checkpoint_v3arch_","")
        if os.path.exists(res):
            r=json.load(open(res))
            last=f"{r['last_fifth']:.2f}" if r["last_fifth"] is not None else "n/a"
            print(f"{name:6s} DONE   last-fifth {last}  best {r['best']:.2f}  score {r['crafter_score']:.2f}%")
            continue
        if not os.path.exists(log):
            print(f"{name:6s} waiting")
            continue
        m=last_upd(log)
        if m is None:
            print(f"{name:6s} warming up")
            continue
        start=os.stat(log).st_birthtime
        frac=m["env"]/STEPS
        eta=(time.time()-start)*(1-frac)/max(frac,1e-6)
        stale=time.time()-os.path.getmtime(log)
        print(f"{name:6s} {100*frac:5.1f}%  upd {int(m['upd'])}  env {int(m['env'])}  eta {fmt_eta(eta)}")
        print(f"       ret10 {m['ret10']:.2f}  cos {m['cos']:.3f}  skill {m['skill']:+.3f}  skill_p {m['skill_p']:+.3f}")
        print(f"       erank {m['erank']:.0f}  klraw {m['klraw']:.1f}  drift {m['drift']:.4f}  gnorm {m['gnorm']:.0f}")
        warn=flags(m)
        if stale>600:
            warn.append(f"log silent {int(stale//60)} min")
        print("       "+("WARN: "+"; ".join(warn) if warn else "ok"))


if __name__=="__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if "-f" in sys.argv:
        while True:
            os.system("clear")
            report()
            time.sleep(30)
    else:
        report()
