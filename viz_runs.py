import os
import re
import sys
import glob
import json
import time
import webbrowser

pat=re.compile(r"([a-z_0-9]+) ([-+]?\d+\.?\d*)")
OUT="runs_dashboard.html"
KEYS=["env","upd","ret10","cos","base","persist","skill","skill_p","erank","klraw","drift","gnorm","kl"]


def parse(log,max_pts=400):
    pts=[]
    with open(log,errors="ignore") as f:
        for l in f:
            if l.startswith("upd "):
                m={k:float(v) for k,v in pat.findall(l)}
                pts.append({k:m[k] for k in KEYS if k in m})
    if len(pts)>max_pts:
        step=len(pts)/max_pts
        pts=[pts[int(i*step)] for i in range(max_pts)]+[pts[-1]]
    return pts


def collect():
    seeds=[]
    for d in sorted(glob.glob("checkpoint_v3arch_seed*")):
        log=os.path.join(d,"train.log")
        if not os.path.exists(log):
            continue
        res=os.path.join(d,"results.json")
        done=None
        if os.path.exists(res):
            r=json.load(open(res))
            done={"last_fifth":r["last_fifth"],"best":r["best"],"score":r["crafter_score"]}
        seeds.append({"name":os.path.basename(d).replace("checkpoint_v3arch_",""),"pts":parse(log),"done":done})
    steps=int(os.environ.get("STEPS",50000))
    return {"seeds":seeds,"generated":time.strftime("%b %d %H:%M:%S"),"steps":steps,"updates":steps*128//(16*50)}


def write():
    data=collect()
    html=TEMPLATE.replace("__DATA__",json.dumps(data))
    with open(OUT,"w") as f:
        f.write(html)
    return data


TEMPLATE=r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="60">
<title>Dreamer-CDP runs</title>
<style>
:root{--surface:#fcfcfb;--page:#f4f4f2;--ink:#0b0b0b;--ink2:#52514e;--muted:#8a8984;--grid:#e6e5e1;--axis:#c9c8c3;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--warn:#fab219;--good:#0ca30c}
@media (prefers-color-scheme:dark){:root{--surface:#1a1a19;--page:#121211;--ink:#fff;--ink2:#c3c2b7;--muted:#8d8c85;--grid:#2c2c2a;--axis:#46453f;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500}}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:13px/1.4 -apple-system,BlinkMacSystemFont,"Helvetica Neue",sans-serif}
header{padding:18px 24px 6px;display:flex;flex-wrap:wrap;gap:18px;align-items:baseline}
h1{font-size:17px;margin:0;font-weight:600}
.sub{color:var(--ink2)}
.legend{display:flex;gap:14px;margin-left:auto}
.legend span{display:flex;align-items:center;gap:6px;color:var(--ink2)}
.legend i{width:14px;height:2px;border-radius:2px;display:inline-block}
.status{padding:4px 24px 10px;display:flex;flex-wrap:wrap;gap:10px}
.card{background:var(--surface);border-radius:8px;padding:10px 14px;min-width:220px}
.card b{font-weight:600}
.card .row{color:var(--ink2);font-variant-numeric:tabular-nums}
.bar{height:4px;background:var(--grid);border-radius:4px;margin:6px 0;overflow:hidden}
.bar div{height:100%;border-radius:4px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(380px,1fr));gap:12px;padding:6px 24px 16px}
.panel{background:var(--surface);border-radius:8px;padding:12px 12px 6px;position:relative}
.panel h2{font-size:13px;margin:0 0 2px;font-weight:600}
.panel p{margin:0 0 4px;color:var(--muted);font-size:12px}
svg{width:100%;height:auto;display:block;overflow:visible}
svg text{font-size:10.5px;fill:var(--muted);font-variant-numeric:tabular-nums}
.tip{position:fixed;pointer-events:none;background:var(--surface);color:var(--ink);border:1px solid var(--axis);border-radius:6px;padding:6px 9px;font-size:12px;display:none;z-index:9;box-shadow:0 2px 8px rgba(0,0,0,.12);font-variant-numeric:tabular-nums}
.tip .k{color:var(--ink2)}
.tip i{width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:5px}
details{padding:0 24px 30px;color:var(--ink2)}
table{border-collapse:collapse;margin-top:8px;background:var(--surface);border-radius:8px;font-variant-numeric:tabular-nums}
td,th{padding:5px 10px;text-align:right;border-bottom:1px solid var(--grid)}
th{color:var(--ink);font-weight:600}
.warn{color:var(--ink)}
.warn::before{content:"\26A0  ";color:var(--warn)}
.ok::before{content:"\2713  ";color:var(--good)}
</style></head><body>
<header><h1>Dreamer-CDP · V3-style backbone · 3 seeds</h1><span class="sub" id="gen"></span><div class="legend" id="legend"></div></header>
<div class="status" id="status"></div>
<div class="grid" id="grid"></div>
<details><summary>Table view (latest values)</summary><div id="table"></div></details>
<div class="tip" id="tip"></div>
<script>
const DATA=__DATA__;
const COLORS=["var(--s1)","var(--s2)","var(--s3)","var(--s4)"];
const METRICS=[
 {k:"ret10",t:"Episode return",d:"mean of last 10 episodes",f:2},
 {k:"skill",t:"Skill vs mean embedding",d:"cos − base; near 0 = predicting the average (collapse)",f:3,ref:0},
 {k:"skill_p",t:"Skill vs copy-last-frame",d:"cos − persist; above 0 = predictor beats copying",f:3,ref:0},
 {k:"cos",t:"Prediction cosine (centered)",d:"predicted vs target embedding",f:3},
 {k:"erank",t:"Effective rank of targets",d:"low = representation collapse",f:0},
 {k:"klraw",t:"KL (raw)",d:"below free bits = posterior ignoring obs",f:2,ref:1,refl:"free bits"},
 {k:"drift",t:"Encoder drift",d:"relative weight change from init",f:4},
 {k:"gnorm",t:"World-model grad norm",d:"clipped at 1000",f:0,ref:1000,refl:"clip"},
];
const S=DATA.seeds;
document.getElementById("gen").textContent="updated "+DATA.generated+" · auto-refreshes every 60 s";
document.getElementById("legend").innerHTML=S.map((s,i)=>`<span><i style="background:${COLORS[i]}"></i>${s.name}</span>`).join("");

function flags(p){const w=[];if(p.upd>=200&&p.klraw<1)w.push("klraw<1");if(p.erank<50)w.push("low erank");if(p.upd>=500&&p.skill<0.05)w.push("cos≈base");if(p.gnorm>1000)w.push("gnorm spike");return w}
document.getElementById("status").innerHTML=S.map((s,i)=>{
 const p=s.pts[s.pts.length-1]||{env:0,upd:0};
 const frac=Math.min(1,(p.env||0)/DATA.steps);
 if(s.done)return `<div class="card"><b>${s.name}</b> · done<div class="bar"><div style="width:100%;background:${COLORS[i]}"></div></div><div class="row">last-fifth ${s.done.last_fifth==null?"n/a":s.done.last_fifth.toFixed(2)} · best ${s.done.best.toFixed(2)} · score ${s.done.score.toFixed(2)}%</div></div>`;
 const w=s.pts.length?flags(p):[];
 return `<div class="card"><b>${s.name}</b> · ${(100*frac).toFixed(1)}%<div class="bar"><div style="width:${100*frac}%;background:${COLORS[i]}"></div></div><div class="row">upd ${p.upd} · env ${p.env}</div><div class="row ${w.length?"warn":"ok"}">${w.length?w.join(", "):"healthy"}</div></div>`;
}).join("")||'<div class="card">no runs yet</div>';

const W=460,H=190,M={l:44,r:40,t:8,b:24};
function ticks(lo,hi,n){const span=hi-lo||1,step0=span/n,mag=Math.pow(10,Math.floor(Math.log10(step0))),err=step0/mag,step=(err>=5?10:err>=2?5:err>=1.5?2:1)*mag;const out=[];for(let v=Math.ceil(lo/step)*step;v<=hi+1e-9;v+=step)out.push(+v.toFixed(10));return out}
function fmtX(v){return v>=1000?(v/1000)+"k":v}
function fmtY(v,f){return Math.abs(v)>=1000?(v/1000).toFixed(1)+"k":v.toFixed(Math.min(f,Math.abs(v)<1&&v!==0?3:f))}
const tip=document.getElementById("tip");
const grid=document.getElementById("grid");
const xmax=Math.max(DATA.updates*0.05,...S.flatMap(s=>s.pts.map(p=>p.upd)));
METRICS.forEach(m=>{
 const vals=S.flatMap(s=>s.pts.map(p=>p[m.k]).filter(v=>v!=null));
 if(m.ref!=null)vals.push(m.ref);
 if(!vals.length)return;
 let lo=Math.min(...vals),hi=Math.max(...vals);const pad=(hi-lo||Math.abs(hi)||1)*0.08;lo-=pad;hi+=pad;
 const X=v=>M.l+(v/xmax)*(W-M.l-M.r),Y=v=>M.t+(1-(v-lo)/(hi-lo))*(H-M.t-M.b);
 let g="";
 ticks(lo,hi,4).forEach(v=>{g+=`<line x1="${M.l}" x2="${W-M.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--grid)"/><text x="${M.l-6}" y="${Y(v)+3.5}" text-anchor="end">${fmtY(v,m.f)}</text>`});
 ticks(0,xmax,5).filter(v=>X(v)<W-M.r-20).forEach(v=>{g+=`<text x="${X(v)}" y="${H-6}" text-anchor="middle">${fmtX(v)}</text>`});
 g+=`<text x="${W-M.r}" y="${H-6}" text-anchor="start" dx="4">updates</text>`;
 g+=`<line x1="${M.l}" x2="${W-M.r}" y1="${H-M.b}" y2="${H-M.b}" stroke="var(--axis)"/>`;
 if(m.ref!=null)g+=`<line x1="${M.l}" x2="${W-M.r}" y1="${Y(m.ref)}" y2="${Y(m.ref)}" stroke="var(--ink2)" stroke-dasharray="4 3" stroke-width="1"/>`+(m.refl?`<text x="${W-M.r+4}" y="${Y(m.ref)+3.5}">${m.refl}</text>`:"");
 const ends=[];
 S.forEach((s,i)=>{const pts=s.pts.filter(p=>p[m.k]!=null);if(!pts.length)return;
  g+=`<path d="${pts.map((p,j)=>(j?"L":"M")+X(p.upd).toFixed(1)+","+Y(p[m.k]).toFixed(1)).join("")}" fill="none" stroke="${COLORS[i]}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`;
  const last=pts[pts.length-1];ends.push({y:Y(last[m.k]),x:X(last.upd),name:s.name,i})});
 ends.sort((a,b)=>a.y-b.y);for(let j=1;j<ends.length;j++)if(ends[j].y-ends[j-1].y<11)ends[j].y=ends[j-1].y+11;
 ends.forEach(e=>{g+=`<text x="${Math.min(e.x+5,W-M.r+4)}" y="${e.y+3.5}" style="fill:var(--ink2)">${e.name}</text>`});
 g+=`<line class="xh" y1="${M.t}" y2="${H-M.b}" stroke="var(--ink2)" stroke-width="1" visibility="hidden"/><g class="dots"></g><rect class="hit" x="${M.l}" y="0" width="${W-M.l-M.r}" height="${H}" fill="transparent"/>`;
 const el=document.createElement("div");el.className="panel";
 el.innerHTML=`<h2>${m.t}</h2><p>${m.d}</p><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${m.t} over updates">${g}</svg>`;
 grid.appendChild(el);
 const svg=el.querySelector("svg"),xh=svg.querySelector(".xh"),dots=svg.querySelector(".dots");
 svg.querySelector(".hit").addEventListener("mousemove",ev=>{
  const r=svg.getBoundingClientRect(),sx=(ev.clientX-r.left)*W/r.width,upd=(sx-M.l)/(W-M.l-M.r)*xmax;
  xh.setAttribute("x1",sx);xh.setAttribute("x2",sx);xh.setAttribute("visibility","visible");
  let rows="",d="";
  S.forEach((s,i)=>{const pts=s.pts.filter(p=>p[m.k]!=null);if(!pts.length)return;
   let best=pts[0];for(const p of pts)if(Math.abs(p.upd-upd)<Math.abs(best.upd-upd))best=p;
   if(Math.abs(best.upd-upd)>xmax*0.06)return;
   d+=`<circle cx="${X(best.upd)}" cy="${Y(best[m.k])}" r="4" fill="${COLORS[i]}" stroke="var(--surface)" stroke-width="2"/>`;
   rows+=`<div><i style="background:${COLORS[i]}"></i><span class="k">${s.name}</span> ${best[m.k].toFixed(m.f)} <span class="k">· env ${best.env.toLocaleString()}</span></div>`});
  dots.innerHTML=d;
  if(!rows){tip.style.display="none";return}
  tip.innerHTML=`<div class="k">${m.t} · upd ${Math.round(upd).toLocaleString()}</div>${rows}`;
  tip.style.display="block";tip.style.left=Math.min(ev.clientX+14,innerWidth-200)+"px";tip.style.top=(ev.clientY+14)+"px"});
 svg.querySelector(".hit").addEventListener("mouseleave",()=>{xh.setAttribute("visibility","hidden");dots.innerHTML="";tip.style.display="none"});
});
const cols=["env","upd",...METRICS.map(m=>m.k)];
document.getElementById("table").innerHTML=`<table><tr><th>seed</th>${cols.map(c=>`<th>${c}</th>`).join("")}</tr>${S.map(s=>{const p=s.pts[s.pts.length-1]||{};return `<tr><td>${s.name}</td>${cols.map(c=>`<td>${p[c]==null?"":p[c]}</td>`).join("")}</tr>`}).join("")}</table>`;
</script></body></html>
"""


if __name__=="__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    write()
    print(f"wrote {os.path.abspath(OUT)}")
    if "--no-open" not in sys.argv:
        webbrowser.open("file://"+os.path.abspath(OUT))
    if "-f" in sys.argv:
        while True:
            time.sleep(60)
            write()
