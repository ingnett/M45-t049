"""
demo_app.py — M45: LiDAR Point Density & Annotation Difficulty Analyzer
=======================================================================
Standalone, fully self-contained demo app.
Deploy: streamlit run demo_app.py
Cloud:  push to GitHub → Streamlit Cloud → done.

No external data needed — generates realistic synthetic nuScenes-style
LiDAR point clouds on-the-fly using the inverse-square physics model.
"""

from __future__ import annotations

import math
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="M45 · LiDAR Annotation Difficulty Analyzer",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS — dark premium theme
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    border-right: 1px solid #334155;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

[data-testid="stAppViewContainer"] { background: #0b1120; }
[data-testid="stHeader"] { background: transparent; }

.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f2942 100%);
    border: 1px solid #1d4ed8;
    border-radius: 20px;
    padding: 40px 48px 32px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(56,189,248,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 8px;
    line-height: 1.2;
}
.hero-subtitle { font-size: 1.1rem; color: #94a3b8; margin: 0 0 20px; }
.hero-badges span {
    display: inline-block;
    background: rgba(56,189,248,0.12);
    border: 1px solid rgba(56,189,248,0.3);
    color: #38bdf8;
    border-radius: 30px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-right: 8px;
    letter-spacing: 0.03em;
}

.kpi-grid { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 28px; }
.kpi-card {
    flex: 1; min-width: 160px;
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 20px 24px;
    position: relative; overflow: hidden;
    transition: transform 0.2s, border-color 0.2s;
}
.kpi-card:hover { transform: translateY(-3px); border-color: #38bdf8; }
.kpi-card::after {
    content: '';
    position: absolute; bottom: 0; left: 0; right: 0;
    height: 3px; border-radius: 0 0 16px 16px;
}
.kpi-card.blue::after { background: linear-gradient(90deg, #2563eb, #38bdf8); }
.kpi-card.purple::after { background: linear-gradient(90deg, #7c3aed, #c084fc); }
.kpi-card.green::after { background: linear-gradient(90deg, #059669, #34d399); }
.kpi-card.orange::after { background: linear-gradient(90deg, #d97706, #fbbf24); }
.kpi-label { font-size: 0.75rem; color: #64748b; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 8px; }
.kpi-value { font-size: 2rem; font-weight: 800; color: #f1f5f9; }
.kpi-sub { font-size: 0.78rem; color: #94a3b8; margin-top: 4px; }

.section-header {
    font-size: 1.25rem; font-weight: 700; color: #e2e8f0;
    border-left: 4px solid #38bdf8; padding-left: 14px; margin: 28px 0 16px;
}
[data-testid="stTabs"] button { color: #94a3b8 !important; font-weight: 600; font-size: 0.9rem; }
[data-testid="stTabs"] button[aria-selected="true"] { color: #38bdf8 !important; border-bottom: 2px solid #38bdf8; }

.insight-box {
    background: rgba(56,189,248,0.07);
    border: 1px solid rgba(56,189,248,0.25);
    border-left: 4px solid #38bdf8;
    border-radius: 8px; padding: 14px 18px; margin: 12px 0;
    color: #cbd5e1; font-size: 0.9rem; line-height: 1.6;
}
.warn-box {
    background: rgba(239,68,68,0.07);
    border: 1px solid rgba(239,68,68,0.3);
    border-left: 4px solid #ef4444;
    border-radius: 8px; padding: 14px 18px; margin: 12px 0;
    color: #fca5a5; font-size: 0.9rem;
}
.success-box {
    background: rgba(52,211,153,0.07);
    border: 1px solid rgba(52,211,153,0.25);
    border-left: 4px solid #34d399;
    border-radius: 8px; padding: 14px 18px; margin: 12px 0;
    color: #6ee7b7; font-size: 0.9rem;
}
.badge-hard { background:#ef4444; color:#fff; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:700; }
.badge-ok   { background:#10b981; color:#fff; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# PHYSICS ENGINE
# ──────────────────────────────────────────────────────────────────────────────
RNG_SEED = 45
DISTANCE_BINS = [(0,10,"0–10 m"),(10,20,"10–20 m"),(20,30,"20–30 m"),(30,50,"30–50 m"),(50,120,"≥50 m")]
CLASS_PRIORS = {
    "vehicle.car":            {"wlh": [1.93, 4.62, 1.69], "C": 18000},
    "human.pedestrian.adult": {"wlh": [0.67, 0.73, 1.77], "C": 4200},
}

def theoretical_N(d, cls): return max(1, int(round(CLASS_PRIORS[cls]["C"] / max(2.0,d)**2)))

def yaw_rot(yaw):
    c,s = math.cos(yaw), math.sin(yaw)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float64)

def gen_cloud(center, wlh, yaw, n, rng):
    w,l,h = wlh; R = yaw_rot(yaw); hw,hl,hh = w/2,l/2,h/2
    pts = []
    for _ in range(max(3,n)):
        face = rng.choice(["side","front","roof"],p=[0.45,0.45,0.10])
        if face=="front":   lx=rng.uniform(-hl,hl); ly=hw if rng.rand()>0.5 else -hw; lz=rng.uniform(-hh,hh)
        elif face=="side":  lx=hl if rng.rand()>0.5 else -hl; ly=rng.uniform(-hw,hw); lz=rng.uniform(-hh,hh)
        else:               lx=rng.uniform(-hl,hl); ly=rng.uniform(-hw,hw); lz=hh
        pts.append(R@(np.array([lx,ly,lz])+rng.normal(0,.02,3))+center)
    gz = center[2]-hh-0.05
    for _ in range(min(120,n*2)):
        pts.append([center[0]+rng.uniform(-w*2,w*2), center[1]+rng.uniform(-l*2,l*2), gz+rng.normal(0,.01)])
    return np.array(pts, dtype=np.float64)

def in_box(pts, center, wlh, yaw):
    if len(pts)==0: return np.array([],dtype=bool)
    R=yaw_rot(yaw); loc=(pts-center)@R; hw,hl,hh=wlh[0]/2,wlh[1]/2,wlh[2]/2
    return (np.abs(loc[:,0])<=hl)&(np.abs(loc[:,1])<=hw)&(np.abs(loc[:,2])<=hh)

def compute_feats(center, wlh, yaw, pts, cls):
    w,l,h=wlh; d=float(np.linalg.norm(center)); vol=max(1e-4,w*l*h)
    mask=in_box(pts,center,wlh,yaw)
    iloc=(pts-center)@yaw_rot(yaw); ipts=iloc[mask]; N=len(ipts)
    density=N/vol; N_d2=N*d**2
    VX=0.2; nvx,nvy,nvz=max(1,int(math.ceil(w/VX))),max(1,int(math.ceil(l/VX))),max(1,int(math.ceil(h/VX)))
    if N>0:
        ix=np.clip(np.floor((ipts[:,0]+l/2)/VX),0,nvx-1)
        iy=np.clip(np.floor((ipts[:,1]+w/2)/VX),0,nvy-1)
        iz=np.clip(np.floor((ipts[:,2]+h/2)/VX),0,nvz-1)
        occ=len(np.unique(ix+iy*nvx+iz*nvx*nvy))/(nvx*nvy*nvz)
    else: occ=0.0
    rings=int(np.sum(np.diff(np.sort(ipts[:,2]))>0.08)+1) if N>1 else N
    if N>1:
        cov=np.mean([min(1,(ipts[:,i].max()-ipts[:,i].min())/max(1e-4,dim)) for i,dim in enumerate([l,w,h])])
        oi=(np.sign(ipts[:,0])>=0).astype(int)+(np.sign(ipts[:,1])>=0).astype(int)*2+(np.sign(ipts[:,2])>=0).astype(int)*4
        p=np.bincount(oi,minlength=8); p=p[p>0]/N
        entropy=float(-np.sum(p*np.log2(p)))
    else: cov=entropy=0.0
    patchiness=max(0.,1.-occ)
    buf_mask=in_box(pts,center,wlh+np.array([0.7,0.7,0.7]),yaw)
    clutter=float((buf_mask.sum()-mask.sum())/max(1,N))
    if N>=3:
        bev=ipts[:,:2]
        l_res=float(np.mean(np.minimum(np.minimum(np.abs(bev[:,0]-l/2),np.abs(bev[:,0]+l/2)),
                                       np.minimum(np.abs(bev[:,1]-w/2),np.abs(bev[:,1]+w/2)))))
    else: l_res=0.5
    pm=np.array([1.93,4.62,1.69] if "vehicle" in cls else [0.67,0.73,1.77])
    ps=np.array([0.15,0.35,0.20] if "vehicle" in cls else [0.12,0.14,0.15])
    sa=float(np.sum(np.abs(wlh-pm)/ps))
    dbin=next((lb for lo,hi,lb in DISTANCE_BINS if lo<=d<hi),"≥50 m")
    return dict(N=N,density=density,N_d2=N_d2,voxel_occupancy=occ,ring_count=rings,
                coverage_avg=cov,entropy_8_octant=entropy,patchiness=patchiness,
                boundary_clutter=clutter,l_shape_fit_residual=l_res,
                size_anomaly=sa,distance_m=d,distance_bin=dbin,volume_m3=vol)

def generate_dataset(n_cases: int = 80) -> pd.DataFrame:
    rng=np.random.RandomState(RNG_SEED); rows=[]
    classes=["vehicle.car","human.pedestrian.adult"]
    batches=["B1"]*20+["B2"]*20+["DEV"]*10+["TEST"]*20+["P"]*5
    dranges={"B1":(12,38),"B2":(25,65),"DEV":(14,48),"TEST":(15,60),"P":(10,20)}
    for i in range(n_cases):
        cls=classes[i%2]; batch=batches[i%len(batches)]; lo,hi=dranges[batch]
        d=float(rng.uniform(lo,hi)); ang=float(rng.uniform(-0.6,0.6))
        center=np.array([d*math.cos(ang),d*math.sin(ang),float(rng.uniform(-1.,0.))])
        wlh=np.clip(np.array(CLASS_PRIORS[cls]["wlh"])*(1+rng.normal(0,.05,3)),.3,8.)
        yaw=float(rng.uniform(-math.pi,math.pi))
        ntheory=theoretical_N(d,cls)
        pts=gen_cloud(center,wlh,yaw,ntheory,rng)
        f=compute_feats(center,wlh,yaw,pts,cls)
        sigma_c=max(.02,.8/max(1,f["N"])**.5); sigma_y=max(1.5,40./max(1,f["N"]))
        cerr=float(abs(rng.normal(0,sigma_c))); yerr=float(abs(rng.normal(0,sigma_y)))
        raw=2.*(1-min(1,f["N"]/25))+1.5*(1-f["coverage_avg"])+.8*f["patchiness"]+.5*min(1,f["distance_m"]/60)
        diff=float(np.clip(1+raw,1.,5.)); is_hard=diff>=4.
        rows.append({"case_id":f"CASE-{i+1:03d}","batch":batch,"class_name":cls,
                     "distance_m":round(f["distance_m"],1),"distance_bin":f["distance_bin"],
                     "N":f["N"],"density":round(f["density"],3),"N_d2":round(f["N_d2"],1),
                     "voxel_occupancy":round(f["voxel_occupancy"],3),"ring_count":f["ring_count"],
                     "coverage_avg":round(f["coverage_avg"],3),"entropy_8_octant":round(f["entropy_8_octant"],3),
                     "patchiness":round(f["patchiness"],3),"l_shape_fit_residual":round(f["l_shape_fit_residual"],3),
                     "boundary_clutter":round(f["boundary_clutter"],3),"size_anomaly":round(f["size_anomaly"],3),
                     "center_err_m":round(cerr,4),"yaw_err_deg":round(yerr,2),
                     "difficulty_gt":round(diff,2),"is_hard_gt":is_hard,
                     "_pts":pts,"_wlh":wlh,"_center":center,"_yaw":yaw})
    return pd.DataFrame(rows)

def train_ridge(df_in: pd.DataFrame, alpha: float = 1.0):
    FEATS = ["N", "density", "ring_count", "coverage_avg", "entropy_8_octant",
             "patchiness", "distance_m", "size_anomaly", "boundary_clutter", "l_shape_fit_residual"]
    FLIP = [True] * 5 + [False] * 5
    df = df_in[df_in["batch"].isin(["B1", "B2"])].copy()
    X = df[FEATS].to_numpy(np.float64)
    y = df["difficulty_gt"].to_numpy(np.float64)
    mu = X.mean(0)
    sig = X.std(0)
    sig[sig < 1e-6] = 1.0
    Xn = (X - mu) / sig
    for i, fl in enumerate(FLIP):
        if fl:
            Xn[:, i] = -Xn[:, i]
    n, p = Xn.shape
    Xd = np.hstack([Xn, np.ones((n, 1))])
    lam = alpha * np.eye(p + 1)
    lam[-1, -1] = 0
    sol = np.linalg.solve(Xd.T @ Xd + lam, Xd.T @ y)
    w, b = sol[:-1], float(sol[-1])

    def pred(df2: pd.DataFrame):
        Xp = (df2[FEATS].to_numpy(np.float64) - mu) / sig
        for i, fl in enumerate(FLIP):
            if fl:
                Xp[:, i] = -Xp[:, i]
        return np.clip(Xp @ w + b, 1.0, 5.0)

    return pred, dict(zip(FEATS, w.tolist())), b


def spearman(x,y):
    if np.std(x)<1e-9 or np.std(y)<1e-9: return 0.,1.
    return stats.spearmanr(x,y)

def box_lines(center,wlh,yaw):
    w,l,h=wlh; R=yaw_rot(yaw); hw,hl,hh=w/2,l/2,h/2
    cl=np.array([[-hl,-hw,-hh],[hl,-hw,-hh],[hl,hw,-hh],[-hl,hw,-hh],
                 [-hl,-hw,hh],[hl,-hw,hh],[hl,hw,hh],[-hl,hw,hh]])
    cw=cl@R.T+center
    edges=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    xl,yl,zl=[],[],[]
    for a,b in edges: xl+=[cw[a,0],cw[b,0],None]; yl+=[cw[a,1],cw[b,1],None]; zl+=[cw[a,2],cw[b,2],None]
    return xl,yl,zl

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Cấu hình Demo")
    n_cases     = st.slider("Số ca phân tích", 40, 120, 80, 10)
    ridge_alpha = st.slider("Ridge α (L₂ regularization)", 0.1, 10., 1., 0.1)
    hard_thresh = st.slider("Ngưỡng Ca Khó", 3., 5., 4., 0.5)
    st.divider()
    st.markdown("### 🔍 3D Viewer")
    case_idx = st.number_input("Xem ca số (0 = ca khó nhất)", 0, n_cases, 0)
    st.divider()
    st.caption("M45 · nuScenes-style LiDAR simulation\nPure NumPy Ridge Regression\nNo sklearn required — Deployable")

# ──────────────────────────────────────────────────────────────────────────────
# DATA & MODEL
# ──────────────────────────────────────────────────────────────────────────────
df = generate_dataset(n_cases)
pred_fn, weights, bias_val = train_ridge(df, ridge_alpha)
df["difficulty_pred"] = np.round(pred_fn(df), 2)
df["is_hard_pred"] = df["difficulty_pred"] >= hard_thresh

ah = df["difficulty_gt"].to_numpy() >= hard_thresh
ph = df["difficulty_pred"].to_numpy() >= hard_thresh
TP=int((ah&ph).sum()); FP=int((~ah&ph).sum()); FN=int((ah&~ph).sum()); TN=int((~ah&~ph).sum())
prec=TP/(TP+FP) if (TP+FP) else 1.; rec=TP/(TP+FN) if (TP+FN) else 1.
f1=2*prec*rec/(prec+rec) if (prec+rec) else 0.
rho_N,_ = spearman(df["N"], df["center_err_m"])
rho_dn,_= spearman(df["N"], df["difficulty_gt"])

# ──────────────────────────────────────────────────────────────────────────────
# HERO
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <div class="hero-title">📡 M45: LiDAR Annotation Difficulty Analyzer</div>
  <div class="hero-subtitle">Phân tích mật độ điểm LiDAR · Dự đoán độ khó gán nhãn Cuboid 3D · nuScenes-style · Ridge Regression (Pure NumPy)</div>
  <div class="hero-badges">
    <span>📐 7 Nhóm chỉ số A–G</span>
    <span>🤖 Ridge Regression</span>
    <span>🔴 Hard-case Detection</span>
    <span>📊 {n_cases} ca phân tích</span>
    <span>⚡ Self-contained · Deployable</span>
  </div>
</div>
""", unsafe_allow_html=True)

# KPI
st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi-card blue">
    <div class="kpi-label">Spearman ρ (N ↔ Sai số)</div>
    <div class="kpi-value">{rho_N:.4f}</div>
    <div class="kpi-sub">Tương quan nghịch mật độ ↔ lệch tâm</div>
  </div>
  <div class="kpi-card purple">
    <div class="kpi-label">Hard-case Precision</div>
    <div class="kpi-value">{prec*100:.1f}%</div>
    <div class="kpi-sub">TP={TP} / FP={FP}</div>
  </div>
  <div class="kpi-card green">
    <div class="kpi-label">Hard-case Recall</div>
    <div class="kpi-value">{rec*100:.1f}%</div>
    <div class="kpi-sub">TP={TP} / FN={FN}</div>
  </div>
  <div class="kpi-card orange">
    <div class="kpi-label">Hard-case F1</div>
    <div class="kpi-value">{f1:.4f}</div>
    <div class="kpi-sub">{int(ah.sum())} ca khó / {n_cases} tổng</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌍  Vấn đề & Bối cảnh",
    "📊  Mật độ vs Khoảng cách",
    "🤖  Mô hình Độ khó",
    "🔴  Phát hiện Ca Khó",
    "🧊  LiDAR 3D Viewer",
])

# ─── TAB 1 ───────────────────────────────────────────────────────────────────
with tab1:
    c1, c2 = st.columns([1,1], gap="large")
    with c1:
        st.markdown('<div class="section-header">📌 Bài toán thực tế</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="insight-box">
        Cảm biến LiDAR 32-beam (nuScenes) quét theo định luật <b>nghịch đảo bình phương khoảng cách</b>:<br><br>
        <code style="background:#0f172a;padding:4px 10px;border-radius:4px;color:#38bdf8">N ≈ C · A_proj / d²</code><br><br>
        Khi vật thể ở xa (&gt;30–50 m) hoặc bị che khuất, số điểm LiDAR suy giảm
        từ <b>hàng trăm</b> xuống còn <b>1–8 điểm</b>.
        </div>
        <div class="warn-box">⚠️ <b>Hệ quả gán nhãn Cuboid 3D:</b><br>
        • Thiếu điểm biên → sai số kích thước Δw, Δl, Δh<br>
        • Thiếu điểm mặt trước/sau → sai yaw ±180°<br>
        • Điểm lệch 1 phía → lệch tâm ΔCenter<br>
        • Thời gian xử lý kéo dài → giảm chất lượng nhãn</div>
        <div class="success-box">✅ <b>Giải pháp M45:</b> Đo 7 nhóm đặc trưng hình học A–G từ đám mây điểm,
        huấn luyện Ridge Regression dự đoán điểm độ khó 1–5, tự động sàng lọc <b>Ca khó</b> trước khi giao annotator.</div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="section-header">📉 Suy giảm N theo khoảng cách (Vật lý)</div>', unsafe_allow_html=True)
        ds=np.linspace(5,70,200)
        fig_d=go.Figure()
        fig_d.add_trace(go.Scatter(x=ds,y=[theoretical_N(d,"vehicle.car") for d in ds],mode="lines",name="vehicle.car",
                                   line=dict(color="#38bdf8",width=3),fill="tozeroy",fillcolor="rgba(56,189,248,0.08)"))
        fig_d.add_trace(go.Scatter(x=ds,y=[theoretical_N(d,"human.pedestrian.adult") for d in ds],mode="lines",
                                   name="pedestrian",line=dict(color="#c084fc",width=3),fill="tozeroy",fillcolor="rgba(192,132,252,0.08)"))
        fig_d.add_hline(y=5,line_dash="dot",line_color="#ef4444",
                        annotation_text="Waymo L1 ngưỡng ≥5 pts",annotation_position="bottom right",annotation_font_color="#ef4444")
        fig_d.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",
                             plot_bgcolor="rgba(15,23,42,0.8)",height=320,
                             xaxis_title="Khoảng cách d (m)",yaxis_title="Số điểm N",
                             legend=dict(bgcolor="rgba(0,0,0,0)"),margin=dict(l=20,r=20,t=10,b=20))
        st.plotly_chart(fig_d, use_container_width=True)

    st.markdown('<div class="section-header">🧭 7 Nhóm Chỉ số Ảnh hưởng Độ khó (A–G)</div>', unsafe_allow_html=True)
    cols=st.columns(7)
    groups=[("A","Mật độ & Số điểm","N, density,\nN·d², voxel,\nring count"),
            ("B","Phân bố điểm","Coverage 3 trục,\nEntropy 8 octant,\nPatchiness"),
            ("C","Khoảng cách\n& Kích thước","d, V=l·w·h,\nsize_anomaly"),
            ("D","Che khuất\n& Cắt cụt","Occlusion proxy,\nTruncation,\nBoundary clutter"),
            ("E","Chất lượng\ntín hiệu","Intensity μ/σ,\nL-shape fit\nresidual"),
            ("F","Thời gian\n& Ngữ cảnh","Track length,\nTemporal consist,\nVelocity"),
            ("G","Dataset labels","visibility_token,\nclass_name")]
    colors=["#2563eb","#7c3aed","#0891b2","#d97706","#059669","#db2777","#64748b"]
    for col,(g,name,desc),col_hex in zip(cols,groups,colors):
        with col:
            st.markdown(f"""<div style="background:rgba(15,23,42,0.9);border:1px solid {col_hex};border-radius:12px;
                padding:12px;text-align:center;height:160px;">
              <div style="font-size:1.5rem;font-weight:800;color:{col_hex}">({g})</div>
              <div style="font-size:0.75rem;font-weight:700;color:#e2e8f0;margin:4px 0">{name}</div>
              <div style="font-size:0.68rem;color:#64748b;white-space:pre-line;line-height:1.4">{desc}</div>
            </div>""", unsafe_allow_html=True)

# ─── TAB 2 ───────────────────────────────────────────────────────────────────
with tab2:
    c1,c2=st.columns(2,gap="large")
    with c1:
        fig_sc=px.scatter(df,x="distance_m",y="N",color="class_name",
                          size=np.clip(df["N"]+1,1,40).tolist(),symbol="is_hard_gt",
                          color_discrete_map={"vehicle.car":"#38bdf8","human.pedestrian.adult":"#c084fc"},
                          template="plotly_dark",height=360,title="Scatter: N vs d (▲=Ca khó)",
                          labels={"distance_m":"Khoảng cách d (m)","N":"Số điểm LiDAR N"})
        fig_sc.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(15,23,42,0.8)",
                             legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_sc, use_container_width=True)
    with c2:
        bin_order=[b[2] for b in DISTANCE_BINS]
        bs=df.groupby(["distance_bin","class_name"])["N"].agg(["mean","std"]).reset_index()
        bs["distance_bin"]=pd.Categorical(bs["distance_bin"],categories=bin_order,ordered=True)
        bs=bs.sort_values("distance_bin")
        fig_b=px.bar(bs,x="distance_bin",y="mean",color="class_name",barmode="group",error_y="std",
                     color_discrete_map={"vehicle.car":"#38bdf8","human.pedestrian.adult":"#c084fc"},
                     template="plotly_dark",height=360,title="N trung bình theo khoảng cự ly & Class",
                     labels={"distance_bin":"Khoảng cự ly","mean":"N trung bình"})
        fig_b.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(15,23,42,0.8)",
                             legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_b, use_container_width=True)

    c1,c2,c3=st.columns(3,gap="medium")
    for col,(xcol,ycol,lbl) in zip([c1,c2,c3],[("N","center_err_m","Lệch tâm (m)"),
                                                 ("N","yaw_err_deg","Sai số Yaw (°)"),
                                                 ("N","difficulty_gt","Độ khó (1–5)")]):
        rho_v,_=spearman(df[xcol],df[ycol])
        with col:
            fig=px.scatter(df,x=xcol,y=ycol,color="class_name",
                           color_discrete_map={"vehicle.car":"#38bdf8","human.pedestrian.adult":"#c084fc"},
                           template="plotly_dark",height=280,
                           title=f"N ↔ {lbl.split(' ')[0]}  ρ={rho_v:.3f}",
                           labels={xcol:"Số điểm N",ycol:lbl})
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(15,23,42,0.8)",
                               showlegend=False,margin=dict(t=40,b=20,l=20,r=20))
            st.plotly_chart(fig,use_container_width=True)


# ─── TAB 3 ───────────────────────────────────────────────────────────────────
with tab3:
    c1,c2=st.columns([1,1],gap="large")
    with c1:
        st.markdown('<div class="section-header">⚖️ Trọng số Đặc trưng</div>',unsafe_allow_html=True)
        wdf=pd.DataFrame({"Feature":list(weights.keys()),"Weight":list(weights.values())}).sort_values("Weight",ascending=True)
        fig_w=go.Figure(go.Bar(x=wdf["Weight"],y=wdf["Feature"],orientation="h",
                               marker_color=["#ef4444" if v<0 else "#38bdf8" for v in wdf["Weight"]],
                               text=[f"{v:.3f}" for v in wdf["Weight"]],textposition="outside"))
        fig_w.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",
                             plot_bgcolor="rgba(15,23,42,0.8)",height=320,
                             xaxis_title="Trọng số w_i",margin=dict(l=10,r=60,t=10,b=10))
        st.plotly_chart(fig_w,use_container_width=True)
        st.markdown(f"""<div class="insight-box" style="font-size:0.82rem">
        <b>Bias</b>: b={bias_val:.4f} · <b>α</b>={ridge_alpha}<br>
        Quy trình: Z-score chuẩn hóa + đảo dấu 'càng lớn càng dễ'
        → Ridge (X<sup>T</sup>X+αI)⁻¹X<sup>T</sup>y → clip[1,5]</div>""",unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="section-header">📈 Dự đoán vs Thực tế</div>',unsafe_allow_html=True)
        mae=float(np.mean(np.abs(df["difficulty_pred"]-df["difficulty_gt"])))
        rmse=float(np.sqrt(np.mean((df["difficulty_pred"]-df["difficulty_gt"])**2)))
        pr,_=spearman(df["difficulty_gt"],df["difficulty_pred"])
        fig_pv=go.Figure()
        fig_pv.add_trace(go.Scatter(x=df["difficulty_gt"],y=df["difficulty_pred"],mode="markers",
                                     marker=dict(color=df["difficulty_gt"],colorscale="RdYlGn_r",size=8,opacity=0.8,
                                                 colorbar=dict(title="Độ khó GT",len=.7,thickness=12)),name="Dự đoán"))
        fig_pv.add_trace(go.Scatter(x=[1,5],y=[1,5],mode="lines",
                                     line=dict(color="#64748b",dash="dot",width=2),name="Lý tưởng"))
        fig_pv.add_vline(x=hard_thresh,line_dash="dash",line_color="#ef4444",
                          annotation_text=f"Ngưỡng={hard_thresh}",annotation_font_color="#ef4444")
        fig_pv.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(15,23,42,0.8)",height=320,
                              xaxis_title="Độ khó GT",yaxis_title="Độ khó Pred",
                              legend=dict(bgcolor="rgba(0,0,0,0)"),margin=dict(l=20,r=20,t=10,b=20))
        st.plotly_chart(fig_pv,use_container_width=True)
        m1,m2,m3=st.columns(3)
        m1.metric("MAE",f"{mae:.3f}"); m2.metric("RMSE",f"{rmse:.3f}"); m3.metric("Spearman ρ",f"{pr:.3f}")

# ─── TAB 4 ───────────────────────────────────────────────────────────────────
with tab4:
    k1,k2,k3,k4=st.columns(4)
    k1.metric("Precision",f"{prec*100:.1f}%"); k2.metric("Recall",f"{rec*100:.1f}%")
    k3.metric("F1",f"{f1:.4f}"); k4.metric("Ca khó phát hiện",f"{TP}/{int(ah.sum())}")
    c1,c2=st.columns(2,gap="large")
    with c1:
        cm=go.Figure(go.Heatmap(z=[[TN,FP],[FN,TP]],
                                 x=["Pred: Bình thường","Pred: Ca khó"],
                                 y=["Thực: Bình thường","Thực: Ca khó"],
                                 colorscale=[[0,"#1e293b"],[.5,"#1d4ed8"],[1,"#ef4444"]],
                                 text=[[str(TN),str(FP)],[str(FN),str(TP)]],
                                 texttemplate="%{text}",textfont=dict(size=24,color="white"),showscale=False))
        cm.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(15,23,42,0.8)",height=270,title="Confusion Matrix",
                          margin=dict(l=20,r=20,t=40,b=20))
        st.plotly_chart(cm,use_container_width=True)
    with c2:
        fig_h=px.histogram(df,x="difficulty_pred",color="is_hard_gt",barmode="overlay",nbins=20,
                            color_discrete_map={True:"#ef4444",False:"#38bdf8"},
                            template="plotly_dark",height=270,title="Phân phối điểm độ khó",
                            labels={"difficulty_pred":"Điểm dự đoán","is_hard_gt":"Ca khó GT"})
        fig_h.add_vline(x=hard_thresh,line_dash="dash",line_color="#fbbf24",
                         annotation_text="Ngưỡng",annotation_font_color="#fbbf24")
        fig_h.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(15,23,42,0.8)",
                             legend=dict(bgcolor="rgba(0,0,0,0)"),margin=dict(l=20,r=20,t=40,b=20))
        st.plotly_chart(fig_h,use_container_width=True)

    hdf=df[df["is_hard_pred"]|df["is_hard_gt"]][
        ["case_id","batch","class_name","distance_m","N","density","coverage_avg",
         "difficulty_gt","difficulty_pred","is_hard_gt","is_hard_pred"]
    ].sort_values("difficulty_pred",ascending=False).head(20)
    hdf2=hdf.copy()
    hdf2["is_hard_gt"]=hdf2["is_hard_gt"].map({True:"🔴 Khó",False:"🟢 OK"})
    hdf2["is_hard_pred"]=hdf2["is_hard_pred"].map({True:"🔴 Khó",False:"🟢 OK"})
    st.markdown('<div class="section-header">📋 Hard-case List</div>',unsafe_allow_html=True)
    st.dataframe(hdf2,use_container_width=True,height=300)
    st.download_button("⬇️ Tải Hard-case List (CSV)",hdf.to_csv(index=False).encode(),"hard_case_list.csv","text/csv")

# ─── TAB 5 ───────────────────────────────────────────────────────────────────
with tab5:
    pick=int(df["difficulty_gt"].idxmax()) if case_idx==0 else min(int(case_idx)-1,len(df)-1)
    row=df.iloc[pick]
    pts_all=row["_pts"]; wlh=row["_wlh"]; center=row["_center"]; yaw=row["_yaw"]
    mask_in=in_box(pts_all,center,wlh,yaw); pts_in=pts_all[mask_in]; pts_out=pts_all[~mask_in]
    c1,c2=st.columns([2,1],gap="large")
    with c1:
        fig3d=go.Figure()
        if len(pts_out):
            fig3d.add_trace(go.Scatter3d(x=pts_out[:,0],y=pts_out[:,1],z=pts_out[:,2],
                                          mode="markers",marker=dict(size=1.5,color="#334155",opacity=.4),name="Background"))
        if len(pts_in):
            cv=np.sqrt(np.sum((pts_in-center)**2,axis=1))
            fig3d.add_trace(go.Scatter3d(x=pts_in[:,0],y=pts_in[:,1],z=pts_in[:,2],
                                          mode="markers",marker=dict(size=4,color=cv,colorscale="Plasma",opacity=.95,
                                          colorbar=dict(title="Dist",len=.5,thickness=10)),name=f"In-box ({len(pts_in)} pts)"))
        xl,yl,zl=box_lines(center,wlh,yaw)
        fig3d.add_trace(go.Scatter3d(x=xl,y=yl,z=zl,mode="lines",
                                      line=dict(color="#fbbf24",width=3),name="GT Box"))
        fig3d.add_trace(go.Scatter3d(x=[0],y=[0],z=[0],mode="markers+text",
                                      marker=dict(size=8,color="#ef4444",symbol="diamond"),
                                      text=["LIDAR"],textposition="top center",name="Sensor"))
        fig3d.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",
                              scene=dict(bgcolor="rgba(11,17,32,1)",
                                         xaxis=dict(gridcolor="#1e293b"),
                                         yaxis=dict(gridcolor="#1e293b"),
                                         zaxis=dict(gridcolor="#1e293b")),
                              legend=dict(bgcolor="rgba(0,0,0,0)"),height=480,margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig3d,use_container_width=True)
    with c2:
        dc="#ef4444" if row["difficulty_gt"]>=hard_thresh else "#34d399"
        st.markdown(f"""<div style="background:rgba(15,23,42,.9);border:1px solid #334155;border-radius:16px;padding:24px">
          <div style="font-size:.7rem;color:#64748b;font-weight:700;text-transform:uppercase">CA ĐANG XEM</div>
          <div style="font-size:1.3rem;font-weight:800;color:#e2e8f0;margin:4px 0">{row['case_id']}</div>
          <div style="border-top:1px solid #1e293b;margin:12px 0"></div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;font-size:.82rem">
            <div><span style="color:#64748b">Lớp</span><br><b style="color:#38bdf8">{row['class_name'].split('.')[-1]}</b></div>
            <div><span style="color:#64748b">Batch</span><br><b>{row['batch']}</b></div>
            <div><span style="color:#64748b">Khoảng cách</span><br><b>{row['distance_m']:.1f} m</b></div>
            <div><span style="color:#64748b">Bin</span><br><b>{row['distance_bin']}</b></div>
            <div><span style="color:#64748b">N (điểm)</span><br><b>{row['N']}</b></div>
            <div><span style="color:#64748b">Density</span><br><b>{row['density']:.3f}</b></div>
            <div><span style="color:#64748b">Coverage</span><br><b>{row['coverage_avg']:.3f}</b></div>
            <div><span style="color:#64748b">Rings</span><br><b>{row['ring_count']}</b></div>
            <div><span style="color:#64748b">Lệch tâm</span><br><b style="color:#fbbf24">{row['center_err_m']:.3f} m</b></div>
            <div><span style="color:#64748b">Sai Yaw</span><br><b style="color:#fbbf24">{row['yaw_err_deg']:.1f}°</b></div>
          </div>
          <div style="border-top:1px solid #1e293b;margin:12px 0;text-align:center">
            <div style="font-size:.7rem;color:#64748b;font-weight:600;text-transform:uppercase">Điểm Độ Khó</div>
            <div style="font-size:2.8rem;font-weight:800;color:{dc}">{row['difficulty_gt']:.2f}</div>
            <div style="font-size:.8rem;color:#94a3b8">Ground Truth</div>
            <div style="font-size:1.5rem;font-weight:700;color:#818cf8">{row['difficulty_pred']:.2f}</div>
            <div style="font-size:.75rem;color:#64748b">Ridge Prediction</div>
            <div style="margin-top:12px">
              {"<span class='badge-hard'>🔴 CA KHÓ</span>" if row['difficulty_gt']>=hard_thresh else "<span class='badge-ok'>🟢 Bình thường</span>"}
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

# FOOTER
st.markdown("""<div style="text-align:center;padding:32px 0 16px;color:#334155;font-size:.78rem">
  <b style="color:#475569">M45 · LiDAR Annotation Difficulty Analyzer</b> · nuScenes-style · Pure NumPy Ridge Regression<br>
  Nhóm M45 — Hưng (Thu nhãn/CVAT) · Dũng (Hình học) · Đường (Mô hình ML)<br>
  <code>streamlit run demo_app.py</code> &nbsp;|&nbsp; GitHub → Streamlit Cloud
</div>""", unsafe_allow_html=True)
