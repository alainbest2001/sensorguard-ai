"""
app.py — SensorGuard AI v3
LSTM Autoencoder · Dataset NASA SMAP/MSL · Streamlit Cloud
Version propre — une seule source de vérité.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from utils.data_loader import list_channels, load_channel
from models.detector   import SensorGuardDetector

# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SensorGuard AI",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #05080f; }
[data-testid="stSidebar"] { background: #0d1421 !important; border-right: 1px solid #1e2d45; }
[data-testid="stSidebar"] * { color: #c8d8f0 !important; }
[data-testid="metric-container"] { background: #0d1421; border: 1px solid #1e2d45; padding: 16px 20px; border-radius: 0px; }
[data-testid="stMetricValue"] { font-family: 'Space Mono', monospace !important; color: #00d4ff !important; font-size:22px !important; }
[data-testid="stMetricLabel"] { color: #4a6080 !important; font-size:10px !important; letter-spacing:3px; text-transform:uppercase; }
h1, h2, h3 { font-family: 'Space Mono', monospace !important; color: #f0f6ff !important; }
[data-baseweb="tab"] { font-family: 'Space Mono', monospace !important; font-size:11px !important; letter-spacing:2px; }
.stButton > button { background: transparent; border: 1px solid #1e2d45; color: #00d4ff; font-family: 'Space Mono', monospace; font-size: 11px; letter-spacing: 2px; border-radius: 0; padding: 10px 20px; }
.stButton > button:hover { background: rgba(0,212,255,0.08); border-color: #00d4ff; }
hr { border-color: #1e2d45 !important; }
.alert-box { background: rgba(255,107,53,0.1); border: 1px solid rgba(255,107,53,0.4); padding: 12px 18px; font-family: 'Space Mono', monospace; font-size: 12px; color: #ff6b35; letter-spacing: 1px; margin: 8px 0; }
.ok-box { background: rgba(57,255,20,0.06); border: 1px solid rgba(57,255,20,0.3); padding: 12px 18px; font-family: 'Space Mono', monospace; font-size: 12px; color: #39ff14; letter-spacing: 1px; }
.mono { font-family: 'Space Mono', monospace; font-size: 11px; color: #4a6080; letter-spacing: 3px; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<p class="mono">◈ SENSORGUARD AI · FUTURIA 08</p>', unsafe_allow_html=True)
    st.markdown("### DATASET")
    dataset_choice = st.selectbox("Source", ["SMAP", "MSL"], index=0)

    st.markdown("---")
    st.markdown("### CANAL CAPTEUR")
    try:
        channels   = list_channels(dataset_choice)
        channel_id = st.selectbox("Channel ID", channels, index=0)
    except Exception as e:
        st.error(f"Erreur : {e}")
        st.stop()

    st.markdown("---")
    st.markdown("### PARAMÈTRES")
    threshold_pct = st.slider("Seuil adaptatif (percentile)", 80, 99, 94, 1)
    window_size   = st.slider("Fenêtre LSTM", 16, 128, 64, 8)
    epochs        = st.slider("Epochs", 10, 50, 30, 5)
    viz_range     = st.slider("Plage affichée (points)", 200, 5000, 1000, 100)

    st.markdown("---")
    run_btn = st.button("▶  LANCER L'ANALYSE", use_container_width=True)

    st.markdown("""
<div class="mono" style="line-height:2.2;">
MODÈLE · LSTM AUTOENCODER<br>
PIPELINE · Seq2Seq Reconstruction<br>
SEUIL · Adaptatif Percentile<br>
VERSION · SensorGuard v3
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
col_title, col_status = st.columns([3, 1])
with col_title:
    st.markdown(f"""
<h1 style="font-size:32px;margin-bottom:0;">🛡 SENSORGUARD AI</h1>
<p class="mono" style="margin-top:4px;">DÉTECTION D'ANOMALIES · {dataset_choice} · CANAL {channel_id}</p>
""", unsafe_allow_html=True)

# Statut — affiché seulement si une analyse est terminée
with col_status:
    r_state = st.session_state.get("result", None)
    if r_state is not None and isinstance(r_state, dict) and r_state.get("predictions") is not None:
        n_anom = int(r_state["predictions"].sum())
        if n_anom > 0:
            st.markdown(f'<div class="alert-box">⚠ {n_anom} ANOMALIES</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="ok-box">✓ SIGNAL NOMINAL</div>', unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSE
# ══════════════════════════════════════════════════════════════════════════════
if run_btn:
    # Reset session
    st.session_state["result"]  = None
    st.session_state["data"]    = None
    st.session_state["metrics"] = None

    # 1. Chargement données
    with st.spinner(f"Chargement canal {channel_id}…"):
        try:
            data = load_channel(channel_id)
            st.session_state["data"] = data
        except Exception as e:
            st.error(f"❌ Erreur chargement : {e}")
            st.stop()

    # 2. Entraînement LSTM
    prog = st.progress(0, text="Initialisation LSTM Autoencoder…")
    with st.spinner("Entraînement LSTM Autoencoder…"):
        detector = SensorGuardDetector(
            window=window_size,
            threshold_pct=threshold_pct,
            hidden=32,
            epochs=epochs,
        )
        def _progress(ep, loss):
            pct = int(ep / epochs * 100)
            prog.progress(pct, text=f"Epoch {ep}/{epochs} — loss={loss:.6f}")
        detector.fit(data["train"], progress_cb=_progress)
        prog.empty()

    # 3. Prédiction
    with st.spinner("Calcul scores d'anomalie…"):
        result  = detector.predict(data["test"])
        metrics = detector.evaluate(result["predictions"], data["labels"])

    st.session_state["result"]   = result
    st.session_state["metrics"]  = metrics
    st.session_state["detector"] = detector
    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# AFFICHAGE RÉSULTATS
# ══════════════════════════════════════════════════════════════════════════════
result  = st.session_state.get("result",  None)
data    = st.session_state.get("data",    None)
metrics = st.session_state.get("metrics", None)

if result is None or data is None:
    st.markdown("""
<div style="text-align:center;padding:80px 40px;font-family:'Space Mono',monospace;color:#4a6080;">
    <div style="font-size:48px;margin-bottom:24px;">🛡</div>
    <div style="font-size:14px;letter-spacing:4px;color:#00d4ff;margin-bottom:16px;">SENSORGUARD AI · PRÊT</div>
    <div style="font-size:12px;letter-spacing:2px;line-height:2;">
        Sélectionnez un dataset et un canal<br>
        puis cliquez <strong style="color:#f0f6ff;">▶ LANCER L'ANALYSE</strong>
    </div>
    <div style="margin-top:40px;font-size:10px;letter-spacing:1px;color:#1e2d45;line-height:2.5;">
        SMAP — Soil Moisture Active Passive (NASA)<br>
        MSL  — Mars Science Laboratory (NASA Curiosity)<br>
        Modèle : LSTM Autoencoder Seq2Seq
    </div>
</div>""", unsafe_allow_html=True)
    st.stop()

# Variables
scores      = result["scores"]
predictions = result["predictions"]
labels      = data["labels"]
test        = data["test"]
threshold   = result["threshold"]
losses      = result.get("losses", [])
T           = min(viz_range, len(scores))

# Bandeau source
source = data.get("source", "synthetic")
if source == "synthetic":
    st.info("⚡ Mode DÉMO — Données synthétiques avec anomalies injectées.")
else:
    st.success(f"✅ Données réelles — Canal {data['chan_id']} · {len(test):,} points · {int(labels.sum())} anomalies NASA.")

# ── KPI ──────────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("F1-Score",  f"{metrics['f1']:.3f}",
          delta="↑ ≥0.6" if metrics['f1'] >= 0.6 else "↓ <0.6")
c2.metric("Précision", f"{metrics['precision']:.3f}")
c3.metric("Recall",    f"{metrics['recall']:.3f}")
c4.metric("FPR",       f"{metrics['fpr']:.3f}",
          delta="✓ <0.05" if metrics['fpr'] < 0.05 else "⚠ élevé")
c5.metric("Seuil",     f"{threshold:.4f}")
c6.metric("Anomalies", f"{int(predictions.sum())}",
          delta=f"/ {len(predictions)} pts")

st.markdown("---")

# ── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 SIGNAL & ANOMALIES",
    "📊 SCORE D'ANOMALIE",
    "🔬 MULTI-FEATURES",
    "📋 RAPPORT",
])

# ── TAB 1 ─────────────────────────────────────────────────────────────────────
with tab1:
    t_arr = np.arange(T)
    feat  = test[:T, 0]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], vertical_spacing=0.05)

    fig.add_trace(go.Scatter(x=t_arr, y=feat, mode="lines",
                             name="Signal", line=dict(color="#00d4ff", width=1)), row=1, col=1)

    for seq in data["anomaly_sequences"]:
        s, e = seq[0], min(seq[1], T-1)
        if s >= T: continue
        fig.add_vrect(x0=s, x1=e, fillcolor="rgba(255,107,53,0.15)",
                      layer="below", line_width=0, row=1, col=1)

    pred_t = np.where(predictions[:T] == 1)[0]
    if len(pred_t) > 0:
        fig.add_trace(go.Scatter(x=pred_t, y=feat[pred_t], mode="markers",
                                 name="Prédiction IA",
                                 marker=dict(color="#ff6b35", size=5, symbol="x")), row=1, col=1)

    fig.add_trace(go.Scatter(x=t_arr, y=scores[:T], mode="lines",
                             name="Score", line=dict(color="#c77dff", width=1.5),
                             fill="tozeroy", fillcolor="rgba(199,125,255,0.08)"), row=2, col=1)

    fig.add_hline(y=threshold, row=2, col=1,
                  line=dict(color="#ff6b35", width=1, dash="dash"),
                  annotation_text=f"Seuil {threshold:.4f}",
                  annotation_font_color="#ff6b35")

    fig.update_layout(plot_bgcolor="#05080f", paper_bgcolor="#05080f",
                      font=dict(family="Space Mono", color="#c8d8f0", size=11),
                      legend=dict(bgcolor="#0d1421", bordercolor="#1e2d45", borderwidth=1),
                      height=520, margin=dict(l=0, r=0, t=20, b=0),
                      xaxis2=dict(title="Timestep", gridcolor="#111827"),
                      yaxis=dict(title="Amplitude", gridcolor="#111827"),
                      yaxis2=dict(title="Score", gridcolor="#111827"))

    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<p class="mono">🟠 Zones oranges = Ground Truth NASA &nbsp;|&nbsp; ✕ = Prédictions LSTM &nbsp;|&nbsp; Violet = Score d\'anomalie</p>', unsafe_allow_html=True)

# ── TAB 2 ─────────────────────────────────────────────────────────────────────
with tab2:
    col_a, col_b = st.columns(2)
    with col_a:
        normal_s  = scores[:T][labels[:T] == 0]
        anomaly_s = scores[:T][labels[:T] == 1]
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=normal_s,  name="Normal",
                                    marker_color="#00d4ff", opacity=0.7, nbinsx=50))
        if len(anomaly_s) > 0:
            fig2.add_trace(go.Histogram(x=anomaly_s, name="Anomalie (GT)",
                                        marker_color="#ff6b35", opacity=0.7, nbinsx=30))
        fig2.add_vline(x=threshold, line=dict(color="#39ff14", width=2, dash="dash"),
                       annotation_text="Seuil", annotation_font_color="#39ff14")
        fig2.update_layout(title="Distribution des scores", barmode="overlay",
                           plot_bgcolor="#05080f", paper_bgcolor="#0d1421",
                           font=dict(family="Space Mono", color="#c8d8f0", size=10),
                           height=350, xaxis=dict(gridcolor="#111827"),
                           yaxis=dict(gridcolor="#111827"),
                           legend=dict(bgcolor="#0d1421"))
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        cm = np.array([[metrics["TN"], metrics["FP"]], [metrics["FN"], metrics["TP"]]])
        fig3 = px.imshow(cm, labels=dict(x="Prédit", y="Réel"),
                         x=["Normal","Anomalie"], y=["Normal","Anomalie"],
                         color_continuous_scale=[[0,"#05080f"],[0.5,"#0066ff"],[1,"#00d4ff"]],
                         text_auto=True, title="Matrice de confusion")
        fig3.update_layout(plot_bgcolor="#05080f", paper_bgcolor="#0d1421",
                           font=dict(family="Space Mono", color="#c8d8f0", size=10),
                           height=350, coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

# ── TAB 3 ─────────────────────────────────────────────────────────────────────
with tab3:
    n_feat = min(test.shape[1], 4)
    fig4 = make_subplots(rows=n_feat, cols=1, shared_xaxes=True,
                         vertical_spacing=0.04,
                         subplot_titles=[f"feat_{i}" for i in range(n_feat)])
    pal = ["#00d4ff","#c77dff","#39ff14","#ff6b35"]
    for i in range(n_feat):
        fig4.add_trace(go.Scatter(x=np.arange(T), y=test[:T, i], mode="lines",
                                  line=dict(color=pal[i], width=1), name=f"feat_{i}"),
                       row=i+1, col=1)
        for seq in data["anomaly_sequences"]:
            s, e = seq[0], min(seq[1], T-1)
            if s < T:
                fig4.add_vrect(x0=s, x1=e, fillcolor="rgba(255,107,53,0.12)",
                               layer="below", line_width=0, row=i+1, col=1)
    fig4.update_layout(plot_bgcolor="#05080f", paper_bgcolor="#05080f",
                       font=dict(family="Space Mono", color="#c8d8f0", size=9),
                       height=160*n_feat, margin=dict(l=0,r=0,t=30,b=0),
                       legend=dict(bgcolor="#0d1421"))
    for i in range(1, n_feat+1):
        fig4.update_yaxes(gridcolor="#111827", row=i, col=1)
        fig4.update_xaxes(gridcolor="#111827", row=i, col=1)
    st.plotly_chart(fig4, use_container_width=True)

# ── TAB 4 ─────────────────────────────────────────────────────────────────────
with tab4:
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.markdown("### RÉSUMÉ D'ANALYSE")
        st.markdown(f"""
<div style="font-family:'Space Mono',monospace;font-size:12px;line-height:2.2;color:#c8d8f0;">
<span style="color:#4a6080;">DATASET &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span> {dataset_choice}<br>
<span style="color:#4a6080;">CANAL &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span> {channel_id}<br>
<span style="color:#4a6080;">FEATURES &nbsp;&nbsp;&nbsp;&nbsp;</span> {test.shape[1]}<br>
<span style="color:#4a6080;">POINTS TEST &nbsp;</span> {len(test):,}<br>
<span style="color:#4a6080;">ANOMALIES GT &nbsp;</span> {int(labels.sum()):,} ({100*labels.sum()/len(labels):.1f}%)<br>
<span style="color:#4a6080;">ANOMALIES PRED</span> {int(predictions.sum()):,}<br>
<span style="color:#4a6080;">SEUIL &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span> {threshold:.6f}<br>
<span style="color:#4a6080;">PERCENTILE &nbsp;&nbsp;</span> {threshold_pct}e<br>
<span style="color:#4a6080;">FENÊTRE LSTM &nbsp;</span> {window_size} pts<br>
<span style="color:#4a6080;">EPOCHS &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span> {epochs}
</div>""", unsafe_allow_html=True)

    with col_r2:
        st.markdown("### MÉTRIQUES")
        st.markdown(f"""
<div style="font-family:'Space Mono',monospace;font-size:12px;line-height:2.2;color:#c8d8f0;">
<span style="color:#00d4ff;">F1-Score &nbsp;&nbsp;</span> {metrics['f1']:.4f}<br>
<span style="color:#00d4ff;">Précision &nbsp;</span> {metrics['precision']:.4f}<br>
<span style="color:#00d4ff;">Recall &nbsp;&nbsp;&nbsp;&nbsp;</span> {metrics['recall']:.4f}<br>
<span style="color:#00d4ff;">FPR &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span> {metrics['fpr']:.4f}<br>
<span style="color:#39ff14;">TP &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span> {metrics['TP']:,}<br>
<span style="color:#ff6b35;">FP &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span> {metrics['FP']:,}<br>
<span style="color:#4a6080;">TN &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span> {metrics['TN']:,}<br>
<span style="color:#ff6b35;">FN &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span> {metrics['FN']:,}
</div>""", unsafe_allow_html=True)

    # Courbe de convergence LSTM
    if losses:
        st.markdown("### CONVERGENCE LSTM")
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(y=losses, mode="lines+markers",
                                      line=dict(color="#00d4ff", width=2),
                                      marker=dict(size=4), name="MSE Loss"))
        fig_loss.update_layout(plot_bgcolor="#05080f", paper_bgcolor="#0d1421",
                               font=dict(family="Space Mono", color="#c8d8f0", size=10),
                               xaxis=dict(title="Epoch", gridcolor="#111827"),
                               yaxis=dict(title="MSE Loss", gridcolor="#111827"),
                               height=220, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig_loss, use_container_width=True)

    # Recommandations
    st.markdown("### RECOMMANDATIONS")
    recs = []
    if metrics["recall"] < 0.5:
        recs.append(("alert", "⬇ Diminuez le seuil (percentile) pour capturer plus d'anomalies."))
    if metrics["fpr"] > 0.1:
        recs.append(("alert", "⬆ Augmentez le seuil pour réduire les faux positifs."))
    if metrics["f1"] >= 0.6:
        recs.append(("ok", "✅ Performance correcte — F1 ≥ 0.6."))
    if metrics["f1"] < 0.4:
        recs.append(("alert", "🔬 Augmentez le nombre d'epochs ou la fenêtre LSTM."))
    if not recs:
        recs.append(("ok", "✓ Paramètres équilibrés."))
    for style, msg in recs:
        css = "alert-box" if style == "alert" else "ok-box"
        st.markdown(f'<div class="{css}">{msg}</div>', unsafe_allow_html=True)