"""
app.py
SensorGuard AI — Dashboard Streamlit
Futuria 8 | Détection d'anomalies IIoT | Dataset NASA SMAP/MSL
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from utils.data_loader import list_channels, load_channel, to_dataframe
from models.detector   import SensorGuardDetector

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG PAGE
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SensorGuard AI",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS custom ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

/* Base */
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #05080f; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0d1421 !important;
    border-right: 1px solid #1e2d45;
}
[data-testid="stSidebar"] * { color: #c8d8f0 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label { color: #4a6080 !important; font-size:12px !important; letter-spacing:2px; text-transform:uppercase; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #0d1421;
    border: 1px solid #1e2d45;
    padding: 16px 20px;
    border-radius: 0px;
}
[data-testid="stMetricValue"] { font-family: 'Space Mono', monospace !important; color: #00d4ff !important; font-size:22px !important; }
[data-testid="stMetricLabel"] { color: #4a6080 !important; font-size:10px !important; letter-spacing:3px; text-transform:uppercase; }
[data-testid="stMetricDelta"] { font-family: 'Space Mono', monospace !important; }

/* Headers */
h1, h2, h3 { font-family: 'Space Mono', monospace !important; color: #f0f6ff !important; letter-spacing: 1px; }

/* Tabs */
[data-baseweb="tab"] { font-family: 'Space Mono', monospace !important; font-size:11px !important; letter-spacing:2px; }
[aria-selected="true"] { color: #00d4ff !important; border-bottom: 2px solid #00d4ff !important; }

/* Buttons */
.stButton > button {
    background: transparent;
    border: 1px solid #1e2d45;
    color: #00d4ff;
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 2px;
    border-radius: 0;
    padding: 10px 20px;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: rgba(0,212,255,0.08);
    border-color: #00d4ff;
}

/* Dividers */
hr { border-color: #1e2d45 !important; }

/* Alert anomaly */
.anomaly-alert {
    background: rgba(255,107,53,0.1);
    border: 1px solid rgba(255,107,53,0.4);
    padding: 12px 18px;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    color: #ff6b35;
    letter-spacing: 1px;
    margin: 8px 0;
}

/* Status normal */
.status-ok {
    background: rgba(57,255,20,0.06);
    border: 1px solid rgba(57,255,20,0.3);
    padding: 12px 18px;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    color: #39ff14;
    letter-spacing: 1px;
}

/* Logo row */
.logo-row {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    color: #4a6080;
    letter-spacing: 4px;
    text-transform: uppercase;
    border-bottom: 1px solid #1e2d45;
    padding-bottom: 12px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Contrôles
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="logo-row">◈ SENSORGUARD AI · FUTURIA 08</div>', unsafe_allow_html=True)

    st.markdown("### DATASET")
    dataset_choice = st.selectbox("Source", ["SMAP", "MSL"], index=0)

    st.markdown("---")
    st.markdown("### CANAL CAPTEUR")

    try:
        channels = list_channels(dataset_choice)
        channel_id = st.selectbox("Channel ID", channels, index=0)
    except Exception as e:
        st.error(f"Erreur chargement channels : {e}")
        st.stop()

    st.markdown("---")
    st.markdown("### PARAMÈTRES DÉTECTION")

    threshold_pct = st.slider(
        "Seuil adaptatif (percentile)",
        min_value=80, max_value=99, value=95, step=1,
        help="Percentile du score de train utilisé comme seuil de détection."
    )

    window_size = st.slider(
        "Fenêtre d'analyse",
        min_value=8, max_value=128, value=32, step=8,
        help="Taille de la fenêtre glissante pour le scoring local."
    )

    viz_range = st.slider(
        "Plage affichée (points)",
        min_value=200, max_value=5000, value=1000, step=100,
        help="Nombre de points de test affichés sur le graphique."
    )

    st.markdown("---")
    run_btn = st.button("▶  LANCER L'ANALYSE", use_container_width=True)

    st.markdown("---")
    st.markdown("""
<div style="font-family:'Space Mono',monospace;font-size:9px;color:#4a6080;line-height:2;letter-spacing:1px;">
SOURCE · NASA SMAP/MSL<br>
PIPELINE · Reconstruction<br>
SCORING · Multi-échelle<br>
SEUIL · Adaptatif<br>
MODÈLE · SensorGuard v1
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HEADER PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════
col_title, col_status = st.columns([3, 1])
with col_title:
    st.markdown(f"""
<h1 style="font-size:32px;margin-bottom:0;">🛡 SENSORGUARD AI</h1>
<p style="font-family:'Space Mono',monospace;font-size:11px;color:#4a6080;letter-spacing:3px;margin-top:4px;">
DÉTECTION D'ANOMALIES · {dataset_choice} · CANAL {channel_id if 'channel_id' in dir() else '—'}
</p>
""", unsafe_allow_html=True)

with col_status:
    if "result" in st.session_state and st.session_state.result is not None:
        r = st.session_state.result
        if r.get("predictions") is not None:
            n_anom = int(r["predictions"].sum())
            if n_anom > 0:
                st.markdown(f'<div class="anomaly-alert">⚠ {n_anom} ANOMALIES DÉTECTÉES</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-ok">✓ SIGNAL NOMINAL</div>', unsafe_allow_html=True)

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# LOGIQUE PRINCIPALE : chargement + détection
# ══════════════════════════════════════════════════════════════════════════════

# Init session state
if "result" not in st.session_state:
    st.session_state.result    = None
if "data"   not in st.session_state:
    st.session_state.data      = None
if "metrics" not in st.session_state:
    st.session_state.metrics   = None

if run_btn:
    with st.spinner(f"Chargement du canal {channel_id}…"):
        try:
            data = load_channel(channel_id)
            st.session_state.data = data
        except Exception as e:
            st.error(f"❌ Erreur chargement : {e}")
            st.stop()

    with st.spinner("Entraînement du modèle sur données normales…"):
        detector = SensorGuardDetector(
            window=window_size,
            threshold_pct=threshold_pct
        )
        detector.fit(data["train"])

    with st.spinner("Calcul des scores d'anomalie sur le test set…"):
        result = detector.predict(data["test"])
        metrics = detector.evaluate(result["predictions"], data["labels"])

    st.session_state.result  = result
    st.session_state.metrics = metrics
    st.session_state.detector = detector
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# AFFICHAGE RÉSULTATS
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.result is not None:
    result  = st.session_state.result
    data    = st.session_state.data
    metrics = st.session_state.metrics

    # Bandeau source
    source = data.get("source", "synthetic") if data else "synthetic"
    if source == "synthetic":
        st.info("⚡ Mode DÉMO — Données synthétiques (anomalies injectées).")
    else:
        st.success(f"✅ Données réelles — Canal {data['chan_id']} · {data['test'].shape[0]:,} points · {int(data['labels'].sum())} anomalies NASA.")

    scores      = result["scores"]
    predictions = result["predictions"]
    labels      = data["labels"]
    test        = data["test"]
    threshold   = result["threshold"]
    T           = min(viz_range, len(scores))

    # ── MÉTRIQUES KPI ──────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("F1-Score",    f"{metrics['f1']:.3f}",
              delta="↑ cible >0.7" if metrics['f1'] >= 0.7 else "↓ < cible")
    c2.metric("Précision",   f"{metrics['precision']:.3f}")
    c3.metric("Recall",      f"{metrics['recall']:.3f}")
    c4.metric("FPR",         f"{metrics['fpr']:.3f}",
              delta="✓ <0.05" if metrics['fpr'] < 0.05 else "⚠ élevé")
    c5.metric("Seuil",       f"{threshold:.4f}")
    c6.metric("Anomalies",   f"{int(predictions.sum())}",
              delta=f"/ {len(predictions)} pts")

    st.markdown("---")

    # ── TABS ───────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 SIGNAL & ANOMALIES",
        "📊 SCORE D'ANOMALIE",
        "🔬 ANALYSE DÉTAILLÉE",
        "📋 RAPPORT",
    ])

    # ── TAB 1 : Signal ─────────────────────────────────────────────────────
    with tab1:
        t_arr = np.arange(T)
        feat  = test[:T, 0]  # première feature du canal multivarié

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.7, 0.3],
                            vertical_spacing=0.05)

        # Signal brut
        fig.add_trace(go.Scatter(
            x=t_arr, y=feat,
            mode="lines", name="Signal brut",
            line=dict(color="#00d4ff", width=1),
        ), row=1, col=1)

        # Zones anomalies réelles (ground truth)
        for seq in data["anomaly_sequences"]:
            s, e = seq[0], min(seq[1], T - 1)
            if s >= T:
                continue
            fig.add_vrect(
                x0=s, x1=e,
                fillcolor="rgba(255,107,53,0.15)",
                layer="below", line_width=0,
                annotation_text="GT", annotation_position="top left",
                annotation_font_size=9, annotation_font_color="#ff6b35",
                row=1, col=1
            )

        # Prédictions du modèle
        pred_t = np.where(predictions[:T] == 1)[0]
        if len(pred_t) > 0:
            fig.add_trace(go.Scatter(
                x=pred_t, y=feat[pred_t],
                mode="markers", name="Prédiction IA",
                marker=dict(color="#ff6b35", size=5, symbol="x"),
            ), row=1, col=1)

        # Score d'anomalie
        fig.add_trace(go.Scatter(
            x=t_arr, y=scores[:T],
            mode="lines", name="Score anomalie",
            line=dict(color="#c77dff", width=1.5),
            fill="tozeroy", fillcolor="rgba(199,125,255,0.08)",
        ), row=2, col=1)

        # Seuil
        fig.add_hline(
            y=threshold, row=2, col=1,
            line=dict(color="#ff6b35", width=1, dash="dash"),
            annotation_text=f"Seuil {threshold:.4f}",
            annotation_font_size=10, annotation_font_color="#ff6b35"
        )

        fig.update_layout(
            plot_bgcolor="#05080f",
            paper_bgcolor="#05080f",
            font=dict(family="Space Mono", color="#c8d8f0", size=11),
            legend=dict(bgcolor="#0d1421", bordercolor="#1e2d45", borderwidth=1),
            height=520,
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis2=dict(title="Timestep", gridcolor="#111827"),
            yaxis=dict(title="Amplitude", gridcolor="#111827"),
            yaxis2=dict(title="Score", gridcolor="#111827"),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
<div style="font-family:'Space Mono',monospace;font-size:10px;color:#4a6080;letter-spacing:1px;">
🟠 Zones oranges = anomalies réelles (Ground Truth NASA) &nbsp;|&nbsp;
✕ Marqueurs = prédictions SensorGuard AI &nbsp;|&nbsp;
Affichage : {T} premiers timesteps
</div>
""", unsafe_allow_html=True)

    # ── TAB 2 : Distribution score ──────────────────────────────────────────
    with tab2:
        col_a, col_b = st.columns(2)

        with col_a:
            # Distribution des scores
            normal_scores = scores[:T][labels[:T] == 0]
            anomaly_scores = scores[:T][labels[:T] == 1]

            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(
                x=normal_scores, name="Normal",
                marker_color="#00d4ff", opacity=0.7,
                nbinsx=50
            ))
            if len(anomaly_scores) > 0:
                fig2.add_trace(go.Histogram(
                    x=anomaly_scores, name="Anomalie (GT)",
                    marker_color="#ff6b35", opacity=0.7,
                    nbinsx=30
                ))
            fig2.add_vline(
                x=threshold,
                line=dict(color="#39ff14", width=2, dash="dash"),
                annotation_text="Seuil",
                annotation_font_color="#39ff14"
            )
            fig2.update_layout(
                title="Distribution des scores",
                plot_bgcolor="#05080f", paper_bgcolor="#0d1421",
                font=dict(family="Space Mono", color="#c8d8f0", size=10),
                barmode="overlay", height=350,
                legend=dict(bgcolor="#0d1421"),
                xaxis=dict(gridcolor="#111827"),
                yaxis=dict(gridcolor="#111827"),
            )
            st.plotly_chart(fig2, use_container_width=True)

        with col_b:
            # Matrice de confusion
            cm = np.array([[metrics["TN"], metrics["FP"]],
                           [metrics["FN"], metrics["TP"]]])
            fig3 = px.imshow(
                cm,
                labels=dict(x="Prédit", y="Réel"),
                x=["Normal", "Anomalie"],
                y=["Normal", "Anomalie"],
                color_continuous_scale=[[0, "#05080f"], [0.5, "#0066ff"], [1, "#00d4ff"]],
                text_auto=True,
                title="Matrice de confusion",
            )
            fig3.update_layout(
                plot_bgcolor="#05080f", paper_bgcolor="#0d1421",
                font=dict(family="Space Mono", color="#c8d8f0", size=10),
                height=350,
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig3, use_container_width=True)

    # ── TAB 3 : Multi-features ──────────────────────────────────────────────
    with tab3:
        n_feat = min(test.shape[1], 6)
        st.markdown(f"**{n_feat} features affichées** sur {test.shape[1]} dimensions du canal `{channel_id}`")

        fig4 = make_subplots(
            rows=n_feat, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            subplot_titles=[f"feat_{i}" for i in range(n_feat)]
        )
        palette = ["#00d4ff","#c77dff","#39ff14","#ff6b35","#ffd60a","#ff006e"]

        for i in range(n_feat):
            fig4.add_trace(go.Scatter(
                x=np.arange(T), y=test[:T, i],
                mode="lines", name=f"feat_{i}",
                line=dict(color=palette[i % len(palette)], width=1),
                showlegend=True,
            ), row=i+1, col=1)

            # Overlay anomalies GT
            for seq in data["anomaly_sequences"]:
                s, e = seq[0], min(seq[1], T - 1)
                if s >= T:
                    continue
                fig4.add_vrect(
                    x0=s, x1=e,
                    fillcolor="rgba(255,107,53,0.12)",
                    layer="below", line_width=0,
                    row=i+1, col=1
                )

        fig4.update_layout(
            plot_bgcolor="#05080f", paper_bgcolor="#05080f",
            font=dict(family="Space Mono", color="#c8d8f0", size=9),
            height=160 * n_feat,
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(bgcolor="#0d1421", bordercolor="#1e2d45", borderwidth=1),
        )
        for i in range(1, n_feat + 1):
            fig4.update_yaxes(gridcolor="#111827", row=i, col=1)
            fig4.update_xaxes(gridcolor="#111827", row=i, col=1)

        st.plotly_chart(fig4, use_container_width=True)

    # ── TAB 4 : Rapport ────────────────────────────────────────────────────
    with tab4:
        col_r1, col_r2 = st.columns(2)

        with col_r1:
            st.markdown("### RÉSUMÉ D'ANALYSE")
            anom_gt   = int(labels.sum())
            anom_pred = int(predictions.sum())
            coverage  = anom_gt / max(len(labels), 1) * 100

            st.markdown(f"""
<div style="font-family:'Space Mono',monospace;font-size:12px;line-height:2.2;color:#c8d8f0;">
<span style="color:#4a6080;">DATASET &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span> {dataset_choice}<br>
<span style="color:#4a6080;">CANAL &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span> {channel_id}<br>
<span style="color:#4a6080;">FEATURES &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span> {test.shape[1]}<br>
<span style="color:#4a6080;">POINTS TEST &nbsp;&nbsp;&nbsp;</span> {len(test):,}<br>
<span style="color:#4a6080;">ANOMALIES GT &nbsp;&nbsp;</span> {anom_gt:,} ({coverage:.1f}%)<br>
<span style="color:#4a6080;">ANOMALIES PRED </span> {anom_pred:,}<br>
<span style="color:#4a6080;">SEUIL &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span> {threshold:.6f}<br>
<span style="color:#4a6080;">PERCENTILE &nbsp;&nbsp;&nbsp;&nbsp;</span> {threshold_pct}e<br>
<span style="color:#4a6080;">FENÊTRE &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span> {window_size} pts
</div>
""", unsafe_allow_html=True)

        with col_r2:
            st.markdown("### MÉTRIQUES DÉTAILLÉES")
            st.markdown(f"""
<div style="font-family:'Space Mono',monospace;font-size:12px;line-height:2.2;color:#c8d8f0;">
<span style="color:#00d4ff;">F1-Score &nbsp;&nbsp;&nbsp;</span> {metrics['f1']:.4f}<br>
<span style="color:#00d4ff;">Précision &nbsp;&nbsp;</span> {metrics['precision']:.4f}<br>
<span style="color:#00d4ff;">Recall &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span> {metrics['recall']:.4f}<br>
<span style="color:#00d4ff;">FPR &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span> {metrics['fpr']:.4f}<br>
<span style="color:#39ff14;">TP &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span> {metrics['TP']:,}<br>
<span style="color:#ff6b35;">FP &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span> {metrics['FP']:,}<br>
<span style="color:#4a6080;">TN &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span> {metrics['TN']:,}<br>
<span style="color:#ff6b35;">FN &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span> {metrics['FN']:,}
</div>
""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### RECOMMANDATIONS AUTOMATIQUES")

        recs = []
        if metrics["recall"] < 0.5:
            recs.append("⬇ Diminuez le seuil (percentile) pour capturer plus d'anomalies vraies.")
        if metrics["fpr"] > 0.1:
            recs.append("⬆ Augmentez le seuil pour réduire les faux positifs.")
        if metrics["f1"] > 0.75:
            recs.append("✅ Performance excellente — ce canal est prêt pour la production.")
        if metrics["f1"] < 0.5:
            recs.append("🔬 Essayez une fenêtre d'analyse plus grande pour ce canal.")
        if not recs:
            recs.append("✓ Paramètres équilibrés — F1 dans la zone cible.")

        for r in recs:
            st.markdown(f"""<div class="{'anomaly-alert' if '⬇' in r or '⬆' in r else 'status-ok'}">{r}</div>""",
                       unsafe_allow_html=True)

else:
    # ── État initial (aucune analyse lancée) ─────────────────────────────────
    st.markdown("""
<div style="
    text-align:center;
    padding: 80px 40px;
    font-family:'Space Mono',monospace;
    color:#4a6080;
">
    <div style="font-size:48px;margin-bottom:24px;">🛡</div>
    <div style="font-size:14px;letter-spacing:4px;color:#00d4ff;margin-bottom:16px;">
        SENSORGUARD AI · PRÊT
    </div>
    <div style="font-size:12px;letter-spacing:2px;line-height:2;">
        Sélectionnez un dataset et un canal dans la sidebar<br>
        puis cliquez <strong style="color:#f0f6ff;">▶ LANCER L'ANALYSE</strong>
    </div>
    <div style="margin-top:40px;font-size:10px;letter-spacing:1px;color:#1e2d45;line-height:2.5;">
        SMAP — Soil Moisture Active Passive (NASA)<br>
        MSL &nbsp;— Mars Science Laboratory (NASA Curiosity)<br>
        Pipeline : PhysicalNorm → MultiScale → AdaptiveThreshold
    </div>
</div>
""", unsafe_allow_html=True)