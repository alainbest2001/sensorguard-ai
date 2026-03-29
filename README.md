# 🛡 SensorGuard AI — Futuria 8

**Détection d'anomalies dans des séries temporelles industrielles**  
Pipeline IA · Dataset NASA SMAP/MSL · Déploiement Streamlit Cloud

---

## 🚀 Déploiement rapide (Streamlit Cloud)

1. Fork ce repo sur GitHub
2. Connecte-toi sur [share.streamlit.io](https://share.streamlit.io)
3. New app → sélectionne ce repo → `app.py`
4. Deploy 🎉

---

## 🏗 Architecture

```
sensorguard/
├── app.py                    # Dashboard Streamlit principal
├── requirements.txt          # Dépendances Python
├── .streamlit/
│   └── config.toml          # Thème dark industriel
├── utils/
│   └── data_loader.py       # Chargement NASA SMAP/MSL (S3 public)
└── models/
    └── detector.py          # Pipeline détection anomalies
```

## 🧠 Pipeline de détection

```
Données SMAP/MSL (NASA S3)
        ↓
PhysicalNormalizer      ← Normalisation MinMax [-1,1]
        ↓
MultiScaleReconstructor ← Reconstruction multi-échelle [4,8,16,32]
        ↓
AnomalyScorer           ← Score = max(résidu, gradient) + lissage
        ↓
AdaptiveThreshold       ← Seuil = percentile sur scores de train
        ↓
Métriques + Visualisation Plotly
```

## 📊 Datasets supportés

| Dataset | Source | Channels | Type |
|---------|--------|----------|------|
| SMAP | NASA Telemanom | ~55 | Satellite telemetry |
| MSL | NASA Curiosity | ~27 | Mars rover |

## ⚙️ Paramètres configurables

- **Seuil adaptatif** : percentile 80–99 (défaut 95)
- **Fenêtre d'analyse** : 8–128 points
- **Plage affichée** : 200–5000 timesteps

---

**Futuria 8** | Portefeuille Futuria | 2026
