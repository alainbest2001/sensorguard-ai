"""
utils/data_loader.py
SensorGuard AI — Chargeur de données SMAP/MSL
Lit depuis data/train/*.csv (committés dans le repo).
Fallback : données synthétiques si fichiers absents.
Pour générer les CSV réels : exécuter fetch_data.py en local puis git push.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st

DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR  = os.path.join(DATA_DIR, "test")


# ── Synthetic fallback ────────────────────────────────────────────────────────

def _synthetic_channel(chan_id: str, seed: int = 42) -> dict:
    np.random.seed(seed)
    T_train, T_test, D = 1000, 600, 4
    t_tr = np.linspace(0, 10*np.pi, T_train)
    t_te = np.linspace(0,  6*np.pi, T_test)

    def make_signal(t, noise=0.08):
        return np.stack([
            np.sin(t) + noise*np.random.randn(len(t)),
            0.6*np.cos(t*0.7) + noise*np.random.randn(len(t)),
            np.sin(t*1.3+0.4) + noise*np.random.randn(len(t)),
            0.4*np.sin(t*2.1+1) + noise*np.random.randn(len(t)),
        ], axis=1)

    train = make_signal(t_tr)
    test  = make_signal(t_te)

    # Anomalies multi-capteurs
    anom_seqs = [[80,110],[280,310],[430,460],[530,550]]
    labels = np.zeros(T_test, dtype=int)
    for s, e in anom_seqs:
        if e < T_test:
            test[s:e, 0] += 2.8
            test[s:e, 2] += 1.5
            labels[s:e] = 1

    return {
        "train": train, "test": test, "labels": labels,
        "chan_id": chan_id, "n_features": D,
        "anomaly_sequences": anom_seqs, "source": "synthetic",
    }


# ── CSV loader ────────────────────────────────────────────────────────────────

def _csv_available() -> bool:
    return os.path.isdir(TRAIN_DIR) and len(os.listdir(TRAIN_DIR)) > 0


def _list_csv_channels() -> list:
    return sorted([f.replace(".csv","") for f in os.listdir(TRAIN_DIR) if f.endswith(".csv")])


def _load_csv_channel(chan_id: str) -> dict:
    df_tr = pd.read_csv(os.path.join(TRAIN_DIR, f"{chan_id}.csv"))
    df_te = pd.read_csv(os.path.join(TEST_DIR,  f"{chan_id}.csv"))
    labels    = df_te["label"].values.astype(int) if "label" in df_te.columns else np.zeros(len(df_te),dtype=int)
    feat_cols = [c for c in df_te.columns if c != "label"]
    train = df_tr[feat_cols].values
    test  = df_te[feat_cols].values
    # Reconstruct anomaly_sequences
    anom_seqs, in_anom, start = [], False, 0
    for i, v in enumerate(labels):
        if v == 1 and not in_anom: in_anom, start = True, i
        elif v == 0 and in_anom:   in_anom = False; anom_seqs.append([start, i-1])
    if in_anom: anom_seqs.append([start, len(labels)-1])
    return {
        "train": train, "test": test, "labels": labels,
        "chan_id": chan_id, "n_features": train.shape[1],
        "anomaly_sequences": anom_seqs, "source": "nasa_csv",
    }


# ── Public API ────────────────────────────────────────────────────────────────

SYNTHETIC_CHANNELS = {
    "SMAP": ["DEMO-P1","DEMO-P2","DEMO-P3","DEMO-P4","DEMO-P5"],
    "MSL" : ["DEMO-M1","DEMO-M2","DEMO-M3"],
}


@st.cache_data(show_spinner=False)
def list_channels(dataset: str = "SMAP") -> list:
    if _csv_available():
        prefix = "P" if dataset == "SMAP" else "M"
        real = [c for c in _list_csv_channels() if c.startswith(prefix)]
        if real:
            return real
    return SYNTHETIC_CHANNELS.get(dataset, ["DEMO-P1"])


@st.cache_data(show_spinner=False)
def load_channel(chan_id: str) -> dict:
    if _csv_available():
        if os.path.exists(os.path.join(TRAIN_DIR, f"{chan_id}.csv")):
            return _load_csv_channel(chan_id)
    seed = abs(hash(chan_id)) % 1000
    return _synthetic_channel(chan_id, seed=seed)


def to_dataframe(data: dict, split: str = "test") -> pd.DataFrame:
    arr  = data[split]
    cols = [f"feat_{i}" for i in range(arr.shape[1])]
    df   = pd.DataFrame(arr, columns=cols)
    df["t"] = np.arange(len(df))
    if split == "test":
        df["anomaly"] = data["labels"]
    return df