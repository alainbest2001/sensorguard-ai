"""
utils/data_loader.py
SensorGuard AI — Chargeur de données SMAP / MSL (NASA)
Télécharge depuis S3 public, met en cache .npy, retourne DataFrame annoté.
"""

import os
import io
import zipfile
import requests
import numpy as np
import pandas as pd
import streamlit as st

# ── URLs publiques NASA / telemanom ──────────────────────────────────────────
SMAP_URL  = "https://s3-us-west-2.amazonaws.com/telemanom/data.zip"
LABELS_URL = (
    "https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv"
)
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "nasa_cache")


# ── Téléchargement & cache ───────────────────────────────────────────────────

def _ensure_cache():
    """Télécharge et décompresse une seule fois, stocke dans CACHE_DIR."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    train_dir = os.path.join(CACHE_DIR, "train")
    test_dir  = os.path.join(CACHE_DIR, "test")

    if os.path.isdir(train_dir) and os.path.isdir(test_dir):
        return  # déjà en cache

    with st.spinner("⬇️  Téléchargement NASA SMAP/MSL (une seule fois ~25 Mo)…"):
        r = requests.get(SMAP_URL, timeout=120)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            z.extractall(CACHE_DIR)

    # Le zip extrait dans data/ → on corrige le chemin si besoin
    extracted = os.path.join(CACHE_DIR, "data")
    if os.path.isdir(extracted):
        import shutil
        for sub in ["train", "test"]:
            src = os.path.join(extracted, sub)
            dst = os.path.join(CACHE_DIR, sub)
            if os.path.isdir(src) and not os.path.isdir(dst):
                shutil.move(src, dst)


def _load_labels() -> pd.DataFrame:
    labels_path = os.path.join(CACHE_DIR, "labeled_anomalies.csv")
    if not os.path.exists(labels_path):
        r = requests.get(LABELS_URL, timeout=30)
        r.raise_for_status()
        with open(labels_path, "wb") as f:
            f.write(r.content)
    return pd.read_csv(labels_path)


# ── API publique ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def list_channels(dataset: str = "SMAP") -> list[str]:
    """Retourne la liste des channels disponibles pour SMAP ou MSL."""
    _ensure_cache()
    labels = _load_labels()
    prefix = "P" if dataset == "SMAP" else "M"  # SMAP=P*, MSL=M*
    channels = labels[labels["chan_id"].str.startswith(prefix)]["chan_id"].tolist()
    return sorted(set(channels))


@st.cache_data(show_spinner=False)
def load_channel(chan_id: str) -> dict:
    """
    Charge train + test pour un channel donné.

    Retourne un dict avec :
      - train      : np.ndarray (T_train, D)
      - test       : np.ndarray (T_test,  D)
      - labels     : np.ndarray (T_test,) — 0/1 booléen anomalie
      - chan_id    : str
      - n_features : int
      - anomaly_sequences : liste de [start, end]
    """
    _ensure_cache()
    labels_df = _load_labels()

    train_path = os.path.join(CACHE_DIR, "train", f"{chan_id}.npy")
    test_path  = os.path.join(CACHE_DIR, "test",  f"{chan_id}.npy")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Channel {chan_id} introuvable dans le cache. "
            "Assurez-vous que le dataset est bien téléchargé."
        )

    train = np.load(train_path)   # (T, D) ou (T,) si univarié
    test  = np.load(test_path)

    # Forcer 2D
    if train.ndim == 1:
        train = train.reshape(-1, 1)
    if test.ndim == 1:
        test = test.reshape(-1, 1)

    # Labels d'anomalie
    row = labels_df[labels_df["chan_id"] == chan_id]
    anomaly_seqs = []
    label_arr    = np.zeros(len(test), dtype=int)

    if not row.empty:
        import ast
        seqs = row.iloc[0]["anomaly_sequences"]
        if isinstance(seqs, str):
            seqs = ast.literal_eval(seqs)
        for start, end in seqs:
            label_arr[start : end + 1] = 1
            anomaly_seqs.append([start, end])

    return {
        "train"             : train,
        "test"              : test,
        "labels"            : label_arr,
        "chan_id"           : chan_id,
        "n_features"        : train.shape[1],
        "anomaly_sequences" : anomaly_seqs,
    }


def to_dataframe(data: dict, split: str = "test") -> pd.DataFrame:
    """Convertit le dict channel en DataFrame Pandas avec colonne 'anomaly'."""
    arr = data[split]
    cols = [f"feat_{i}" for i in range(arr.shape[1])]
    df   = pd.DataFrame(arr, columns=cols)
    df["t"] = np.arange(len(df))

    if split == "test":
        df["anomaly"] = data["labels"]

    return df
