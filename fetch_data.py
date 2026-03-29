"""
fetch_data.py — À exécuter UNE SEULE FOIS sur ton PC local (pas Streamlit Cloud)
Télécharge SMAP/MSL NASA et génère les fichiers CSV dans data/
Puis committe data/ sur GitHub → Streamlit Cloud n'aura plus besoin de télécharger.

Usage :
    python fetch_data.py
"""
import os, io, zipfile, requests, numpy as np, pandas as pd, ast

SMAP_URL   = "https://s3-us-west-2.amazonaws.com/telemanom/data.zip"
LABELS_URL = "https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv"
DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")

# Channels clés à embarquer (légers, représentatifs)
SMAP_CHANNELS = ["P-1","P-2","P-3","P-4","P-7","P-10","P-15"]
MSL_CHANNELS  = ["M-1","M-2","M-3","M-4","M-6","M-7"]

def main():
    os.makedirs(os.path.join(DATA_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "test"),  exist_ok=True)

    # ── 1. Téléchargement ───────────────────────────────────────────────────
    print("⬇  Téléchargement NASA S3...")
    r = requests.get(SMAP_URL, timeout=120)
    r.raise_for_status()

    cache_dir = os.path.join(DATA_DIR, "_raw")
    os.makedirs(cache_dir, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall(cache_dir)

    raw_train = os.path.join(cache_dir, "data", "train")
    raw_test  = os.path.join(cache_dir, "data", "test")
    print("✅ Téléchargement OK")

    # ── 2. Labels ───────────────────────────────────────────────────────────
    print("⬇  Téléchargement labels...")
    lr = requests.get(LABELS_URL, timeout=30)
    lr.raise_for_status()
    labels_df = pd.read_csv(io.StringIO(lr.text))
    labels_df.to_csv(os.path.join(DATA_DIR, "labeled_anomalies.csv"), index=False)
    print("✅ Labels OK")

    # ── 3. Extraction canaux sélectionnés → CSV ──────────────────────────────
    all_channels = SMAP_CHANNELS + MSL_CHANNELS
    for chan in all_channels:
        tr_path = os.path.join(raw_train, f"{chan}.npy")
        te_path = os.path.join(raw_test,  f"{chan}.npy")
        if not os.path.exists(tr_path):
            print(f"  ⚠ {chan} introuvable, skipped")
            continue

        tr = np.load(tr_path)
        te = np.load(te_path)
        if tr.ndim == 1: tr = tr.reshape(-1,1)
        if te.ndim == 1: te = te.reshape(-1,1)

        # Labels
        row = labels_df[labels_df["chan_id"] == chan]
        labels_arr = np.zeros(len(te), dtype=int)
        if not row.empty:
            seqs = row.iloc[0]["anomaly_sequences"]
            if isinstance(seqs, str): seqs = ast.literal_eval(seqs)
            for s, e in seqs:
                labels_arr[s:e+1] = 1

        # Sauvegarder train/test/labels en CSV
        pd.DataFrame(tr, columns=[f"f{i}" for i in range(tr.shape[1])]) \
          .to_csv(os.path.join(DATA_DIR,"train",f"{chan}.csv"), index=False)
        df_te = pd.DataFrame(te, columns=[f"f{i}" for i in range(te.shape[1])])
        df_te["label"] = labels_arr
        df_te.to_csv(os.path.join(DATA_DIR,"test",f"{chan}.csv"), index=False)

        n_anom = labels_arr.sum()
        print(f"  ✅ {chan}: train={tr.shape} test={te.shape} anomalies={n_anom}")

    # ── 4. Nettoyer cache brut ───────────────────────────────────────────────
    import shutil
    shutil.rmtree(cache_dir)
    print()
    print("🎉 Terminé ! Maintenant committe le dossier data/ :")
    print("   git add data/")
    print('   git commit -m "data: add NASA SMAP/MSL channels"')
    print("   git push")

if __name__ == "__main__":
    main()