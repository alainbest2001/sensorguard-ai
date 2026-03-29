"""
Génère des données synthétiques réalistes calées sur les VRAIES métadonnées SMAP/MSL
(tailles, labels, séquences d'anomalies) extraites de labeled_anomalies.csv NASA.
Produit des fichiers CSV dans data/train/ et data/test/ à committer sur GitHub.
"""
import os, ast, io, requests
import numpy as np
import pandas as pd

OUT_TRAIN = "data/train"
OUT_TEST  = "data/test"
os.makedirs(OUT_TRAIN, exist_ok=True)
os.makedirs(OUT_TEST,  exist_ok=True)

# ── Vraies métadonnées extraites de labeled_anomalies.csv NASA ───────────────
CHANNELS = {
    # chan_id: (spacecraft, anomaly_sequences, num_values_test, n_features)
    "P-1" : ("SMAP", [[2149,2349],[4536,4844],[3539,3779]], 8505, 25),
    "P-2" : ("SMAP", [[5350,6575]], 8209, 25),
    "P-3" : ("SMAP", [[5401,6736]], 8493, 25),
    "P-4" : ("SMAP", [[950,1080],[2150,2350],[4770,4880]], 7783, 25),
    "P-7" : ("SMAP", [[4950,6600]], 8071, 25),
    "T-1" : ("SMAP", [[2399,3898],[6550,6585]], 8612, 25),
    "T-2" : ("SMAP", [[6840,8624]], 8625, 25),
    "E-1" : ("SMAP", [[5000,5030],[5610,6086]], 8516, 25),
    "A-1" : ("SMAP", [[4690,4774]], 8640, 25),
    "G-1" : ("SMAP", [[4770,4890]], 8469, 25),
    "M-1" : ("MSL",  [[1110,2250]], 2277, 55),
    "M-2" : ("MSL",  [[1110,2250]], 2277, 55),
    "M-3" : ("MSL",  [[1250,1500]], 2127, 55),
    "M-4" : ("MSL",  [[1250,1500]], 2038, 55),
    "P-10": ("MSL",  [[4590,4720]], 6100, 55),
    "P-15": ("MSL",  [[1390,1410]], 2856, 55),
    "F-7" : ("MSL",  [[1250,1450],[2670,2790],[3325,3425]], 5054, 55),
    "C-1" : ("MSL",  [[550,750],[2100,2210]], 2264, 55),
}

def make_channel(chan_id, anom_seqs, T_test, D, seed=0):
    np.random.seed(seed)
    T_train = T_test  # même longueur approx pour le train

    # Signal de base : mix de sinusoïdes + bruit (simule télémétrie)
    t_tr = np.linspace(0, 4*np.pi, T_train)
    t_te = np.linspace(0, 4*np.pi, T_test)

    freqs = 0.3 + 0.7 * np.random.rand(D)
    phases = 2*np.pi * np.random.rand(D)
    amps   = 0.5 + 0.5 * np.random.rand(D)

    def signal(t):
        base = np.stack([
            amps[i] * np.sin(freqs[i]*t + phases[i]) for i in range(D)
        ], axis=1)
        noise = 0.05 * np.random.randn(len(t), D)
        return base + noise

    train = signal(t_tr)
    test  = signal(t_te)

    # Inject anomalies sur le test
    labels = np.zeros(T_test, dtype=int)
    for s, e in anom_seqs:
        s = min(s, T_test-1)
        e = min(e, T_test-1)
        if s >= e:
            continue
        # Anomalie = shift + spike sur quelques dimensions
        n_anom_dims = max(1, D // 8)
        anom_dims = np.random.choice(D, n_anom_dims, replace=False)
        amplitude = 2.5 + 1.5 * np.random.rand()
        test[s:e+1, anom_dims] += amplitude * np.random.choice([-1,1])
        labels[s:e+1] = 1

    return train, test, labels

print("Génération des canaux SMAP/MSL synthétiques réalistes...")
for i, (chan_id, (spacecraft, anom_seqs, T_test, D)) in enumerate(CHANNELS.items()):
    train, test, labels = make_channel(chan_id, anom_seqs, T_test, D, seed=i*7)

    # Sauvegarder (on garde seulement les 4 premières features pour alléger)
    D_save = min(D, 4)
    cols = [f"f{j}" for j in range(D_save)]

    pd.DataFrame(train[:, :D_save], columns=cols)\
      .to_csv(f"{OUT_TRAIN}/{chan_id}.csv", index=False)

    df_te = pd.DataFrame(test[:, :D_save], columns=cols)
    df_te["label"] = labels
    df_te.to_csv(f"{OUT_TEST}/{chan_id}.csv", index=False)

    n_anom = labels.sum()
    pct = 100*n_anom/T_test
    print(f"  ✅ {chan_id:5s} ({spacecraft}) — test={T_test:5d} pts  anomalies={n_anom:4d} ({pct:.1f}%)")

# Sauvegarder aussi les labels CSV pour référence
labels_data = []
for chan_id, (spacecraft, anom_seqs, T_test, D) in CHANNELS.items():
    labels_data.append({"chan_id": chan_id, "spacecraft": spacecraft,
                         "anomaly_sequences": str(anom_seqs), "num_values": T_test})
pd.DataFrame(labels_data).to_csv("data/labeled_anomalies.csv", index=False)

print()
print(f"✅ {len(CHANNELS)} canaux générés → data/train/ et data/test/")
print()
print("Maintenant exécute :")
print("  git add data/")
print('  git commit -m "data: add synthetic SMAP/MSL channels (NASA metadata)"')
print("  git push")
