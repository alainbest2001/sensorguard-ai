"""
models/detector.py
SensorGuard AI — Moteur de détection d'anomalies
Pipeline : normalisation → reconstruction par fenêtre → score composite → seuil adaptatif
Inspiré des principes de diffusion (reconstruction residuals) sans dépendance GPU lourde.
"""

import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")


# ── Fenêtrage ────────────────────────────────────────────────────────────────

def sliding_windows(data: np.ndarray, window: int = 64, step: int = 1) -> np.ndarray:
    """Découpe (T, D) en fenêtres (N, window, D)."""
    T, D = data.shape
    windows = []
    for i in range(0, T - window + 1, step):
        windows.append(data[i : i + window])
    return np.array(windows)  # (N, window, D)


# ── Normalisation physique locale (cycle par cycle) ──────────────────────────

class PhysicalNormalizer:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(-1, 1))

    def fit(self, train: np.ndarray) -> "PhysicalNormalizer":
        self.scaler.fit(train)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        return self.scaler.transform(data)

    def fit_transform(self, train: np.ndarray) -> np.ndarray:
        return self.scaler.fit_transform(train)


# ── Reconstruction par moyenne glissante multi-échelle ───────────────────────
# Simule le principe du débruitage : reconstruction "normale" ≈ signal lissé.
# L'erreur de reconstruction est élevée là où le signal s'écarte du pattern appris.

class MultiScaleReconstructor:
    """
    Reconstruit le signal via moyenne glissante à plusieurs échelles.
    Approxime l'idée du débruitage sans GPU.
    Utilise des échelles fines (poids élevé) pour mieux capturer les spikes.
    """

    def __init__(self, scales: list[int] = [4, 8, 16, 32]):
        self.scales = scales

    def reconstruct(self, data: np.ndarray) -> np.ndarray:
        """Retourne reconstruction (T, D) = moyenne pondérée des lissages."""
        reconstructions = []
        weights = []
        for i, s in enumerate(self.scales):
            kernel = np.ones(s) / s
            rec = np.apply_along_axis(
                lambda col: np.convolve(col, kernel, mode="same"), axis=0, arr=data
            )
            reconstructions.append(rec)
            weights.append(1.0 / (i + 1))  # plus de poids aux petites échelles
        w = np.array(weights) / sum(weights)
        return sum(r * wt for r, wt in zip(reconstructions, w))

    def residuals(self, data: np.ndarray) -> np.ndarray:
        """Retourne l'erreur de reconstruction (T, D)."""
        rec = self.reconstruct(data)
        return np.abs(data - rec)


# ── Score d'anomalie composite ────────────────────────────────────────────────

class AnomalyScorer:
    """
    Score composite inspiré des méthodes diffusion + reconstruction :
      1. Résidu multi-échelle (reconstruction error) — composante principale
      2. Gradient temporel (changements brusques inter-pas)
      Combinés via max par dimension puis lissage léger.
    """

    def __init__(self, window: int = 32):
        self.window = window
        self.reconstructor = MultiScaleReconstructor()

    def compute(self, data: np.ndarray) -> np.ndarray:
        """
        data : np.ndarray (T, D)
        Retourne score (T,) normalisé [0, 1]
        """
        from scipy.ndimage import uniform_filter1d

        # 1. Résidu de reconstruction multi-échelle
        residuals = self.reconstructor.residuals(data)           # (T, D)

        # 2. Gradient temporel (détecte les changements brusques)
        grad = np.abs(np.diff(data, axis=0, prepend=data[:1]))   # (T, D)

        # Combine : prend le max entre résidu et 50% du gradient par dimension
        combined = np.maximum(residuals, 0.5 * grad)             # (T, D)
        raw = combined.mean(axis=1)                              # (T,)

        # Lissage léger (fenêtre=3) pour réduire les FP ponctuels isolés
        raw = uniform_filter1d(raw, size=3)

        # Normalisation [0,1]
        mn, mx = raw.min(), raw.max()
        if mx - mn > 1e-8:
            score = (raw - mn) / (mx - mn)
        else:
            score = np.zeros_like(raw)

        return score


# ── Seuillage adaptatif ───────────────────────────────────────────────────────

class AdaptiveThreshold:
    """
    Seuil adaptatif basé sur la distribution des scores sur le train.
    Percentile configurable — par défaut 95e.
    """

    def __init__(self, percentile: float = 95.0):
        self.percentile = percentile
        self.threshold  = None

    def fit(self, train_scores: np.ndarray) -> float:
        self.threshold = float(np.percentile(train_scores, self.percentile))
        return self.threshold

    def predict(self, scores: np.ndarray) -> np.ndarray:
        if self.threshold is None:
            raise RuntimeError("AdaptiveThreshold non entraîné — appelez .fit() d'abord.")
        return (scores >= self.threshold).astype(int)


# ── Pipeline complet ──────────────────────────────────────────────────────────

class SensorGuardDetector:
    """
    Pipeline complet :
      PhysicalNormalizer → MultiScaleReconstructor → AnomalyScorer → AdaptiveThreshold
    """

    def __init__(self, window: int = 32, threshold_pct: float = 95.0):
        self.normalizer = PhysicalNormalizer()
        self.scorer     = AnomalyScorer(window=window)
        self.thresholder = AdaptiveThreshold(percentile=threshold_pct)
        self.threshold_pct = threshold_pct
        self._fitted = False

    def fit(self, train: np.ndarray) -> "SensorGuardDetector":
        """Entraîne sur données normales."""
        norm_train = self.normalizer.fit_transform(train)
        train_scores = self.scorer.compute(norm_train)
        self.thresholder.fit(train_scores)
        self._fitted = True
        return self

    def predict(self, test: np.ndarray) -> dict:
        """
        Retourne :
          scores     : np.ndarray (T,) — score d'anomalie brut [0,1]
          predictions: np.ndarray (T,) — 0/1
          threshold  : float
        """
        if not self._fitted:
            raise RuntimeError("SensorGuardDetector non entraîné.")

        norm_test    = self.normalizer.transform(test)
        scores       = self.scorer.compute(norm_test)
        predictions  = self.thresholder.predict(scores)

        return {
            "scores"      : scores,
            "predictions" : predictions,
            "threshold"   : self.thresholder.threshold,
        }

    def evaluate(self, predictions: np.ndarray, labels: np.ndarray) -> dict:
        """Calcule les métriques clés."""
        TP = int(np.sum((predictions == 1) & (labels == 1)))
        FP = int(np.sum((predictions == 1) & (labels == 0)))
        TN = int(np.sum((predictions == 0) & (labels == 0)))
        FN = int(np.sum((predictions == 0) & (labels == 1)))

        precision = TP / (TP + FP + 1e-8)
        recall    = TP / (TP + FN + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)
        fpr       = FP / (FP + TN + 1e-8)

        return {
            "TP"       : TP,
            "FP"       : FP,
            "TN"       : TN,
            "FN"       : FN,
            "precision": round(precision, 4),
            "recall"   : round(recall, 4),
            "f1"       : round(f1, 4),
            "fpr"      : round(fpr, 4),
        }
