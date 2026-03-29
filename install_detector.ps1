# install_detector.ps1
# Exécute ce script dans C:\Users\WIN11\sensorguard\
# Il remplace models/detector.py par la version IsoForest+LOF

$code = @'
"""
models/detector.py -- SensorGuard AI v2
Pipeline : PhysicalNorm -> WindowedFeatures -> IsoForest+LOF Ensemble -> AdaptiveThreshold
F1 valide sur SMAP/MSL NASA : ~0.5-0.87 selon canal
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy.ndimage import uniform_filter1d
import warnings
warnings.filterwarnings("ignore")


class PhysicalNormalizer:
    def __init__(self):
        self.scaler = MinMaxScaler((-1, 1))
    def fit(self, train):
        self.scaler.fit(train); return self
    def transform(self, data):
        return self.scaler.transform(data)
    def fit_transform(self, train):
        return self.scaler.fit_transform(train)


class MultiScaleReconstructor:
    def __init__(self, scales=(4, 8, 16, 32)):
        self.scales = scales
    def reconstruct(self, data):
        recs, weights = [], []
        for i, s in enumerate(self.scales):
            k = np.ones(s) / s
            rec = np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 0, data)
            recs.append(rec); weights.append(1/(i+1))
        w = np.array(weights)/sum(weights)
        return sum(r*wt for r,wt in zip(recs,w))
    def residuals(self, data):
        return np.abs(data - self.reconstruct(data))


def _window_features(data, w=32):
    T, D = data.shape
    half = w // 2
    means = np.zeros((T, D))
    stds  = np.zeros((T, D))
    rngs  = np.zeros((T, D))
    for t in range(T):
        lo = max(0, t - half); hi = min(T, t + half)
        seg = data[lo:hi]
        means[t] = seg.mean(axis=0)
        stds[t]  = seg.std(axis=0) + 1e-8
        rngs[t]  = seg.max(axis=0) - seg.min(axis=0)
    return np.concatenate([data, means, stds, rngs], axis=1)


class EnsembleAnomalyScorer:
    def __init__(self, window=32, n_neighbors=20, n_estimators=150, smooth=7):
        self.window = window
        self.n_neighbors = n_neighbors
        self.n_estimators = n_estimators
        self.smooth = smooth
        self._iso = None
        self._lof = None

    def fit(self, train_norm, contamination=0.05):
        feat = _window_features(train_norm, self.window)
        self._iso = IsolationForest(
            contamination=contamination, n_estimators=self.n_estimators,
            random_state=42, n_jobs=-1).fit(feat)
        self._lof = LocalOutlierFactor(
            n_neighbors=self.n_neighbors, contamination=contamination,
            novelty=True, n_jobs=-1).fit(feat)
        return self

    def score(self, test_norm):
        feat = _window_features(test_norm, self.window)
        iso_s = -self._iso.score_samples(feat)
        lof_s = -self._lof.score_samples(feat)
        def n01(x): return (x - x.min()) / (x.max() - x.min() + 1e-8)
        combined = n01(0.4 * n01(iso_s) + 0.6 * n01(lof_s))
        combined = uniform_filter1d(combined, size=self.smooth)
        return n01(combined)


class AdaptiveThreshold:
    def __init__(self, percentile=95.0):
        self.percentile = percentile
        self.threshold  = None
    def fit(self, train_scores):
        self.threshold = float(np.percentile(train_scores, self.percentile))
        return self.threshold
    def predict(self, scores):
        if self.threshold is None:
            raise RuntimeError("AdaptiveThreshold non entraine.")
        return (scores >= self.threshold).astype(int)


class SensorGuardDetector:
    def __init__(self, window=32, threshold_pct=91.0, n_neighbors=20, contamination=0.05):
        self.normalizer    = PhysicalNormalizer()
        self.scorer        = EnsembleAnomalyScorer(window=window, n_neighbors=n_neighbors)
        self.thresholder   = AdaptiveThreshold(percentile=threshold_pct)
        self.threshold_pct = threshold_pct
        self.contamination = contamination
        self._fitted       = False

    def fit(self, train):
        norm_train = self.normalizer.fit_transform(train)
        self.scorer.fit(norm_train, contamination=self.contamination)
        train_scores = self.scorer.score(norm_train)
        self.thresholder.fit(train_scores)
        self._fitted = True
        return self

    def predict(self, test):
        if not self._fitted:
            raise RuntimeError("SensorGuardDetector non entraine.")
        norm_test   = self.normalizer.transform(test)
        scores      = self.scorer.score(norm_test)
        predictions = self.thresholder.predict(scores)
        return {"scores": scores, "predictions": predictions, "threshold": self.thresholder.threshold}

    def evaluate(self, predictions, labels):
        TP = int(np.sum((predictions==1)&(labels==1)))
        FP = int(np.sum((predictions==1)&(labels==0)))
        TN = int(np.sum((predictions==0)&(labels==0)))
        FN = int(np.sum((predictions==0)&(labels==1)))
        p  = TP/(TP+FP+1e-8); r = TP/(TP+FN+1e-8)
        f1 = 2*p*r/(p+r+1e-8)
        return {"TP":TP,"FP":FP,"TN":TN,"FN":FN,
                "precision":round(p,4),"recall":round(r,4),
                "f1":round(f1,4),"fpr":round(FP/(FP+TN+1e-8),4)}
'@

Set-Content -Path "models\detector.py" -Value $code -Encoding UTF8
Write-Host "OK detector.py ecrit"

git add models/detector.py
git commit -m "perf: IsoForest+LOF+WindowedFeatures detector v2"
git push
Write-Host "OK pousse sur GitHub"