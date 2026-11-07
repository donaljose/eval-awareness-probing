"""Layer-wise linear probing and steering-direction derivation.

Implements:
 * per-layer logistic-regression probes (Chapter 3, sections 3.3–3.5)
 * optional PCA dimensionality reduction before probing
 * AUC as the primary metric, selectivity (AUC − control AUC) as the honest
 primary metric when the control AUC is high
 * a control task (shuffled labels) (selectivity control)
 * the difference-of-means steering direction 

PCA rationale
─────────────
When n_train << d_model (e.g. 35 examples in 1536 dimensions for Qwen-1.5B),
any labelling is linearly separable regardless of content, so AUC of 1.0 is
uninformative. Projecting to the top-k PCA components (k ≈ 50 by default)
reduces the effective dimensionality while retaining most variance.
PCA is fit on the training split only and applied to both train and test,
preventing any information leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


@dataclass
class LayerResult:
 layer: int
 auc: float
 accuracy: float
 control_auc: Optional[float] = None

 @property
 def selectivity(self) -> Optional[float]:
 """Probe AUC minus control AUC (Hewitt & Liang-style selectivity).

 A high selectivity means the signal is in the model's representation
 of the framing, not merely an artefact of the probe's capacity.
 This is the honest primary metric when W002 is triggered.
 """
 if self.control_auc is None:
 return None
 return self.auc - self.control_auc


@dataclass
class ProbeResults:
 layers: List[LayerResult]
 best_layer: int
 best_auc: float
 best_selectivity: Optional[float] # None if control task was not run
 scaler: StandardScaler
 classifier: LogisticRegression
 pca: Optional[PCA] = None # fitted PCA for the best layer (may be None)

 def to_dict(self) -> Dict:
 layer_dicts = []
 for r in self.layers:
 layer_dicts.append({
 "layer": r.layer,
 "auc": r.auc,
 "accuracy": r.accuracy,
 "control_auc": r.control_auc,
 "selectivity": r.selectivity,
 })
 return {
 "best_layer": self.best_layer,
 "best_auc": self.best_auc,
 "best_selectivity": self.best_selectivity,
 "layers": layer_dicts,
 }


def _fit_one_layer(
 X_train: np.ndarray,
 y_train: np.ndarray,
 X_test: np.ndarray,
 y_test: np.ndarray,
 cfg,
 pca: Optional[PCA] = None,
):
 """Fit a standardised logistic-regression probe; return (auc, acc, scaler, clf).

 If *pca* is not None it is applied after standardisation (PCA must already
 be fitted on the training set before this call).
 """
 scaler = StandardScaler().fit(X_train)
 Xtr = scaler.transform(X_train)
 Xte = scaler.transform(X_test)

 if pca is not None:
 Xtr = pca.transform(Xtr)
 Xte = pca.transform(Xte)

 clf = LogisticRegression(
 C=cfg.probe_C, max_iter=cfg.probe_max_iter, random_state=cfg.seed
 )
 clf.fit(Xtr, y_train)

 # A degenerate test split (one class only) makes AUC undefined.
 if len(np.unique(y_test)) < 2:
 auc = float("nan")
 else:
 auc = float(roc_auc_score(y_test, clf.predict_proba(Xte)[:, 1]))
 acc = float(accuracy_score(y_test, clf.predict(Xte)))
 return auc, acc, scaler, clf


def _maybe_fit_pca(X_train: np.ndarray, n_components: Optional[int]) -> Optional[PCA]:
 """Fit PCA on the (already standardised) training data if requested.

 Caps n_components at min(n_samples, n_features) − 1 to avoid sklearn errors
 when the training set is tiny.
 """
 if n_components is None:
 return None
 cap = min(X_train.shape[0], X_train.shape[1]) - 1
 k = min(n_components, cap)
 if k < 1:
 return None
 pca = PCA(n_components=k, random_state=0)
 pca.fit(X_train)
 return pca


def train_layer_probes(
 activations: np.ndarray,
 labels: np.ndarray,
 train_idx: np.ndarray,
 test_idx: np.ndarray,
 cfg,
) -> ProbeResults:
 """Train one probe per layer and select the best by test AUC.

 activations: (n_layers, n_examples, d_model)
 labels: (n_examples,) 1 = evaluation framing, 0 = deployment
 """
 if activations.ndim != 3:
 raise ValueError(f"activations must be 3-D, got shape {activations.shape}")
 n_layers, n_examples, _ = activations.shape
 if labels.shape[0] != n_examples:
 raise ValueError(
 f"labels ({labels.shape[0]}) does not match activations ({n_examples})"
 )

 y_train, y_test = labels[train_idx], labels[test_idx]
 if len(np.unique(y_train)) < 2:
 raise ValueError("Training split contains only one class.")

 rng = np.random.RandomState(cfg.seed)
 y_train_shuffled = rng.permutation(y_train) if cfg.run_control_task else None

 # PCA is fitted once per layer on the standardised training activations.
 # We pass n_components from config; _maybe_fit_pca caps it safely.
 pca_components = getattr(cfg, "probe_pca_components", None)

 results: List[LayerResult] = []
 fitted = {}

 for layer in range(n_layers):
 X = activations[layer]
 X_tr_raw = X[train_idx]
 X_te_raw = X[test_idx]

 # Fit PCA on standardised training data for this layer.
 if pca_components is not None:
 _scaler_pre = StandardScaler().fit(X_tr_raw)
 X_tr_scaled_pre = _scaler_pre.transform(X_tr_raw)
 pca = _maybe_fit_pca(X_tr_scaled_pre, pca_components)
 else:
 pca = None

 auc, acc, scaler, clf = _fit_one_layer(
 X_tr_raw, y_train, X_te_raw, y_test, cfg, pca=pca
 )

 control_auc = None
 if cfg.run_control_task:
 c_auc, _, _, _ = _fit_one_layer(
 X_tr_raw, y_train_shuffled, X_te_raw, y_test, cfg, pca=pca
 )
 control_auc = c_auc

 results.append(LayerResult(layer, auc, acc, control_auc))
 fitted[layer] = (scaler, clf, pca)

 valid = [r for r in results if not np.isnan(r.auc)]
 if not valid:
 raise ValueError("No layer produced a valid AUC (test split degenerate).")

 # Optionally exclude the first N layers from the "best" selection.
 # Layer 0 (the token-embedding output) is dominated by surface vocabulary,
 # so a probe there measures word choice rather than latent awareness.
 min_layer = int(getattr(cfg, "probe_min_layer", 0))
 eligible = [r for r in valid if r.layer >= min_layer]
 if not eligible:
 # Fall back to all valid layers rather than crash if min_layer is set
 # higher than the number of layers in the model.
 eligible = valid

 best = max(eligible, key=lambda r: r.auc)
 scaler, clf, _pca = fitted[best.layer]

 # Compute best selectivity for convenient top-level access.
 best_r = next(r for r in results if r.layer == best.layer)
 best_selectivity = best_r.selectivity

 return ProbeResults(
 layers=results,
 best_layer=best.layer,
 best_auc=best.auc,
 best_selectivity=best_selectivity,
 scaler=scaler,
 classifier=clf,
 pca=_pca,
 )


def difference_of_means_direction(
 activations_layer: np.ndarray, labels: np.ndarray
) -> np.ndarray:
 """Unit vector from the deployment-mean to the evaluation-mean.

 activations_layer: (n_examples, d_model)
 Returns a unit-norm (d_model,) vector.
 """
 if activations_layer.ndim != 2:
 raise ValueError(f"expected 2-D activations, got {activations_layer.shape}")
 if activations_layer.shape[0] != labels.shape[0]:
 raise ValueError("activations and labels length mismatch")

 pos = activations_layer[labels == 1]
 neg = activations_layer[labels == 0]
 if len(pos) == 0 or len(neg) == 0:
 raise ValueError("Both classes must be present to compute a direction.")

 direction = pos.mean(axis=0) - neg.mean(axis=0)
 norm = np.linalg.norm(direction)
 if norm == 0:
 raise ValueError("Class means are identical; direction is undefined.")
 return direction / norm


def _apply_probe_transform(probe: ProbeResults, activations_layer: np.ndarray) -> np.ndarray:
 """Apply scaler (and PCA if fitted) that were used when training the probe."""
 X = probe.scaler.transform(activations_layer)
 if probe.pca is not None:
 X = probe.pca.transform(X)
 return X


def predict_per_example(
 probe: ProbeResults, activations_layer: np.ndarray
) -> tuple:
 """Per-example probe output.

 Returns (predictions, confidences) where prediction 1 means the probe
 judged this activation to come from an EVALUATION framing, and confidence
 is P(evaluation) in [0, 1]. This is what answers, for a single prompt,
 "does the model appear to know it is being tested?"
 """
 X = _apply_probe_transform(probe, activations_layer)
 proba = probe.classifier.predict_proba(X)[:, 1]
 preds = (proba >= 0.5).astype(int)
 return preds, proba


def score_with_probe(
 probe: ProbeResults, activations_layer: np.ndarray, labels: np.ndarray
) -> float:
 """AUC of the fitted probe on (possibly steered) activations."""
 if len(np.unique(labels)) < 2:
 return float("nan")
 X = _apply_probe_transform(probe, activations_layer)
 return float(roc_auc_score(labels, probe.classifier.predict_proba(X)[:, 1]))
