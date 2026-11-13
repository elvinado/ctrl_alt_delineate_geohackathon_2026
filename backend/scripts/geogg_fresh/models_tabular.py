"""Tabular model zoo for stage 07: every candidate with its (small, documented) grid.

Each model is built from ``(name, kind, params, n_rows)``; ``kind`` is 'binary' or
'regression'.  All models see the SAME frozen 06 transform of the SAME predictors, so
differences come from the learner, not from the preprocessing.

| name         | binary                         | regression           | why it is here                         |
|--------------|--------------------------------|----------------------|----------------------------------------|
| prior        | class prevalence               | training mean        | the floor: no information              |
| depth_trend  | logistic on TWT, TWT^2         | linear on TWT, TWT^2 | burial/compaction alone (NOT seismic)  |
| linear       | L2 logistic regression         | ridge                | regularised linear benchmark           |
| extratrees   | extremely randomised trees     | idem                 | robust non-linear, few assumptions     |
| hist_gbm     | sklearn histogram boosting     | idem                 | gradient boosting without extra deps   |
| xgboost      | XGBoost (if installed)         | idem                 | the brief's boosted-tree benchmark     |
| mlp          | 2-layer perceptron             | idem                 | smooth non-linear benchmark            |

``depth_trend`` uses the sample's TWT, which every sample has at inference time; it is a
reference only - it measures how much of a target is explained by burial depth, which
every seismic model must beat to claim it sees rock properties.
"""

from __future__ import annotations

import importlib
import itertools

import numpy as np

GRIDS: dict[str, dict[str, list]] = {
    "prior": {},
    "depth_trend": {},
    "linear": {"C": [0.01, 0.1, 1.0, 10.0]},                          # ridge alpha = 1 / C
    "extratrees": {"min_samples_leaf": [3, 10, 30], "max_features": [0.5]},
    "hist_gbm": {"learning_rate": [0.05], "max_leaf_nodes": [7, 15], "min_samples_leaf": [20, 60]},
    "xgboost": {"max_depth": [2, 4], "learning_rate": [0.05], "n_estimators": [300], "min_child_weight": [1, 10]},
    "mlp": {"alpha": [1e-3, 1e-1], "hidden_layer_sizes": [[64, 32]]},
}
BASELINES = ("prior", "depth_trend")


def available(name: str) -> tuple[bool, str]:
    """(usable, reason).  Only xgboost is optional."""
    if name == "xgboost":
        try:
            importlib.import_module("xgboost")
        except ImportError:
            return False, "xgboost is not installed in this environment"
    if name not in GRIDS:
        return False, f"unknown model '{name}'"
    return True, ""


def grid(name: str, override: dict | None = None) -> list[dict]:
    """All parameter combinations (a single empty dict for models without a grid)."""
    g = dict(GRIDS[name])
    if override:
        g.update(override)
    if not g:
        return [{}]
    keys = sorted(g)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(g[k] for k in keys))]


def build(name: str, kind: str, params: dict, seed: int = 2026, n_jobs: int = -1, class_weight: str | None = "balanced",
          pos_weight: float = 1.0):
    """A fresh, unfitted estimator with fit / predict (and predict_proba for binary)."""
    binary = kind == "binary"
    if name == "prior":
        from sklearn.dummy import DummyClassifier, DummyRegressor

        return DummyClassifier(strategy="prior") if binary else DummyRegressor(strategy="mean")
    if name in ("linear", "depth_trend"):
        from sklearn.linear_model import LogisticRegression, Ridge

        C = float(params.get("C", 1e4 if name == "depth_trend" else 1.0))
        if binary:
            return LogisticRegression(C=C, class_weight=class_weight, max_iter=2000)
        return Ridge(alpha=1.0 / C)
    if name == "extratrees":
        from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

        kw = dict(n_estimators=int(params.get("n_estimators", 200)),
                  min_samples_leaf=int(params.get("min_samples_leaf", 10)),
                  max_features=params.get("max_features", 0.5), n_jobs=n_jobs, random_state=seed)
        return ExtraTreesClassifier(class_weight=class_weight, **kw) if binary else ExtraTreesRegressor(**kw)
    if name == "hist_gbm":
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

        kw = dict(learning_rate=float(params.get("learning_rate", 0.05)), max_iter=int(params.get("max_iter", 200)),
                  max_leaf_nodes=int(params.get("max_leaf_nodes", 15)),
                  min_samples_leaf=int(params.get("min_samples_leaf", 20)),
                  l2_regularization=float(params.get("l2_regularization", 1.0)), random_state=seed)
        return HistGradientBoostingClassifier(class_weight=class_weight, **kw) if binary else \
            HistGradientBoostingRegressor(**kw)
    if name == "xgboost":
        import xgboost as xgb

        kw = dict(n_estimators=int(params.get("n_estimators", 300)), max_depth=int(params.get("max_depth", 3)),
                  learning_rate=float(params.get("learning_rate", 0.05)),
                  min_child_weight=float(params.get("min_child_weight", 1)), subsample=0.8, colsample_bytree=0.8,
                  reg_lambda=1.0, tree_method="hist", n_jobs=n_jobs, random_state=seed)
        if binary:
            return xgb.XGBClassifier(scale_pos_weight=float(pos_weight) if class_weight else 1.0,
                                     eval_metric="logloss", **kw)
        return xgb.XGBRegressor(**kw)
    if name == "mlp":
        from sklearn.neural_network import MLPClassifier, MLPRegressor

        kw = dict(hidden_layer_sizes=tuple(params.get("hidden_layer_sizes", [64, 32])),
                  alpha=float(params.get("alpha", 1e-3)), learning_rate_init=1e-3, max_iter=400,
                  early_stopping=True, validation_fraction=0.15, n_iter_no_change=20, random_state=seed)
        if binary:
            return MLPClassifier(**kw)
        from sklearn.compose import TransformedTargetRegressor
        from sklearn.preprocessing import StandardScaler

        # fractions of 0.05-0.4 are tiny for a network's output layer: learn the standardised target
        return TransformedTargetRegressor(regressor=MLPRegressor(**kw), transformer=StandardScaler())
    raise ValueError(f"unknown model {name}")


def predict(model, kind: str, X: np.ndarray) -> np.ndarray:
    """Probability of class 1 (binary) or the regression value."""
    if kind == "binary":
        P = model.predict_proba(X)
        classes = list(getattr(model, "classes_", [0, 1]))
        if 1 in classes:
            return P[:, classes.index(1)]
        return np.zeros(X.shape[0])                    # the training rows held no class 1
    return np.asarray(model.predict(X), float)


def depth_features(twt: np.ndarray, centre: float, scale: float) -> np.ndarray:
    """[z, z^2] with z = (TWT - centre) / scale (centre / scale from the fit rows)."""
    z = (np.asarray(twt, float) - centre) / (scale if scale > 0 else 1.0)
    return np.column_stack([z, z ** 2])


def params_key(params: dict) -> str:
    """Stable, readable text for a parameter set (used in tables and for the final vote)."""
    return ";".join(f"{k}={params[k]}" for k in sorted(params)) or "default"
