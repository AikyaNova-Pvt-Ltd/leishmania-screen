"""
Core inference engine. Loads all artefacts once (lazy, thread-safe singleton)
and exposes a predict() function that accepts one or many SMILES strings.
"""

from __future__ import annotations

import importlib.resources
import threading
import warnings
from dataclasses import dataclass
from typing import List, Union

import joblib
import numpy as np
import pandas as pd
import torch

from ._model import _LeishNet
from ._features import validate_smiles, compute_features

THRESHOLD: float = 0.6
_LOCK = threading.Lock()
_ARTEFACTS: "_Artefacts | None" = None


@dataclass
class PredictionResult:
    smiles: str
    label: str            # "Active" | "Inactive" | "Invalid"
    probability: float | None
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "smiles": self.smiles,
            "label": self.label,
            "probability": round(self.probability, 4) if self.probability is not None else None,
            "error": self.error,
        }


class _Artefacts:
    """Holds all loaded model artefacts. Instantiated once."""

    def __init__(self):
        data_pkg = importlib.resources.files("leishmania_screen") / "data"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.selected_features: list[str] = list(
                joblib.load(str(data_pkg / "selected_features_LD.pkl"))
            )
            self.train_columns: list[str] = joblib.load(
                str(data_pkg / "train_columns_LD.pkl")
            )
            # Per-feature scalers: dict[feature_name -> StandardScaler]
            # Only the 129 features that were scaled during training are included.
            self.scalers: dict = joblib.load(str(data_pkg / "scalers_LD.pkl"))
            self.pca = joblib.load(str(data_pkg / "pca_100_LD.pkl"))

        self.model = _LeishNet(d=100)
        state = torch.load(
            str(data_pkg / "NN_model.pth"),
            map_location="cpu",
            weights_only=True,
        )
        self.model.load_state_dict(state)
        self.model.eval()


def _get_artefacts() -> _Artefacts:
    global _ARTEFACTS
    if _ARTEFACTS is None:
        with _LOCK:
            if _ARTEFACTS is None:
                _ARTEFACTS = _Artefacts()
    return _ARTEFACTS


def _transform(raw_df: pd.DataFrame, art: _Artefacts) -> np.ndarray:
    """Apply feature selection → per-feature scaling → PCA."""
    # Align to the 900 training features; missing columns filled with 0
    missing = {col: 0.0 for col in art.selected_features if col not in raw_df.columns}
    if missing:
        raw_df = pd.concat([raw_df, pd.DataFrame(missing, index=raw_df.index)], axis=1)

    X_df = raw_df[art.train_columns].copy()

    # Apply per-feature scalers only to features that were scaled during training
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for feature, scaler in art.scalers.items():
            if feature in X_df.columns:
                X_df[feature] = scaler.transform(X_df[[feature]])
        X_pca = art.pca.transform(X_df.values.astype(np.float64))

    return X_pca.astype(np.float32)


def _run_model(X_pca: np.ndarray, art: _Artefacts) -> np.ndarray:
    tensor = torch.tensor(X_pca, dtype=torch.float32)
    with torch.no_grad():
        logits = art.model(tensor)
        probs  = torch.sigmoid(logits).cpu().numpy().ravel()
    return probs


def predict(
    smiles: Union[str, List[str]],
) -> Union[PredictionResult, List[PredictionResult]]:
    """
    Predict antileishmanial activity for one or more SMILES strings.

    Parameters
    ----------
    smiles : str or list of str
        SMILES string(s) to screen.

    Returns
    -------
    PredictionResult or list of PredictionResult
        Each result contains .smiles, .label, .probability, .error.
    """
    single = isinstance(smiles, str)
    inputs: list[str] = [smiles] if single else list(smiles)

    art = _get_artefacts()

    # --- validate all SMILES ---
    valid_idx:   list[int]        = []
    valid_mols:  list            = []
    valid_smi:   list[str]       = []
    results: list[PredictionResult | None] = [None] * len(inputs)

    for i, smi in enumerate(inputs):
        mol, err = validate_smiles(smi)
        if err:
            results[i] = PredictionResult(
                smiles=smi, label="Invalid", probability=None, error=err
            )
        else:
            valid_idx.append(i)
            valid_mols.append(mol)
            valid_smi.append(smi)

    # --- feature generation for valid molecules ---
    if valid_mols:
        feature_rows = [
            compute_features(smi, mol)
            for smi, mol in zip(valid_smi, valid_mols)
        ]
        raw_df = pd.concat(feature_rows, ignore_index=True)
        X_pca  = _transform(raw_df, art)
        probs  = _run_model(X_pca, art)

        for local_j, global_i in enumerate(valid_idx):
            p = float(probs[local_j])
            results[global_i] = PredictionResult(
                smiles=valid_smi[local_j],
                label="Active" if p >= THRESHOLD else "Inactive",
                probability=p,
            )

    out = [r for r in results]  # type: list[PredictionResult]
    return out[0] if single else out
