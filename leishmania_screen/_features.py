"""
Molecular feature generation — replicates the training pipeline exactly.

Feature vector layout (2,946 columns total):
  [0:218]    RDKit descriptors          (218)
  [218:730]  Morgan r=2 512-bit (Mfpt_) (512)
  [730:1242] Avalon 512-bit             (512)
  [1242:1754] Topological Torsion 512-bit (512)
  [1754:2266] Atom Pair 512-bit         (512)
  [2266:2433] MACCS keys               (167)
  [2433:2945] RDKit path-based 512-bit  (512)
  [2944:2946] ID column (dropped before model; kept here for tracing)
"""

from __future__ import annotations

import os
import warnings
from contextlib import contextmanager
from typing import List, Tuple

import numpy as np
import pandas as pd

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    try:
        from rdkit.Avalon import pyAvalonTools
        _AVALON_OK = True
    except ImportError:
        _AVALON_OK = False

RDLogger.DisableLog("rdApp.*")


@contextmanager
def _silence_stderr():
    devnull = open(os.devnull, "w")
    old = os.dup(2)
    os.dup2(devnull.fileno(), 2)
    try:
        yield
    finally:
        os.dup2(old, 2)
        devnull.close()


_DESCRIPTOR_NAMES: list[str] | None = None


def _descriptor_names() -> list[str]:
    global _DESCRIPTOR_NAMES
    if _DESCRIPTOR_NAMES is None:
        _DESCRIPTOR_NAMES = [n for n, _ in Descriptors.descList]
    return _DESCRIPTOR_NAMES


def validate_smiles(smiles: str) -> Tuple[Chem.Mol | None, str | None]:
    """
    Return (mol, None) on success or (None, error_message) on failure.
    Applies both MolFromSmiles and SanitizeMol checks.
    """
    if not smiles or not isinstance(smiles, str):
        return None, "Input is empty or not a string."
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None, f"RDKit could not parse SMILES: '{smiles}'"
    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:
        return None, f"Sanitization failed: {exc}"
    return mol, None


def _rdkit_descriptors(mol: Chem.Mol) -> np.ndarray:
    desc_dict = Descriptors.CalcMolDescriptors(mol)
    return np.array([desc_dict[n] for n in _descriptor_names()], dtype=np.float64)


def _bitvect_to_array(bv) -> np.ndarray:
    arr = np.zeros(bv.GetNumBits(), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


def _morgan(mol: Chem.Mol) -> np.ndarray:
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=512)
    return _bitvect_to_array(bv)


def _avalon(smiles: str) -> np.ndarray:
    if _AVALON_OK:
        bv = pyAvalonTools.GetAvalonFP(smiles, isSmiles=True, nBits=512)
        return _bitvect_to_array(bv)
    # Fallback: zero vector if Avalon not available
    return np.zeros(512, dtype=np.uint8)


def _topo_torsion(mol: Chem.Mol) -> np.ndarray:
    bv = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=512)
    return _bitvect_to_array(bv)


def _atom_pair(mol: Chem.Mol) -> np.ndarray:
    bv = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=512)
    return _bitvect_to_array(bv)


def _maccs(mol: Chem.Mol) -> np.ndarray:
    bv = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
    return _bitvect_to_array(bv)


def _rdkit_path(mol: Chem.Mol) -> np.ndarray:
    bv = AllChem.RDKFingerprint(mol, minPath=5, maxPath=7, fpSize=512)
    return _bitvect_to_array(bv)


def _column_names() -> List[str]:
    desc_names = _descriptor_names()
    morgan_names  = [f"Mfpt_{i}"   for i in range(512)]
    avalon_names  = [f"Avfpt_{i}"  for i in range(512)]
    tt_names      = [f"TT_fpt_{i}" for i in range(512)]
    ap_names      = [f"AP_fpt_{i}" for i in range(512)]
    mc_names      = [f"MC_fpt_{i}" for i in range(167)]
    rd_names      = [f"RD_fpt_{i}" for i in range(512)]
    return desc_names + morgan_names + avalon_names + tt_names + ap_names + mc_names + rd_names


def compute_features(smiles: str, mol: Chem.Mol) -> pd.DataFrame:
    """
    Build the full 2,946-column feature row for one molecule.
    Returns a single-row DataFrame with named columns.
    """
    with warnings.catch_warnings(), _silence_stderr():
        warnings.simplefilter("ignore")
        desc  = _rdkit_descriptors(mol)
        mfp   = _morgan(mol)
        avfp  = _avalon(smiles)
        ttfp  = _topo_torsion(mol)
        apfp  = _atom_pair(mol)
        mcfp  = _maccs(mol)
        rdfp  = _rdkit_path(mol)

    row = np.concatenate([desc, mfp, avfp, ttfp, apfp, mcfp, rdfp])
    return pd.DataFrame([row], columns=_column_names())
