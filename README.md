# leishmania-screen

Neural network–based virtual screening for antileishmanial activity against *Leishmania donovani*.

Trained on a curated dataset of 6,699 compounds. Achieves **ROC-AUC 0.884** on the held-out test set.

---

## Installation

```bash
pip install leishmania-screen
```

> **RDKit note:** RDKit is listed as a dependency (`rdkit>=2023.3`). If your environment manages RDKit via conda, install it there first:
> ```bash
> conda install -c conda-forge rdkit
> pip install leishmania-screen
> ```

---

## Quick start — Python API

```python
from leishmania_screen import predict

# Single compound
result = predict("CCO")
print(result.label)        # "Active" or "Inactive"
print(result.probability)  # float in [0, 1]

# Batch
results = predict(["CCO", "CC(=O)Oc1ccccc1C(=O)O", "not_valid"])
for r in results:
    print(r.smiles, r.label, r.probability, r.error)
```

### `PredictionResult` fields

| Field | Type | Description |
|---|---|---|
| `smiles` | `str` | The input SMILES string |
| `label` | `str` | `"Active"`, `"Inactive"`, or `"Invalid"` |
| `probability` | `float \| None` | Sigmoid output of the model (0–1); `None` for invalid inputs |
| `error` | `str \| None` | Reason for invalidity; `None` for valid inputs |

---

## Quick start — Command line

```bash
# Single SMILES
leishscreen --smiles "CC(=O)Oc1ccccc1C(=O)O"

# Batch from CSV (must have a column named 'smiles')
leishscreen --file compounds.csv --output results.csv

# Batch from plain text (one SMILES per line)
leishscreen --file smiles.txt --output results.csv

# Custom column name
leishscreen --file library.csv --smiles-col SMILES --output results.csv

# Version
leishscreen --version
```

---

## Model details

| Item | Value |
|---|---|
| Target | *Leishmania donovani* (binary: Active / Inactive) |
| Training dataset | 6,699 compounds (2,574 active, 4,125 inactive) |
| Feature pipeline | 218 RDKit descriptors + 2,728 fingerprint bits → 900-feature MI selection → StandardScaler → PCA (100 components) |
| Architecture | Linear(100→512)→BN→GELU→Drop(0.30) → Linear(512→256)→BN→GELU→Drop(0.25) → Linear(256→128)→GELU→Drop(0.15) → Linear(128→1) |
| Loss | `BCEWithLogitsLoss` with class-imbalance `pos_weight` |
| Optimizer | AdamW (lr=8e-4, weight_decay=2e-4) + gradient clipping |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |
| Training strategy | Multi-seed ensemble (seeds 42, 52, 62); early stopping (patience=18) |
| Classification threshold | **0.60** (optimised on validation set: precision ≥ 0.70, max F1) |
| Test ROC-AUC | **0.884** |
| Test PR-AUC | **0.828** |
| Test Accuracy | **0.815** |
| Test Balanced Accuracy | **0.810** |

### Fingerprints used

| Type | Parameters | Bits |
|---|---|---|
| Morgan (ECFP-like) | radius=2 | 512 |
| Avalon | — | 512 |
| Topological Torsion | — | 512 |
| Atom Pair | — | 512 |
| MACCS Keys | — | 167 |
| RDKit Path-Based | minPath=5, maxPath=7 | 512 |

---

## Citation

If you use this package in your research, please cite:

```
[Citation to be added upon publication]
```

---

## License

MIT
