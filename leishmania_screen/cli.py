"""
Command-line interface for leishmania-screen.

Usage examples
--------------
Single SMILES:
    leishscreen --smiles "CCO"

Batch from CSV (must have a 'smiles' column):
    leishscreen --file compounds.csv --output results.csv

Batch from plain text (one SMILES per line):
    leishscreen --file smiles.txt --output results.csv

Print version:
    leishscreen --version
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="leishscreen",
        description=(
            "Virtual screening for antileishmanial activity against "
            "Leishmania donovani using a trained neural network."
        ),
    )
    p.add_argument("--version", action="store_true", help="Print version and exit.")

    group = p.add_mutually_exclusive_group()
    group.add_argument("--smiles", type=str, metavar="SMILES",
                       help="A single SMILES string to predict.")
    group.add_argument("--file", type=Path, metavar="FILE",
                       help="Path to a CSV (with 'smiles' column) or plain-text file (one SMILES per line).")

    p.add_argument("--output", type=Path, metavar="FILE", default=None,
                   help="Output CSV path. If omitted, results are printed to stdout.")
    p.add_argument("--smiles-col", type=str, default="smiles", metavar="COL",
                   help="Column name to read SMILES from in a CSV input (default: 'smiles').")
    return p


def _read_smiles_from_file(path: Path, smiles_col: str) -> list[str]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if smiles_col not in (reader.fieldnames or []):
                sys.exit(
                    f"Error: column '{smiles_col}' not found in {path}.\n"
                    f"Available columns: {reader.fieldnames}\n"
                    f"Use --smiles-col to specify the correct column name."
                )
            return [row[smiles_col] for row in reader if row[smiles_col].strip()]
    else:
        # Plain text: one SMILES per line
        lines = path.read_text(encoding="utf-8").splitlines()
        return [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]


def _write_results(results, output: Path | None) -> None:
    rows = [r.to_dict() for r in results]
    fieldnames = ["smiles", "label", "probability", "error"]

    if output is None:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    else:
        with open(output, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Results saved to: {output}")
        _print_summary(results)


def _print_summary(results) -> None:
    total    = len(results)
    active   = sum(1 for r in results if r.label == "Active")
    inactive = sum(1 for r in results if r.label == "Inactive")
    invalid  = sum(1 for r in results if r.label == "Invalid")
    print(f"\nSummary: {total} compounds — {active} Active | {inactive} Inactive | {invalid} Invalid")


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    if args.version:
        from leishmania_screen import __version__
        print(f"leishmania-screen {__version__}")
        return

    if args.smiles is None and args.file is None:
        parser.print_help()
        sys.exit(0)

    # Lazy import so --version/--help don't pay the torch/rdkit load cost
    from leishmania_screen import predict

    if args.smiles:
        result = predict(args.smiles)
        if args.output:
            _write_results([result], args.output)
        else:
            print(f"SMILES      : {result.smiles}")
            print(f"Label       : {result.label}")
            if result.probability is not None:
                print(f"Probability : {result.probability:.4f}")
            if result.error:
                print(f"Error       : {result.error}")
        return

    # Batch mode
    if not args.file.exists():
        sys.exit(f"Error: file not found: {args.file}")

    smiles_list = _read_smiles_from_file(args.file, args.smiles_col)
    if not smiles_list:
        sys.exit("Error: no SMILES found in the input file.")

    print(f"Screening {len(smiles_list)} compounds …", file=sys.stderr)
    results = predict(smiles_list)
    _write_results(results, args.output)


if __name__ == "__main__":
    main()
