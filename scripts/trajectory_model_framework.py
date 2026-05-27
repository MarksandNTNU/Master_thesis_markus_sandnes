"""Per-trajectory VIV sensing data framework (no model training).

Pipeline per trajectory CSV:
1) Prepare sensing dataset with POD-based sensor selection
2) Save prepared arrays and POD/scaler metadata
3) Provide reusable utilities for reconstruction + metrics

Model training is intentionally excluded so it can be done in notebooks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from scripts.sensing_data_augmentation import prepare_sensing_data


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = y_true - y_pred
    return float(np.mean(np.mean(err ** 2, axis=-1)))


def mre_percent(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = y_true.reshape(-1, y_true.shape[-1])
    yp = y_pred.reshape(-1, y_pred.shape[-1])
    num = np.sqrt(np.sum((yt - yp) ** 2, axis=-1))
    den = np.sqrt(np.sum(yt ** 2, axis=-1)) + 1e-8
    return float(np.mean(num / den) * 100.0)


def r2_score_global(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2) + 1e-12
    return float(1.0 - ss_res / ss_tot)


def reconstruct_full_field(y_pred_scaled: np.ndarray, dataset: Dict[str, Any]) -> np.ndarray:
    phi = dataset["pod_basis"].astype(np.float32)
    mean = dataset["train_mean"].astype(np.float32)
    scaler = dataset["scaler"]

    nx = phi.shape[0]
    shape_prefix = y_pred_scaled.shape[:-1]

    pod_unscaled = scaler.inverse_transform(y_pred_scaled.reshape(-1, y_pred_scaled.shape[-1]))
    y_rec = pod_unscaled @ phi.T + mean
    y_rec = y_rec.reshape(*shape_prefix, nx)
    return y_rec.astype(np.float32)


def evaluate_predictions(
    y_pred_scaled: np.ndarray,
    y_truth: np.ndarray,
    dataset: Dict[str, Any],
) -> Dict[str, float]:
    """Evaluate scaled POD predictions against full-field truth.

    Args:
        y_pred_scaled: Predicted scaled POD coefficients with the same shape as dataset split Y.
        y_truth: Full-field truth from dataset split Y_truth.
        dataset: Prepared trajectory dataset dictionary from prepare_sensing_data.
    """
    y_pred_field = reconstruct_full_field(y_pred_scaled, dataset)
    return {
        "mse": mse(y_truth, y_pred_field),
        "mre_percent": mre_percent(y_truth, y_pred_field),
        "r2": r2_score_global(y_truth, y_pred_field),
    }


def evaluate_split_predictions(
    y_pred_scaled: np.ndarray,
    dataset: Dict[str, Any],
    split: str,
) -> Dict[str, float]:
    """Evaluate one split using precomputed scaled POD predictions."""
    if split not in dataset:
        raise KeyError(f"Unknown split '{split}'. Expected one of train/valid/test.")
    return evaluate_predictions(y_pred_scaled, dataset[split]["Y_truth"], dataset)


def run_one_trajectory(csv_path: Path, args: argparse.Namespace) -> Dict[str, Any]:
    dataset = prepare_sensing_data(
        csv_path=csv_path,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        n_sensors=args.n_sensors,
        n_modes=args.n_modes,
        random_state=args.seed,
        transpose=not args.no_transpose,
        seq_len=args.seq_len,
        stride=args.stride,
        energy_threshold=args.energy_threshold,
    )

    result = {
        "trajectory": csv_path.name,
        "sensor_idx": dataset["sensor_idx"].tolist(),
        "n_modes_selected": int(dataset.get("n_modes_selected", args.n_modes)),
        "reconstruction_error_train_percent": dataset["reconstruction_error"]["train"] * 100.0,
        "reconstruction_error_valid_percent": dataset["reconstruction_error"]["valid"] * 100.0,
        "reconstruction_error_test_percent": dataset["reconstruction_error"]["test"] * 100.0,
        "x_train_shape": list(dataset["train"]["X"].shape),
        "y_train_shape": list(dataset["train"]["Y"].shape),
        "y_truth_train_shape": list(dataset["train"]["Y_truth"].shape),
        "x_valid_shape": list(dataset["valid"]["X"].shape),
        "y_valid_shape": list(dataset["valid"]["Y"].shape),
        "y_truth_valid_shape": list(dataset["valid"]["Y_truth"].shape),
        "x_test_shape": list(dataset["test"]["X"].shape),
        "y_test_shape": list(dataset["test"]["Y"].shape),
        "y_truth_test_shape": list(dataset["test"]["Y_truth"].shape),
    }

    if args.save_datasets:
        dataset_dir = Path(args.output_dir) / "trajectory_datasets"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        out_path = dataset_dir / f"{csv_path.stem}.npz"
        np.savez_compressed(
            out_path,
            sensor_idx=dataset["sensor_idx"],
            pod_basis=dataset["pod_basis"],
            singular_values=dataset["singular_values"],
            train_mean=dataset["train_mean"],
            scaler_mean=dataset["scaler"].mean_.astype(np.float32),
            scaler_scale=dataset["scaler"].scale_.astype(np.float32),
            X_train=dataset["train"]["X"],
            Y_train=dataset["train"]["Y"],
            Y_train_truth=dataset["train"]["Y_truth"],
            X_valid=dataset["valid"]["X"],
            Y_valid=dataset["valid"]["Y"],
            Y_valid_truth=dataset["valid"]["Y_truth"],
            X_test=dataset["test"]["X"],
            Y_test=dataset["test"]["Y"],
            Y_test_truth=dataset["test"]["Y_truth"],
        )
        result["dataset_file"] = str(out_path)

    return result


def find_csv_files(data_dir: Path, pattern: str) -> List[Path]:
    files = sorted(data_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched pattern '{pattern}' in {data_dir}")
    return files


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare one sensing dataset per trajectory CSV (no model training).",
    )

    parser.add_argument("--data-dir", default="NDP38m_extracted_csv", help="Directory containing trajectory CSV files")
    parser.add_argument("--pattern", default="*CF.csv", help="Glob pattern for trajectories")
    parser.add_argument("--max-trajectories", type=int, default=None, help="Optional cap on number of trajectories")

    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--n-sensors", type=int, default=5)
    parser.add_argument("--n-modes", type=int, default=12)
    parser.add_argument("--energy-threshold", type=float, default=99.0)
    parser.add_argument("--seq-len", type=int, default=1000)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-transpose", action="store_true")

    parser.add_argument("--output-dir", default="processed_data")
    parser.add_argument("--save-datasets", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    csv_files = find_csv_files(data_dir, args.pattern)
    if args.max_trajectories is not None:
        csv_files = csv_files[: args.max_trajectories]

    results: List[Dict[str, Any]] = []
    for i, csv_path in enumerate(csv_files, start=1):
        print("\n" + "=" * 80)
        print(f"[{i}/{len(csv_files)}] Trajectory: {csv_path.name}")
        print("=" * 80)
        try:
            result = run_one_trajectory(csv_path, args)
            results.append(result)
            print("prepared shapes:")
            print(
                f"train X={tuple(result['x_train_shape'])}, Y={tuple(result['y_train_shape'])}, "
                f"Y_truth={tuple(result['y_truth_train_shape'])}"
            )
            print(
                f"valid X={tuple(result['x_valid_shape'])}, Y={tuple(result['y_valid_shape'])}, "
                f"Y_truth={tuple(result['y_truth_valid_shape'])}"
            )
            print(
                f"test  X={tuple(result['x_test_shape'])}, Y={tuple(result['y_test_shape'])}, "
                f"Y_truth={tuple(result['y_truth_test_shape'])}"
            )
        except Exception as exc:
            print(f"FAILED for {csv_path.name}: {exc}")
            results.append(
                {
                    "trajectory": csv_path.name,
                    "failed": True,
                    "error": str(exc),
                }
            )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)
    csv_out = out_dir / "trajectory_framework_summary.csv"
    json_out = out_dir / "trajectory_framework_summary.json"
    results_df.to_csv(csv_out, index=False)
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nSaved summary:")
    print(csv_out)
    print(json_out)


if __name__ == "__main__":
    main()
