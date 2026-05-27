"""Prepare POD-based sensing datasets from one or more CSV field files.

This module builds train/valid/test splits for sensing tasks where:
- X: sparse sensor measurements at POD-optimal sensor locations (QDEIM)
- Y: POD coefficients of the full field (optionally standardized)

The POD basis and optional output normalization are fit on training data only.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import qr
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd


def _relative_reconstruction_error(
    snapshots: np.ndarray,
    train_mean: np.ndarray,
    phi: np.ndarray,
) -> float:
    centered = snapshots - train_mean
    coeffs = centered @ phi
    recon = coeffs @ phi.T
    denom = np.linalg.norm(centered)
    if denom == 0:
        return 0.0
    return float(np.linalg.norm(centered - recon) / denom)


def _select_modes_by_energy(
    train_centered: np.ndarray,
    train_field: np.ndarray,
    valid_field: np.ndarray,
    test_field: np.ndarray,
    train_mean: np.ndarray,
    min_modes: int,
    max_modes: int,
    random_state: int,
    energy_threshold: float,
) -> tuple[np.ndarray, np.ndarray, int, float, float, float]:
    """Select the minimum number of POD modes whose cumulative energy (sum of
    squared singular values) exceeds ``energy_threshold`` percent.

    The search is done in a single randomized SVD call with ``max_modes``
    components, so it is cheap regardless of the chosen threshold.
    """
    _, singular_values_all, vt_all = randomized_svd(
        train_centered,
        n_components=max_modes,
        random_state=random_state,
    )
    energy = singular_values_all ** 2
    cumulative_pct = np.cumsum(energy) / energy.sum() * 100.0

    exceeded = np.where(cumulative_pct >= energy_threshold)[0]
    n_modes_sel = int(exceeded[0] + 1) if len(exceeded) > 0 else max_modes
    n_modes_sel = max(min_modes, min(n_modes_sel, max_modes))

    phi = vt_all[:n_modes_sel].T.astype(np.float32)
    singular_values = singular_values_all[:n_modes_sel]

    err_train = _relative_reconstruction_error(train_field, train_mean, phi)
    err_valid = _relative_reconstruction_error(valid_field, train_mean, phi)
    err_test  = _relative_reconstruction_error(test_field,  train_mean, phi)

    print(
        f"Energy threshold {energy_threshold:.4g}% reached with {n_modes_sel} modes "
        f"(cumulative energy = {cumulative_pct[n_modes_sel - 1]:.4f}%)"
    )
    return phi, singular_values.astype(np.float32), n_modes_sel, err_train, err_valid, err_test


def _validate_split_ratios(train_ratio: float, valid_ratio: float) -> None:
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be in (0, 1).")
    if not (0.0 < valid_ratio < 1.0):
        raise ValueError("valid_ratio must be in (0, 1).")
    if train_ratio + valid_ratio >= 1.0:
        raise ValueError("train_ratio + valid_ratio must be < 1.0.")


def _load_csv_field(csv_path: str | Path, transpose: bool = True) -> np.ndarray:
    data = pd.read_csv(csv_path).to_numpy(dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D CSV data, got shape={data.shape}.")
    if transpose:
        data = data.T
    if data.shape[0] < 3:
        raise ValueError("Not enough timesteps after loading CSV.")
    return data


def _load_and_stack_fields(
    csv_path: str | Path | Sequence[str | Path],
    transpose: bool,
    stride: int,
) -> tuple[list[np.ndarray], list[str]]:
    if isinstance(csv_path, (str, Path)):
        csv_paths = [csv_path]
    else:
        csv_paths = list(csv_path)

    if len(csv_paths) == 0:
        raise ValueError("At least one CSV path must be provided.")

    fields = []
    source_files: list[str] = []
    expected_nx: int | None = None
    for path in csv_paths:
        field = _load_csv_field(csv_path=path, transpose=transpose)
        field = _apply_stride(field, stride=stride)

        if expected_nx is None:
            expected_nx = field.shape[1]
        elif field.shape[1] != expected_nx:
            raise ValueError(
                "All CSV files must have the same spatial dimension. "
                f"Expected nx={expected_nx}, got nx={field.shape[1]} for {path}."
            )

        fields.append(field)
        source_files.append(str(Path(path)))

    return fields, source_files


def _apply_stride(field: np.ndarray, stride: int) -> np.ndarray:
    if stride <= 0:
        raise ValueError(f"stride must be a positive integer, got {stride}.")
    return field[::stride]


def _temporal_split(
    field: np.ndarray,
    train_ratio: float,
    valid_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_t = field.shape[0]
    n_train = int(train_ratio * n_t)
    n_valid_end = int((train_ratio + valid_ratio) * n_t)

    if n_train < 1 or (n_valid_end - n_train) < 1 or (n_t - n_valid_end) < 1:
        raise ValueError(
            "Temporal split produced an empty partition. "
            f"n_t={n_t}, n_train={n_train}, n_valid={n_valid_end - n_train}, n_test={n_t - n_valid_end}."
        )

    train = field[:n_train]
    valid = field[n_train:n_valid_end]
    test = field[n_valid_end:]
    return train, valid, test


def _truncate_and_reshape_split(
    arr: np.ndarray,
    seq_len: int,
    split_name: str,
) -> np.ndarray:
    if seq_len <= 0:
        raise ValueError(f"seq_len must be a positive integer, got {seq_len}.")

    n_t, n_f = arr.shape
    while seq_len > 0:
        n_full = n_t // seq_len
        n_keep = n_full * seq_len
        if n_keep > 0:
            return arr[:n_keep].reshape(n_full, seq_len, n_f)
        seq_len //= 2

    raise ValueError(
        f"Split '{split_name}' has only {n_t} timesteps, too short to form any sequence."
    )


def _maybe_sequence_splits(
    x_train: np.ndarray,
    x_valid: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    y_test: np.ndarray,
    y_truth_train: np.ndarray,
    y_truth_valid: np.ndarray,
    y_truth_test: np.ndarray,
    seq_len: int | None,
) -> tuple[np.ndarray, ...]:
    if seq_len is None:
        return (
            x_train,
            x_valid,
            x_test,
            y_train,
            y_valid,
            y_test,
            y_truth_train,
            y_truth_valid,
            y_truth_test,
        )

    return (
        _truncate_and_reshape_split(x_train, seq_len=seq_len, split_name="train"),
        _truncate_and_reshape_split(x_valid, seq_len=seq_len, split_name="valid"),
        _truncate_and_reshape_split(x_test, seq_len=seq_len, split_name="test"),
        _truncate_and_reshape_split(y_train, seq_len=seq_len, split_name="train"),
        _truncate_and_reshape_split(y_valid, seq_len=seq_len, split_name="valid"),
        _truncate_and_reshape_split(y_test, seq_len=seq_len, split_name="test"),
        _truncate_and_reshape_split(
            y_truth_train,
            seq_len=seq_len,
            split_name="train",
        ),
        _truncate_and_reshape_split(
            y_truth_valid,
            seq_len=seq_len,
            split_name="valid",
        ),
        _truncate_and_reshape_split(
            y_truth_test,
            seq_len=seq_len,
            split_name="test",
        ),
    )


def _spatial_derivatives(
    field: np.ndarray,
    dx: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (first_derivative, second_derivative) along the last axis.

    Works for both 2-D (n_t, n_x) and 3-D (n_seq, seq_len, n_x) arrays.
    """
    d1 = np.gradient(field, dx, axis=-1).astype(np.float32)
    d2 = np.gradient(d1, dx, axis=-1).astype(np.float32)
    return d1, d2


def _pod_max_sensor_placement(phi: np.ndarray, n_sensors: int) -> np.ndarray:
    """Select sensors greedily by maximum absolute amplitude in each POD mode.

    For each mode (in order of importance), pick the spatial index with the
    largest absolute value that has not yet been chosen.  Cycle through modes
    repeatedly until ``n_sensors`` unique indices are collected.
    """
    n_x, n_modes = phi.shape
    chosen: list[int] = []
    chosen_set: set[int] = set()
    mode_idx = 0
    while len(chosen) < n_sensors:
        mode = np.abs(phi[:, mode_idx % n_modes])
        # mask already-chosen indices
        mask = np.ones(n_x, dtype=bool)
        for idx in chosen_set:
            mask[idx] = False
        candidates = np.where(mask)[0]
        if len(candidates) == 0:
            break
        best = candidates[np.argmax(mode[candidates])]
        chosen.append(int(best))
        chosen_set.add(int(best))
        mode_idx += 1
    return np.sort(np.array(chosen, dtype=np.int32))


def _build_split_arrays(
    split_parts: list[np.ndarray],
    split_name: str,
    sensor_idx: np.ndarray,
    train_mean: np.ndarray,
    phi: np.ndarray,
    scaler: StandardScaler | None,
    scale_outputs: bool,
    seq_len: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_parts = []
    y_parts = []
    y_truth_parts = []

    for part_idx, part in enumerate(split_parts):
        x = part[:, sensor_idx].astype(np.float32)
        y_unscaled = ((part - train_mean) @ phi).astype(np.float32)
        if scale_outputs:
            if scaler is None:
                raise ValueError("scale_outputs=True requires a fitted scaler.")
            y = scaler.transform(y_unscaled).astype(np.float32)
        else:
            y = y_unscaled
        y_truth = part.astype(np.float32)

        if seq_len is not None:
            name = f"{split_name}[{part_idx}]"
            x = _truncate_and_reshape_split(x, seq_len=seq_len, split_name=name)
            y = _truncate_and_reshape_split(y, seq_len=seq_len, split_name=name)
            y_truth = _truncate_and_reshape_split(y_truth, seq_len=seq_len, split_name=name)

        x_parts.append(x)
        y_parts.append(y)
        y_truth_parts.append(y_truth)

    return (
        np.concatenate(x_parts, axis=0),
        np.concatenate(y_parts, axis=0),
        np.concatenate(y_truth_parts, axis=0),
    )


def prepare_sensing_data(
    csv_path: str | Path | Sequence[str | Path],
    train_ratio: float,
    valid_ratio: float,
    n_sensors: int = 5,
    n_modes: int = 12,
    random_state: int = 42,
    transpose: bool = True,
    seq_len: int | None = None,
    stride: int = 1,
    energy_threshold: float | None = 99.0,
    sensor_idx: Sequence[int] | np.ndarray | None = None,
    sensor_placement: str = "qdeim",
    x_coords: np.ndarray | None = None,
    scale_outputs: bool = True,
) -> Dict[str, Any]:
    """Build sensing inputs/targets from one or multiple full-field CSV files.

    Args:
        csv_path: Path to one CSV or a list/tuple of CSV paths.
        train_ratio: Fraction for first temporal train split.
        valid_ratio: Fraction for following temporal validation split.
        n_sensors: Number of sensors to select automatically.  Ignored when
            ``sensor_idx`` is provided.
        n_modes: Minimum number of POD modes (also the exact number used when
            ``energy_threshold`` is None).
        random_state: Random state for randomized SVD.
        transpose: If True, transpose CSV after reading.
        seq_len: Optional sequence length. If provided, the training split is
            truncated to a multiple of seq_len and reshaped to
            (n_seq, seq_len, features). Valid/test are kept as single full
            sequences (1, n_t, features).
        stride: Temporal downsampling stride (stride=2 keeps every other
            timestep).
        energy_threshold: Cumulative POD energy percentage target (0–100).
            The minimum number of modes whose cumulative squared-singular-value
            energy exceeds this value is selected, clamped to
            [n_modes, max_modes]. Set to None to use exactly ``n_modes``.
            Default is 99.0 (99 %).
        sensor_idx: Optional explicit sensor indices (0-based, within
            ``[0, Nx)``) to use instead of automatic placement.  When supplied,
            ``n_sensors`` and ``sensor_placement`` are ignored.
        sensor_placement: Algorithm used when ``sensor_idx`` is None.
            ``"qdeim"`` (default) — QR-pivoted QDEIM on the POD basis.
            ``"pod_max"`` — greedily pick the index of maximum absolute
            amplitude from each POD mode in turn.
        x_coords: Optional 1-D array of spatial coordinates (length n_x).
            Used to compute the correct ``dx`` for spatial derivatives.
            If None, unit spacing (dx=1) is assumed.
        scale_outputs: If True (default), standardize POD coefficients Y with
            a training-fit StandardScaler. If False, keep Y unscaled.

    Returns:
        Dictionary containing train/valid/test X and Y, plus POD/scaler metadata.
        Each split sub-dict contains:
            ``X``       — sensor measurements
            ``Y``       — POD coefficients (scaled or unscaled)
            ``Y_truth`` — full reconstructed field
            ``dY_dx``   — first spatial derivative of the field
            ``d2Y_dx2`` — second spatial derivative (curvature)
    """
    _validate_split_ratios(train_ratio, valid_ratio)

    if energy_threshold is not None and not (0.0 < energy_threshold <= 100.0):
        raise ValueError("energy_threshold must be in (0, 100].")

    fields, source_files = _load_and_stack_fields(
        csv_path=csv_path,
        transpose=transpose,
        stride=stride,
    )

    # Split each trajectory independently, then concatenate each split.
    train_parts = []
    valid_parts = []
    test_parts = []
    for field in fields:
        tr, vl, te = _temporal_split(
            field=field,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
        )
        train_parts.append(tr)
        valid_parts.append(vl)
        test_parts.append(te)

    train_field = np.concatenate(train_parts, axis=0).astype(np.float32)
    valid_field = np.concatenate(valid_parts, axis=0).astype(np.float32)
    test_field = np.concatenate(test_parts, axis=0).astype(np.float32)

    n_x = train_field.shape[1]

    if not (1 <= n_sensors <= n_x):
        raise ValueError(f"n_sensors must be in [1, {n_x}], got {n_sensors}.")

    max_modes = min(train_field.shape[0] - 1, n_x)
    if not (1 <= n_modes <= max_modes):
        raise ValueError(f"n_modes must be in [1, {max_modes}], got {n_modes}.")

    train_mean = train_field.mean(axis=0, keepdims=True)
    train_centered = train_field - train_mean

    if energy_threshold is None:
        _, singular_values, vt = randomized_svd(
            train_centered,
            n_components=n_modes,
            random_state=random_state,
        )
        phi = vt.T.astype(np.float32)
        n_modes_selected = n_modes
        recon_err_train = _relative_reconstruction_error(train_field, train_mean, phi)
        recon_err_valid = _relative_reconstruction_error(valid_field, train_mean, phi)
        recon_err_test  = _relative_reconstruction_error(test_field,  train_mean, phi)
    else:
        (
            phi,
            singular_values,
            n_modes_selected,
            recon_err_train,
            recon_err_valid,
            recon_err_test,
        ) = _select_modes_by_energy(
            train_centered=train_centered,
            train_field=train_field,
            valid_field=valid_field,
            test_field=test_field,
            train_mean=train_mean,
            min_modes=n_modes,
            max_modes=max_modes,
            random_state=random_state,
            energy_threshold=energy_threshold,
        )

    # Sensor placement: use caller-supplied indices, or an automatic method.
    if sensor_idx is not None:
        sensor_idx = np.sort(np.asarray(sensor_idx, dtype=np.int32))
        if sensor_idx.ndim != 1 or len(sensor_idx) == 0:
            raise ValueError("sensor_idx must be a non-empty 1-D array of indices.")
        if sensor_idx.min() < 0 or sensor_idx.max() >= n_x:
            raise ValueError(
                f"sensor_idx values must be in [0, {n_x - 1}], "
                f"got min={sensor_idx.min()}, max={sensor_idx.max()}."
            )
    elif sensor_placement == "pod_max":
        sensor_idx = _pod_max_sensor_placement(phi, n_sensors)
        print(f"POD-max sensor placement: {sensor_idx.tolist()}")
    else:
        if sensor_placement != "qdeim":
            raise ValueError(f"Unknown sensor_placement={sensor_placement!r}. Choose 'qdeim' or 'pod_max'.")
        _, _, pivot_idx = qr(phi @ phi.T, pivoting=True)
        sensor_idx = np.sort(pivot_idx[:n_sensors]).astype(np.int32)
        print(f"QDEIM sensor placement: {sensor_idx.tolist()}")

    y_train_unscaled = ((train_field - train_mean) @ phi).astype(np.float32)

    print(f"Selected POD modes: {n_modes_selected}")
    print(f"POD reconstruction error (train): {recon_err_train * 100:.4e} %")
    print(f"POD reconstruction error (valid): {recon_err_valid * 100:.4e} %")
    print(f"POD reconstruction error (test):  {recon_err_test * 100:.4e} %")

    scaler: StandardScaler | None = None
    if scale_outputs:
        scaler = StandardScaler().fit(y_train_unscaled)

    x_train, y_train, y_truth_train = _build_split_arrays(
        split_parts=train_parts,
        split_name="train",
        sensor_idx=sensor_idx,
        train_mean=train_mean,
        phi=phi,
        scaler=scaler,
        scale_outputs=scale_outputs,
        seq_len=seq_len,
    )
    x_valid, y_valid, y_truth_valid = _build_split_arrays(
        split_parts=valid_parts,
        split_name="valid",
        sensor_idx=sensor_idx,
        train_mean=train_mean,
        phi=phi,
        scaler=scaler,
        scale_outputs=scale_outputs,
        seq_len=seq_len,
    )
    x_test, y_test, y_truth_test = _build_split_arrays(
        split_parts=test_parts,
        split_name="test",
        sensor_idx=sensor_idx,
        train_mean=train_mean,
        phi=phi,
        scaler=scaler,
        scale_outputs=scale_outputs,
        seq_len=seq_len,
    )

    # Build index maps: for each split, record which sequence indices
    # belong to which source CSV file.
    def _compute_source_index_map(parts, effective_seq_len):
        index_map = {}
        cursor = 0
        for i, part in enumerate(parts):
            if effective_seq_len is not None:
                sl = effective_seq_len
                while sl > 0 and part.shape[0] // sl == 0:
                    sl //= 2
                n_seqs = part.shape[0] // sl if sl > 0 else 0
            else:
                n_seqs = part.shape[0]
            index_map[source_files[i]] = list(range(cursor, cursor + n_seqs))
            cursor += n_seqs
        return index_map

    source_index = {
        "train": _compute_source_index_map(train_parts, seq_len),
        "valid": _compute_source_index_map(valid_parts, seq_len),
        "test":  _compute_source_index_map(test_parts,  seq_len),
    }

    print(
        "X shapes:",
        {
            "train": x_train.shape,
            "valid": x_valid.shape,
            "test": x_test.shape,
        },
    )
    print(
        "Y shapes:",
        {
            "train": y_train.shape,
            "valid": y_valid.shape,
            "test": y_test.shape,
        },
    )

    
    dx = float(np.mean(np.diff(np.asarray(x_coords, dtype=np.float64))))


    dy_train,  d2y_train  = _spatial_derivatives(y_truth_train,  dx)
    dy_valid,  d2y_valid  = _spatial_derivatives(y_truth_valid,  dx)
    dy_test,   d2y_test   = _spatial_derivatives(y_truth_test,   dx)

    # Spatial derivatives of the POD basis modes (n_x, n_modes) along axis=0.
    dphi_dx  = np.gradient(phi,  dx, axis=0).astype(np.float32)
    d2phi_dx2 = np.gradient(dphi_dx, dx, axis=0).astype(np.float32)

    return {
        "source_files": source_files,
        "n_source_files": len(source_files),
        "sensor_idx": sensor_idx,
        "pod_basis": phi,
        "pod_basis_d1": dphi_dx,
        "pod_basis_d2": d2phi_dx2,
        "singular_values": singular_values.astype(np.float32),
        "n_modes_selected": n_modes_selected,
        "train_mean": train_mean.astype(np.float32),
        "stride": stride,
        "seq_len": seq_len,
        "energy_threshold": energy_threshold,
        "scale_outputs": scale_outputs,
        "x_coords": np.asarray(x_coords, dtype=np.float32) if x_coords is not None else None,
        "dx": float(dx),
        "reconstruction_error": {
            "train": recon_err_train,
            "valid": recon_err_valid,
            "test": recon_err_test,
        },
        "scaler": scaler,
        "source_index": source_index,
        # Raw split parts kept so sensor extraction can be redone cheaply
        # without re-reading CSVs or re-running POD.  Shape: list of (n_t, n_x).
        "raw_parts": {"train": train_parts, "valid": valid_parts, "test": test_parts},
        "train": {"X": x_train, "Y": y_train, "Y_truth": y_truth_train,
                  "dY_dx": dy_train, "d2Y_dx2": d2y_train},
        "valid": {"X": x_valid, "Y": y_valid, "Y_truth": y_truth_valid,
                  "dY_dx": dy_valid, "d2Y_dx2": d2y_valid},
        "test":  {"X": x_test,  "Y": y_test,  "Y_truth": y_truth_test,
                  "dY_dx": dy_test,  "d2Y_dx2": d2y_test},
    }


def rebuild_sensor_inputs(
    dataset: Dict[str, Any],
    sensor_idx: Sequence[int] | np.ndarray | None = None,
    n_sensors: int | None = None,
) -> Dict[str, Any]:
    """Re-extract sensor inputs from stored raw parts without re-loading CSVs or re-running POD.

    Provide exactly one of ``sensor_idx`` (explicit indices) or ``n_sensors``
    (QDEIM placement on the existing POD basis).  Returns a shallow copy of
    ``dataset`` with updated ``sensor_idx`` and ``train``/``valid``/``test`` X arrays.

    Args:
        dataset:    Dictionary returned by ``prepare_sensing_data``.
        sensor_idx: Explicit 0-based sensor indices to use.
        n_sensors:  Number of QDEIM sensors to select from the existing POD basis.

    Returns:
        Updated dataset dict (shares Y / Y_truth / POD arrays with the original).
    """
    if (sensor_idx is None) == (n_sensors is None):
        raise ValueError("Provide exactly one of sensor_idx or n_sensors.")

    phi        = dataset["pod_basis"]
    train_mean = dataset["train_mean"]
    seq_len    = dataset["seq_len"]
    raw_parts  = dataset["raw_parts"]
    n_x        = phi.shape[0]

    if sensor_idx is not None:
        sidx = np.sort(np.asarray(sensor_idx, dtype=np.int32))
        if sidx.ndim != 1 or len(sidx) == 0:
            raise ValueError("sensor_idx must be a non-empty 1-D array of indices.")
        if sidx.min() < 0 or sidx.max() >= n_x:
            raise ValueError(
                f"sensor_idx values must be in [0, {n_x - 1}], "
                f"got min={sidx.min()}, max={sidx.max()}."
            )
    else:
        _, _, pivot_idx = qr(phi @ phi.T, pivoting=True)
        sidx = np.sort(pivot_idx[:n_sensors]).astype(np.int32)

    def _rebuild_x(parts, split_name):
        x_parts = []
        for part_idx, part in enumerate(parts):
            x = part[:, sidx].astype(np.float32)
            if seq_len is not None:
                x = _truncate_and_reshape_split(x, seq_len=seq_len,
                                                split_name=f"{split_name}[{part_idx}]")
            x_parts.append(x)
        return np.concatenate(x_parts, axis=0)

    new_dataset = dict(dataset)   # shallow copy — Y/Y_truth/POD arrays are shared
    new_dataset["sensor_idx"] = sidx
    new_dataset["train"] = dict(dataset["train"], X=_rebuild_x(raw_parts["train"], "train"))
    new_dataset["valid"] = dict(dataset["valid"], X=_rebuild_x(raw_parts["valid"], "valid"))
    new_dataset["test"]  = dict(dataset["test"],  X=_rebuild_x(raw_parts["test"],  "test"))
    return new_dataset


def save_dataset_npz(dataset: Dict[str, Any], output_path: str | Path) -> None:
    """Persist prepared arrays and scaler parameters to an .npz file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scaler: StandardScaler | None = dataset["scaler"]
    y_dim = int(dataset["pod_basis"].shape[1])
    scaler_mean = (
        scaler.mean_.astype(np.float32)
        if scaler is not None
        else np.zeros((y_dim,), dtype=np.float32)
    )
    scaler_scale = (
        scaler.scale_.astype(np.float32)
        if scaler is not None
        else np.ones((y_dim,), dtype=np.float32)
    )
    np.savez_compressed(
        output_path,
        sensor_idx=dataset["sensor_idx"],
        pod_basis=dataset["pod_basis"],
        singular_values=dataset["singular_values"],
        train_mean=dataset["train_mean"],
        scale_outputs=np.bool_(dataset.get("scale_outputs", True)),
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare POD sensing dataset from one or more CSV files.")
    parser.add_argument(
        "--csv",
        required=True,
        nargs="+",
        help="One or more input CSV file paths.",
    )
    parser.add_argument("--train-ratio", type=float, required=True, help="Train split ratio.")
    parser.add_argument("--valid-ratio", type=float, required=True, help="Validation split ratio.")
    parser.add_argument("--n-sensors", type=int, default=5, help="Number of QDEIM sensors.")
    parser.add_argument("--n-modes", type=int, default=12, help="Number of POD modes.")
    parser.add_argument(
        "--energy-threshold",
        type=float,
        default=99.0,
        help=(
            "Cumulative POD energy percentage target (0–100). "
            "The minimum number of modes whose cumulative energy exceeds this value "
            "is selected (clamped to [n_modes, max_modes]). "
            "Pass 0 or omit to use exactly --n-modes."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Random state for randomized SVD.")
    parser.add_argument(
        "--seq-len",
        "--se-len",
        dest="seq_len",
        type=int,
        default=None,
        help="Optional sequence length for train/valid/test reshaping.",
    )
    parser.add_argument(
        "--no-scale-outputs",
        action="store_true",
        help="Disable standardization of Y (POD coefficients).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Temporal sampling stride (2 means every other timestep).",
    )
    parser.add_argument(
        "--no-transpose",
        action="store_true",
        help="Use raw CSV orientation without transpose.",
    )
    parser.add_argument(
        "--output",
        default="processed_data/sensing_dataset.npz",
        help="Output .npz path.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    dataset = prepare_sensing_data(
        csv_path=args.csv,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        n_sensors=args.n_sensors,
        n_modes=args.n_modes,
        random_state=args.seed,
        transpose=not args.no_transpose,
        seq_len=args.seq_len,
        stride=args.stride,
        energy_threshold=args.energy_threshold,
        scale_outputs=not args.no_scale_outputs,
    )
    save_dataset_npz(dataset, args.output)

    print(f"Saved dataset to: {args.output}")
    print(f"Source files: {len(dataset['source_files'])}")
    print(f"Scale outputs: {dataset['scale_outputs']}")
    print(f"Sensors: {dataset['sensor_idx'].tolist()}")
    print(f"X_train: {dataset['train']['X'].shape}, Y_train: {dataset['train']['Y'].shape}")
    print(f"X_valid: {dataset['valid']['X'].shape}, Y_valid: {dataset['valid']['Y'].shape}")
    print(f"X_test:  {dataset['test']['X'].shape}, Y_test:  {dataset['test']['Y'].shape}")


if __name__ == "__main__":
    main()
