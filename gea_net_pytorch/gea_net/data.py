from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset


class DatasetFormatError(RuntimeError):
    pass


@dataclass(frozen=True)
class Recording:
    path: Path
    label: str
    condition: str
    group: str


def canonical(text: str) -> str:
    text = text.strip().lower().replace("+", "_plus_")
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def _condition_from_seu_stem(stem: str) -> tuple[str, str]:
    match = re.search(r"_(200-2400-200|1200|1800|2400)$", stem, flags=re.IGNORECASE)
    if not match:
        raise DatasetFormatError(f"Cannot parse SEU condition from filename: {stem}")
    condition = match.group(1)
    label_part = stem[: match.start()]
    return label_part, condition


def discover_seu(root: Path) -> list[Recording]:
    files = sorted(root.rglob("*.csv"))
    recordings: list[Recording] = []
    for path in files:
        try:
            label_part, condition = _condition_from_seu_stem(path.stem)
        except DatasetFormatError:
            continue
        parent = canonical(path.parent.name)
        if "bearing" in parent:
            group = "bearing"
        elif "parallel" in parent or "gearbox" in parent:
            group = "gearbox"
        elif "mixed" in parent:
            group = "mixed"
        else:
            group = parent or "seu"
        # Group-scoped labels preserve the 16 states in the local release,
        # including the two distinct "normal" experiments.
        label = f"{group}::{canonical(label_part)}"
        recordings.append(Recording(path.resolve(), label, condition, group))
    return recordings


_MCC_FAULT_ALIASES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(health|healthy|normal)", re.I), "health"),
    (re.compile(r"(gear[_ -]?pitting|pitting)", re.I), "pitting"),
    (re.compile(r"(gear[_ -]?wear|surface[_ -]?wear|wear)", re.I), "wear"),
    (re.compile(r"(root[_ -]?crack|teeth?[_ -]?crack|gear[_ -]?crack)", re.I), "root_crack"),
    (re.compile(r"(missing[_ -]?tooth|tooth[_ -]?missing)", re.I), "missing_tooth"),
    (re.compile(r"(broken[_ -]?tooth|teeth?[_ -]?break|gear[_ -]?tooth[_ -]?break)", re.I), "broken_tooth"),
    (re.compile(r"(inner[_ -]?race|bearing[_ -]?inner)", re.I), "bearing_inner_race"),
    (re.compile(r"(outer[_ -]?race|bearing[_ -]?outer)", re.I), "bearing_outer_race"),
]


def _parse_mcc_label(stem: str) -> str:
    for pattern, label in _MCC_FAULT_ALIASES:
        if pattern.search(stem):
            return label
    # Fallback for future/extended MCC releases: remove common condition and
    # replicate suffixes while retaining the fault descriptor.
    cleaned = re.sub(r"(?i)_(speed|torque|load|steady|circulation).*$", "", stem)
    cleaned = re.sub(r"(?i)_(\d+\s*nm|\d+\s*rpm|L|M|H|level\d+|\d+)$", "", cleaned)
    return canonical(cleaned)


def _parse_mcc_condition(path: Path) -> str:
    stem = path.stem
    torque = re.findall(r"(?i)(\d+(?:\.\d+)?)\s*nm", stem)
    speed = re.findall(r"(?i)(\d+(?:\.\d+)?)\s*rpm", stem)
    mode = re.search(r"(?i)_(speed|torque|load|steady|circulation)_", stem)
    parts: list[str] = []
    if mode:
        parts.append(mode.group(1).lower())
    if torque:
        parts.append(f"{torque[-1]}Nm")
    if speed:
        parts.append(f"{speed[-1]}rpm")
    if parts:
        return "-".join(parts)
    return canonical(path.parent.name) or "unknown"


def discover_mcc5_thu(root: Path) -> list[Recording]:
    recordings = []
    for path in sorted(root.rglob("*.csv")):
        label = _parse_mcc_label(path.stem)
        condition = _parse_mcc_condition(path)
        recordings.append(Recording(path.resolve(), label, condition, "mcc5_thu"))
    return recordings


def discover_recordings(dataset_name: str, root: Path) -> list[Recording]:
    name = canonical(dataset_name)
    if name in {"seu", "mixed_fault_dataset"}:
        recordings = discover_seu(root)
    elif name in {"mcc5_thu", "mcc5thu", "mcc_thu"}:
        recordings = discover_mcc5_thu(root)
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")
    if not recordings:
        if name.startswith("mcc"):
            raise DatasetFormatError(
                "No MCC5-THU CSV data files were found. The current local folder "
                "contains only README.md and two MATLAB plotting scripts. Download "
                "the official Version 2 data and place/extract its CSV files anywhere "
                f"under: {root}"
            )
        raise DatasetFormatError(f"No compatible CSV files found under: {root}")
    return recordings


def _sniff_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        first_line = handle.readline()
    try:
        return csv.Sniffer().sniff(first_line, delimiters=",;\t").delimiter
    except csv.Error:
        return ";" if first_line.count(";") > first_line.count(",") else ","


def _normalized_columns(columns: Iterable[Any]) -> dict[str, str]:
    return {str(column): canonical(str(column)) for column in columns}


def _select_columns(
    frame: pd.DataFrame,
    dataset_name: str,
    requested_channels: Sequence[str] | None,
) -> tuple[list[str], str | None]:
    normalized = _normalized_columns(frame.columns)
    time_column = next(
        (column for column, name in normalized.items() if name.startswith("time")),
        None,
    )
    all_nan = {column for column in frame.columns if frame[column].isna().all()}
    candidates = [
        column
        for column in frame.columns
        if column not in all_nan and column != time_column
    ]

    if requested_channels:
        lookup = {name: column for column, name in normalized.items()}
        missing = [name for name in requested_channels if canonical(name) not in lookup]
        if missing:
            raise DatasetFormatError(
                f"Requested channels {missing} are absent. Available: {list(frame.columns)}"
            )
        selected = [lookup[canonical(name)] for name in requested_channels]
        return selected, time_column

    dataset_name = canonical(dataset_name)
    if dataset_name in {"seu", "mixed_fault_dataset"}:
        selected = [
            column
            for column in candidates
            if normalized[column].startswith("ch")
        ]
    else:
        selected = [
            column
            for column in candidates
            if normalized[column] not in {"speed", "torque"}
            and "speed" not in normalized[column]
            and "torque" not in normalized[column]
        ]
    if not selected:
        selected = candidates
    if not selected:
        raise DatasetFormatError(f"No signal columns found. Columns: {list(frame.columns)}")
    return selected, time_column


def _read_recording(
    recording: Recording,
    dataset_name: str,
    requested_channels: Sequence[str] | None,
) -> tuple[np.ndarray, list[str], float | None]:
    delimiter = _sniff_delimiter(recording.path)
    frame = pd.read_csv(
        recording.path,
        sep=delimiter,
        engine="c",
        low_memory=False,
    )
    selected, time_column = _select_columns(frame, dataset_name, requested_channels)
    signals = frame[selected].apply(pd.to_numeric, errors="coerce")
    signals = signals.interpolate(axis=0, limit_direction="both").fillna(0.0)
    values = signals.to_numpy(dtype=np.float32, copy=True)

    inferred_rate = None
    if time_column is not None:
        time_values = pd.to_numeric(frame[time_column], errors="coerce").to_numpy()
        differences = np.diff(time_values[np.isfinite(time_values)])
        differences = differences[differences > 0]
        if differences.size:
            inferred_rate = float(1.0 / np.median(differences))
    return values, [str(item) for item in selected], inferred_rate


def _window_signal(
    signal: np.ndarray,
    window_size: int,
    stride: int,
    normalization: str,
    max_windows: int | None,
) -> np.ndarray:
    if signal.shape[0] < window_size:
        raise DatasetFormatError(
            f"Recording has {signal.shape[0]} rows, shorter than window_size={window_size}"
        )
    if normalization == "per_recording":
        mean = signal.mean(axis=0, keepdims=True)
        std = signal.std(axis=0, keepdims=True)
        signal = (signal - mean) / np.maximum(std, 1e-6)
    view = np.lib.stride_tricks.sliding_window_view(
        signal,
        window_shape=window_size,
        axis=0,
    )
    windows = view[::stride]  # [windows, sensors, time]
    if max_windows is not None and len(windows) > max_windows:
        selected = np.linspace(0, len(windows) - 1, max_windows, dtype=np.int64)
        windows = windows[selected]
    windows = np.asarray(windows, dtype=np.float32)
    if normalization == "per_window":
        mean = windows.mean(axis=-1, keepdims=True)
        std = windows.std(axis=-1, keepdims=True)
        windows = (windows - mean) / np.maximum(std, 1e-6)
    elif normalization not in {"none", "per_recording"}:
        raise ValueError(f"Unknown normalization mode: {normalization}")
    return np.ascontiguousarray(windows, dtype=np.float32)


def _filter_recordings(recordings: list[Recording], config: dict[str, Any]) -> list[Recording]:
    include_variable = bool(config.get("include_variable_condition", False))
    if not include_variable:
        recordings = [
            item
            for item in recordings
            if canonical(item.condition) not in {"200_2400_200", "variable"}
        ]

    include_conditions = config.get("include_conditions")
    if include_conditions:
        allowed = {canonical(str(item)) for item in include_conditions}
        recordings = [
            item for item in recordings if canonical(item.condition) in allowed
        ]

    include_labels = config.get("include_labels")
    if include_labels:
        allowed = {canonical(str(item)) for item in include_labels}
        recordings = [
            item
            for item in recordings
            if canonical(item.label) in allowed
            or canonical(item.label.split("::")[-1]) in allowed
        ]

    max_classes = config.get("max_classes")
    if max_classes is not None:
        labels = sorted({item.label for item in recordings})[: int(max_classes)]
        recordings = [item for item in recordings if item.label in labels]

    max_per_class_condition = config.get("max_recordings_per_class_condition")
    if max_per_class_condition is not None:
        grouped: dict[tuple[str, str], list[Recording]] = defaultdict(list)
        for item in recordings:
            grouped[(item.label, item.condition)].append(item)
        recordings = [
            item
            for key in sorted(grouped)
            for item in grouped[key][: int(max_per_class_condition)]
        ]
    return recordings


def _dataset_signature(
    recordings: Sequence[Recording],
    dataset_config: dict[str, Any],
) -> str:
    relevant = {
        "dataset": dataset_config.get("name"),
        "window_size": dataset_config.get("window_size"),
        "stride": dataset_config.get("stride"),
        "normalization": dataset_config.get("normalization"),
        "channel_names": dataset_config.get("channel_names"),
        "max_windows_per_recording": dataset_config.get("max_windows_per_recording"),
        "files": [
            {
                "path": str(item.path),
                "size": item.path.stat().st_size,
                "mtime_ns": item.path.stat().st_mtime_ns,
                "label": item.label,
                "condition": item.condition,
            }
            for item in recordings
        ],
    }
    payload = json.dumps(relevant, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def prepare_dataset(dataset_config: dict[str, Any]) -> dict[str, Any]:
    dataset_name = str(dataset_config["name"])
    root = Path(dataset_config["root"]).expanduser().resolve()
    if not root.exists():
        raise DatasetFormatError(f"Dataset root does not exist: {root}")

    recordings = _filter_recordings(
        discover_recordings(dataset_name, root),
        dataset_config,
    )
    if not recordings:
        raise DatasetFormatError("All recordings were removed by dataset filters")

    labels = sorted({item.label for item in recordings})
    conditions = sorted({item.condition for item in recordings}, key=canonical)
    if len(labels) < 2:
        raise DatasetFormatError(f"At least two classes are required, found: {labels}")
    label_to_index = {name: index for index, name in enumerate(labels)}
    condition_to_index = {name: index for index, name in enumerate(conditions)}

    cache_base = Path(dataset_config["cache_dir"]).expanduser().resolve()
    cache_dir = cache_base / _dataset_signature(recordings, dataset_config)
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    cache_dir.mkdir(parents=True, exist_ok=True)
    recordings_dir = cache_dir / "recordings"
    recordings_dir.mkdir(parents=True, exist_ok=True)
    window_size = int(dataset_config.get("window_size", 1024))
    stride = int(dataset_config.get("stride", window_size))
    normalization = str(dataset_config.get("normalization", "per_window"))
    max_windows = dataset_config.get("max_windows_per_recording")
    max_windows = None if max_windows is None else int(max_windows)
    requested_channels = dataset_config.get("channel_names")

    entries: list[dict[str, Any]] = []
    expected_sensors = None
    reference_channels = None
    inferred_rates = []
    for index, recording in enumerate(recordings):
        signal, channel_names, inferred_rate = _read_recording(
            recording,
            dataset_name,
            requested_channels,
        )
        if expected_sensors is None:
            expected_sensors = signal.shape[1]
            reference_channels = channel_names
        elif signal.shape[1] != expected_sensors:
            raise DatasetFormatError(
                f"Inconsistent sensor count in {recording.path}: "
                f"{signal.shape[1]} versus expected {expected_sensors}"
            )
        windows = _window_signal(
            signal,
            window_size=window_size,
            stride=stride,
            normalization=normalization,
            max_windows=max_windows,
        )
        filename = f"{index:04d}_{hashlib.sha1(str(recording.path).encode()).hexdigest()[:10]}.npy"
        cached_path = recordings_dir / filename
        np.save(cached_path, windows, allow_pickle=False)
        if inferred_rate is not None:
            inferred_rates.append(inferred_rate)
        entries.append(
            {
                "recording_id": index,
                "source_path": str(recording.path),
                "cache_path": str(cached_path),
                "label": recording.label,
                "label_index": label_to_index[recording.label],
                "condition": recording.condition,
                "condition_index": condition_to_index[recording.condition],
                "group": recording.group,
                "num_windows": int(windows.shape[0]),
                "channel_names": channel_names,
                "inferred_sampling_rate": inferred_rate,
            }
        )

    manifest = {
        "dataset_name": dataset_name,
        "dataset_root": str(root),
        "cache_dir": str(cache_dir),
        "window_size": window_size,
        "stride": stride,
        "normalization": normalization,
        "num_sensors": expected_sensors,
        "channel_names": reference_channels,
        "class_names": labels,
        "condition_names": conditions,
        "label_to_index": label_to_index,
        "condition_to_index": condition_to_index,
        "inferred_sampling_rate_median": (
            float(np.median(inferred_rates)) if inferred_rates else None
        ),
        "recordings": entries,
        "total_windows": int(sum(item["num_windows"] for item in entries)),
    }
    temporary_path = manifest_path.with_suffix(".json.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    temporary_path.replace(manifest_path)
    return manifest


def resolve_condition_split(
    manifest: dict[str, Any],
    dataset_config: dict[str, Any],
) -> tuple[list[str], list[str]]:
    conditions = list(manifest["condition_names"])
    requested = dataset_config.get("target_conditions", None)
    if requested is None:
        target_fraction = float(dataset_config.get("target_fraction", 1.0 / 3.0))
        target_count = max(1, int(math.ceil(len(conditions) * target_fraction)))
        target = conditions[-target_count:]
    else:
        lookup = {canonical(name): name for name in conditions}
        missing = [
            str(name) for name in requested if canonical(str(name)) not in lookup
        ]
        if missing:
            raise DatasetFormatError(
                f"Target conditions {missing} not found. Available: {conditions}"
            )
        target = [lookup[canonical(str(name))] for name in requested]
    source = [name for name in conditions if name not in target]
    if not source:
        raise DatasetFormatError(
            "No source condition remains. Use an empty target_conditions list for "
            "ordinary within-condition supervised training."
        )
    return source, target


def _split_bounds(num_windows: int, fractions: Sequence[float]) -> dict[str, tuple[int, int]]:
    if len(fractions) != 3 or not np.isclose(sum(fractions), 1.0):
        raise ValueError("split_fractions must contain three values summing to one")
    if num_windows < 3:
        return {
            "train": (0, max(1, num_windows - 2)),
            "val": (max(1, num_windows - 2), max(1, num_windows - 1)),
            "test": (max(1, num_windows - 1), num_windows),
        }
    train_end = max(1, int(num_windows * fractions[0]))
    val_end = max(train_end + 1, int(num_windows * (fractions[0] + fractions[1])))
    val_end = min(val_end, num_windows - 1)
    return {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, num_windows),
    }


class CachedWindowDataset(Dataset[tuple[Tensor, int, int]]):
    def __init__(
        self,
        manifest: dict[str, Any],
        split: str,
        conditions: Sequence[str] | None = None,
        labels: Sequence[str] | None = None,
        split_fractions: Sequence[float] = (0.7, 0.15, 0.15),
        max_open_recordings: int = 8,
    ) -> None:
        if split not in {"train", "val", "test", "all"}:
            raise ValueError(f"Unknown split: {split}")
        condition_set = set(conditions or manifest["condition_names"])
        label_set = set(labels or manifest["class_names"])
        self.manifest = manifest
        self.split = split
        self.max_open_recordings = max_open_recordings
        self._arrays: OrderedDict[int, np.ndarray] = OrderedDict()
        self.entries = {
            int(item["recording_id"]): item
            for item in manifest["recordings"]
            if item["condition"] in condition_set and item["label"] in label_set
        }
        self.samples: list[tuple[int, int, int, int]] = []
        for recording_id, entry in self.entries.items():
            if split == "all":
                start, end = 0, int(entry["num_windows"])
            else:
                start, end = _split_bounds(
                    int(entry["num_windows"]),
                    split_fractions,
                )[split]
            for window_index in range(start, end):
                self.samples.append(
                    (
                        recording_id,
                        window_index,
                        int(entry["label_index"]),
                        int(entry["condition_index"]),
                    )
                )
        self.class_to_indices: dict[int, list[int]] = defaultdict(list)
        for dataset_index, sample in enumerate(self.samples):
            self.class_to_indices[sample[2]].append(dataset_index)

    def __len__(self) -> int:
        return len(self.samples)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_arrays"] = OrderedDict()
        return state

    def close(self) -> None:
        """Close memory maps explicitly (important when deleting caches on Windows)."""
        while self._arrays:
            _, array = self._arrays.popitem(last=False)
            memory_map = getattr(array, "_mmap", None)
            if memory_map is not None:
                memory_map.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _array(self, recording_id: int) -> np.ndarray:
        if recording_id in self._arrays:
            array = self._arrays.pop(recording_id)
            self._arrays[recording_id] = array
            return array
        path = self.entries[recording_id]["cache_path"]
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        self._arrays[recording_id] = array
        while len(self._arrays) > self.max_open_recordings:
            _, old_array = self._arrays.popitem(last=False)
            memory_map = getattr(old_array, "_mmap", None)
            if memory_map is not None:
                memory_map.close()
        return array

    def __getitem__(self, index: int) -> tuple[Tensor, int, int]:
        recording_id, window_index, label, condition = self.samples[index]
        # Copy avoids exposing a read-only memmap to PyTorch.
        window = np.array(self._array(recording_id)[window_index], copy=True)
        return torch.from_numpy(window), label, condition


@dataclass
class Episode:
    support_x: Tensor
    support_y: Tensor
    query_x: Tensor
    query_y: Tensor
    global_classes: Tensor


class EpisodeGenerator:
    def __init__(
        self,
        support_dataset: CachedWindowDataset,
        ways: int,
        shots: int,
        queries: int,
        seed: int,
        query_dataset: CachedWindowDataset | None = None,
    ) -> None:
        self.support_dataset = support_dataset
        self.query_dataset = query_dataset or support_dataset
        self.ways = ways
        self.shots = shots
        self.queries = queries
        self.rng = np.random.default_rng(seed)
        support_classes = {
            label
            for label, indices in support_dataset.class_to_indices.items()
            if len(indices) >= shots
        }
        query_classes = {
            label
            for label, indices in self.query_dataset.class_to_indices.items()
            if len(indices) >= queries
        }
        if self.query_dataset is self.support_dataset:
            eligible = {
                label
                for label, indices in support_dataset.class_to_indices.items()
                if len(indices) >= shots + queries
            }
        else:
            eligible = support_classes & query_classes
        self.eligible_classes = sorted(eligible)
        if len(self.eligible_classes) < ways:
            raise DatasetFormatError(
                f"Only {len(self.eligible_classes)} episode-eligible classes are "
                f"available, but ways={ways}. Reduce ways/shots/queries or add data."
            )

    def _stack(self, dataset: CachedWindowDataset, indices: Sequence[int]) -> Tensor:
        return torch.stack([dataset[int(index)][0] for index in indices], dim=0)

    def sample(self) -> Episode:
        chosen = self.rng.choice(self.eligible_classes, size=self.ways, replace=False)
        support_indices: list[int] = []
        query_indices: list[int] = []
        support_labels: list[int] = []
        query_labels: list[int] = []
        for local_label, global_label in enumerate(chosen.tolist()):
            support_pool = self.support_dataset.class_to_indices[global_label]
            if self.query_dataset is self.support_dataset:
                selected = self.rng.choice(
                    support_pool,
                    size=self.shots + self.queries,
                    replace=False,
                )
                support_selected = selected[: self.shots]
                query_selected = selected[self.shots :]
            else:
                query_pool = self.query_dataset.class_to_indices[global_label]
                support_selected = self.rng.choice(
                    support_pool,
                    size=self.shots,
                    replace=False,
                )
                query_selected = self.rng.choice(
                    query_pool,
                    size=self.queries,
                    replace=False,
                )
            support_indices.extend(int(item) for item in support_selected)
            query_indices.extend(int(item) for item in query_selected)
            support_labels.extend([local_label] * self.shots)
            query_labels.extend([local_label] * self.queries)
        return Episode(
            support_x=self._stack(self.support_dataset, support_indices),
            support_y=torch.tensor(support_labels, dtype=torch.long),
            query_x=self._stack(self.query_dataset, query_indices),
            query_y=torch.tensor(query_labels, dtype=torch.long),
            global_classes=torch.tensor(chosen, dtype=torch.long),
        )


def estimate_adjacency_prior(
    dataset: CachedWindowDataset,
    threshold: float = 0.3,
    top_k: int = 2,
    max_samples: int = 256,
    physical_edges: Sequence[Sequence[int]] | None = None,
    seed: int = 42,
) -> np.ndarray:
    if len(dataset) == 0:
        raise DatasetFormatError("Cannot estimate graph prior from an empty dataset")
    rng = np.random.default_rng(seed)
    count = min(max_samples, len(dataset))
    indices = rng.choice(len(dataset), size=count, replace=False)
    windows = np.stack([dataset[int(index)][0].numpy() for index in indices])
    num_sensors = windows.shape[1]
    flattened = windows.transpose(1, 0, 2).reshape(num_sensors, -1)
    correlation = np.nan_to_num(np.abs(np.corrcoef(flattened)), nan=0.0)
    adjacency = (correlation >= threshold).astype(np.float32)

    if top_k > 0 and num_sensors > 1:
        for sensor in range(num_sensors):
            candidates = np.argsort(correlation[sensor])[::-1]
            candidates = candidates[candidates != sensor][: min(top_k, num_sensors - 1)]
            adjacency[sensor, candidates] = 1.0
    if physical_edges:
        for edge in physical_edges:
            if len(edge) != 2:
                raise ValueError(f"Physical edge must contain two indices: {edge}")
            left, right = map(int, edge)
            if not (0 <= left < num_sensors and 0 <= right < num_sensors):
                raise ValueError(f"Physical edge out of range for {num_sensors} sensors: {edge}")
            adjacency[left, right] = adjacency[right, left] = 1.0
    adjacency = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, 1.0)
    return adjacency


def dataset_summary(
    manifest: dict[str, Any],
    source_conditions: Sequence[str],
    target_conditions: Sequence[str],
) -> dict[str, Any]:
    by_label: dict[str, int] = defaultdict(int)
    by_condition: dict[str, int] = defaultdict(int)
    for item in manifest["recordings"]:
        by_label[item["label"]] += int(item["num_windows"])
        by_condition[item["condition"]] += int(item["num_windows"])
    return {
        "dataset": manifest["dataset_name"],
        "root": manifest["dataset_root"],
        "recordings": len(manifest["recordings"]),
        "windows": manifest["total_windows"],
        "sensors": manifest["num_sensors"],
        "channels": manifest["channel_names"],
        "classes": len(manifest["class_names"]),
        "conditions": manifest["condition_names"],
        "source_conditions": list(source_conditions),
        "target_conditions": list(target_conditions),
        "inferred_sampling_rate_hz": manifest["inferred_sampling_rate_median"],
        "windows_by_class": dict(by_label),
        "windows_by_condition": dict(by_condition),
        "cache_dir": manifest["cache_dir"],
    }
