from __future__ import annotations

import json
import random
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch import Tensor, nn
from torch.func import functional_call
from torch.utils.data import DataLoader

from .data import (
    CachedWindowDataset,
    Episode,
    EpisodeGenerator,
    dataset_summary,
    estimate_adjacency_prior,
    prepare_dataset,
    resolve_condition_split,
)
from .model import GEANet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _model_from_config(config: dict[str, Any], manifest: dict[str, Any]) -> GEANet:
    model_config = config["model"]
    return GEANet(
        num_sensors=int(manifest["num_sensors"]),
        num_classes=len(manifest["class_names"]),
        input_dim=int(model_config.get("input_dim", 1)),
        embed_dim=int(model_config.get("embed_dim", 16)),
        memory_size=int(model_config.get("memory_size", 16)),
        graph_layers=int(model_config.get("graph_layers", 2)),
        graph_beta=float(model_config.get("graph_beta", 0.5)),
        temporal_kernel_size=int(model_config.get("temporal_kernel_size", 7)),
        dropout=float(model_config.get("dropout", 0.1)),
        attention_normalization_steps=int(
            model_config.get("attention_normalization_steps", 1)
        ),
    )


def prepare_experiment(
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[str], list[str], dict[str, Any]]:
    manifest = prepare_dataset(config["dataset"])
    source_conditions, target_conditions = resolve_condition_split(
        manifest,
        config["dataset"],
    )
    summary = dataset_summary(manifest, source_conditions, target_conditions)
    return manifest, source_conditions, target_conditions, summary


def build_model_and_prior(
    config: dict[str, Any],
    manifest: dict[str, Any],
    source_conditions: list[str],
    device: torch.device,
) -> tuple[GEANet, CachedWindowDataset]:
    dataset_config = config["dataset"]
    split_fractions = dataset_config.get("split_fractions", [0.7, 0.15, 0.15])
    train_dataset = CachedWindowDataset(
        manifest,
        split="train",
        conditions=source_conditions,
        split_fractions=split_fractions,
    )
    graph_config = config.get("graph", {})
    adjacency = estimate_adjacency_prior(
        train_dataset,
        threshold=float(graph_config.get("correlation_threshold", 0.3)),
        top_k=int(graph_config.get("correlation_top_k", 2)),
        max_samples=int(graph_config.get("prior_samples", 256)),
        physical_edges=graph_config.get("physical_edges"),
        seed=int(config.get("seed", 42)),
    )
    model = _model_from_config(config, manifest)
    model.set_adjacency_prior(torch.from_numpy(adjacency))
    return model.to(device), train_dataset


def _regularized_loss(
    cross_entropy: Tensor,
    auxiliary: dict[str, Tensor],
    lambda_consistency: float,
    lambda_edge: float,
) -> Tensor:
    return (
        cross_entropy
        + lambda_consistency * auxiliary["consistency"].mean()
        + lambda_edge * auxiliary["edge_sparsity"]
    )


def classification_metrics(labels: list[int], predictions: list[int]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_recall": float(recall_score(labels, predictions, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }


@torch.no_grad()
def evaluate_supervised(
    model: GEANet,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> dict[str, float]:
    model.eval()
    losses = []
    labels: list[int] = []
    predictions: list[int] = []
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        losses.append(float(criterion(logits, y).item()))
        labels.extend(y.cpu().tolist())
        predictions.extend(logits.argmax(dim=-1).cpu().tolist())
    metrics = classification_metrics(labels, predictions)
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    return metrics


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _checkpoint_payload(
    model: GEANet,
    config: dict[str, Any],
    manifest: dict[str, Any],
    source_conditions: list[str],
    target_conditions: list[str],
    mode: str,
    epoch: int,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "state_dict": model.state_dict(),
        "model_kwargs": model.model_kwargs(),
        "config": config,
        "manifest_cache_dir": manifest["cache_dir"],
        "class_names": manifest["class_names"],
        "condition_names": manifest["condition_names"],
        "source_conditions": source_conditions,
        "target_conditions": target_conditions,
        "mode": mode,
        "epoch": epoch,
        "metrics": metrics,
    }


def train_supervised(
    config: dict[str, Any],
    manifest: dict[str, Any],
    source_conditions: list[str],
    target_conditions: list[str],
    device: torch.device,
) -> dict[str, Any]:
    model, train_dataset = build_model_and_prior(
        config,
        manifest,
        source_conditions,
        device,
    )
    dataset_config = config["dataset"]
    train_config = config["training"]
    split_fractions = dataset_config.get("split_fractions", [0.7, 0.15, 0.15])
    val_dataset = CachedWindowDataset(
        manifest,
        split="val",
        conditions=source_conditions,
        split_fractions=split_fractions,
    )
    evaluation_conditions = target_conditions or source_conditions
    test_dataset = CachedWindowDataset(
        manifest,
        split="test",
        conditions=evaluation_conditions,
        split_fractions=split_fractions,
    )
    batch_size = int(train_config.get("batch_size", 256))
    workers = int(train_config.get("num_workers", 0))
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_config.get("learning_rate", 0.01)),
        weight_decay=float(train_config.get("weight_decay", 0.0)),
    )
    epochs = int(train_config.get("epochs", 50))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=float(train_config.get("minimum_learning_rate", 1e-5)),
    )
    criterion = nn.CrossEntropyLoss()
    lambda_consistency = float(train_config.get("lambda_consistency", 0.1))
    lambda_edge = float(train_config.get("lambda_edge", 0.01))
    gradient_clip = float(train_config.get("gradient_clip", 5.0))
    patience = int(train_config.get("early_stopping_patience", 10))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    best_accuracy = -1.0
    best_epoch = -1
    stale_epochs = 0
    history = []
    start_time = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        labels: list[int] = []
        predictions: list[int] = []
        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits, auxiliary = model(x, return_aux=True)
            loss = _regularized_loss(
                criterion(logits, y),
                auxiliary,
                lambda_consistency,
                lambda_edge,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            epoch_losses.append(float(loss.item()))
            labels.extend(y.detach().cpu().tolist())
            predictions.extend(logits.detach().argmax(dim=-1).cpu().tolist())
        scheduler.step()
        train_metrics = classification_metrics(labels, predictions)
        train_metrics["loss"] = float(np.mean(epoch_losses))
        val_metrics = evaluate_supervised(model, val_loader, device, criterion)
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": val_metrics,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))

        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                _checkpoint_payload(
                    model,
                    config,
                    manifest,
                    source_conditions,
                    target_conditions,
                    "supervised",
                    epoch,
                    record,
                ),
                output_dir / "best.pt",
            )
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break

    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    test_metrics = evaluate_supervised(model, test_loader, device, criterion)
    elapsed = time.perf_counter() - start_time
    summary = {
        "mode": "supervised",
        "device": str(device),
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_accuracy,
        "test_conditions": evaluation_conditions,
        "test": test_metrics,
        "elapsed_seconds": elapsed,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    torch.save(
        _checkpoint_payload(
            model,
            config,
            manifest,
            source_conditions,
            target_conditions,
            "supervised",
            best_epoch,
            summary,
        ),
        output_dir / "last.pt",
    )
    _write_json(output_dir / "history.json", history)
    _write_json(output_dir / "summary.json", summary)
    return summary


def class_prototypes(features: Tensor, labels: Tensor, ways: int) -> Tensor:
    prototypes = []
    for class_index in range(ways):
        mask = labels == class_index
        if not torch.any(mask):
            raise RuntimeError(f"Episode class {class_index} has no support sample")
        prototypes.append(features[mask].mean(dim=0))
    return torch.stack(prototypes, dim=0)


def squared_distance_logits(features: Tensor, prototypes: Tensor) -> Tensor:
    return -torch.cdist(features, prototypes, p=2).pow(2)


def proto_maml_query_loss(
    model: GEANet,
    episode: Episode,
    device: torch.device,
    inner_steps: int,
    inner_lr: float,
    first_order: bool,
    lambda_consistency: float,
    lambda_edge: float,
) -> tuple[Tensor, float]:
    support_x = episode.support_x.to(device)
    support_y = episode.support_y.to(device)
    query_x = episode.query_x.to(device)
    query_y = episode.query_y.to(device)
    ways = int(episode.global_classes.numel())

    fast_parameters = OrderedDict(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if not name.startswith("classifier.")
    )
    support_features, _ = functional_call(
        model,
        fast_parameters,
        (support_x,),
        {"return_aux": True, "features_only": True},
    )
    prototypes = class_prototypes(support_features, support_y, ways)
    local_weight = (2.0 * prototypes.detach()).requires_grad_(True)
    local_bias = (-prototypes.detach().pow(2).sum(dim=-1)).requires_grad_(True)

    for _ in range(inner_steps):
        support_features, support_aux = functional_call(
            model,
            fast_parameters,
            (support_x,),
            {"return_aux": True, "features_only": True},
        )
        support_logits = support_features @ local_weight.t() + local_bias
        support_loss = _regularized_loss(
            nn.functional.cross_entropy(support_logits, support_y),
            support_aux,
            lambda_consistency,
            lambda_edge,
        )
        parameter_values = tuple(fast_parameters.values())
        gradients = torch.autograd.grad(
            support_loss,
            parameter_values + (local_weight, local_bias),
            create_graph=not first_order,
            allow_unused=True,
        )
        updated = OrderedDict()
        for (name, parameter), gradient in zip(fast_parameters.items(), gradients):
            if gradient is None:
                updated[name] = parameter
            else:
                updated[name] = parameter - inner_lr * (
                    gradient.detach() if first_order else gradient
                )
        fast_parameters = updated
        weight_gradient, bias_gradient = gradients[-2:]
        if weight_gradient is not None:
            local_weight = local_weight - inner_lr * (
                weight_gradient.detach() if first_order else weight_gradient
            )
        if bias_gradient is not None:
            local_bias = local_bias - inner_lr * (
                bias_gradient.detach() if first_order else bias_gradient
            )

    query_features, query_aux = functional_call(
        model,
        fast_parameters,
        (query_x,),
        {"return_aux": True, "features_only": True},
    )
    query_logits = query_features @ local_weight.t() + local_bias
    query_loss = _regularized_loss(
        nn.functional.cross_entropy(query_logits, query_y),
        query_aux,
        lambda_consistency,
        lambda_edge,
    )
    accuracy = float((query_logits.argmax(dim=-1) == query_y).float().mean().item())
    return query_loss, accuracy


@torch.no_grad()
def evaluate_prototype_episodes(
    model: GEANet,
    generator: EpisodeGenerator,
    episodes: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    losses = []
    accuracies = []
    for _ in range(episodes):
        episode = generator.sample()
        support_x = episode.support_x.to(device)
        support_y = episode.support_y.to(device)
        query_x = episode.query_x.to(device)
        query_y = episode.query_y.to(device)
        support_features = model(support_x, features_only=True)
        query_features = model(query_x, features_only=True)
        prototypes = class_prototypes(
            support_features,
            support_y,
            int(episode.global_classes.numel()),
        )
        logits = squared_distance_logits(query_features, prototypes)
        losses.append(float(nn.functional.cross_entropy(logits, query_y).item()))
        accuracies.append(float((logits.argmax(dim=-1) == query_y).float().mean().item()))
    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies, ddof=1)) if len(accuracies) > 1 else 0.0,
    }


def train_fewshot(
    config: dict[str, Any],
    manifest: dict[str, Any],
    source_conditions: list[str],
    target_conditions: list[str],
    device: torch.device,
) -> dict[str, Any]:
    if not target_conditions:
        raise ValueError("Few-shot training requires at least one target condition")
    model, source_train = build_model_and_prior(
        config,
        manifest,
        source_conditions,
        device,
    )
    dataset_config = config["dataset"]
    train_config = config["training"]
    split_fractions = dataset_config.get("split_fractions", [0.7, 0.15, 0.15])
    source_val = CachedWindowDataset(
        manifest,
        split="val",
        conditions=source_conditions,
        split_fractions=split_fractions,
    )
    target_support = CachedWindowDataset(
        manifest,
        split="train",
        conditions=target_conditions,
        split_fractions=split_fractions,
    )
    target_query = CachedWindowDataset(
        manifest,
        split="test",
        conditions=target_conditions,
        split_fractions=split_fractions,
    )

    ways = int(train_config.get("ways", 5))
    shots = int(train_config.get("shots", 5))
    queries = int(train_config.get("queries", 15))
    seed = int(config.get("seed", 42))
    train_generator = EpisodeGenerator(
        source_train,
        ways=ways,
        shots=shots,
        queries=queries,
        seed=seed,
    )
    val_generator = EpisodeGenerator(
        source_train,
        query_dataset=source_val,
        ways=ways,
        shots=shots,
        queries=queries,
        seed=seed + 1,
    )
    target_generator = EpisodeGenerator(
        target_support,
        query_dataset=target_query,
        ways=ways,
        shots=shots,
        queries=queries,
        seed=seed + 2,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_config.get("learning_rate", 0.001)),
        weight_decay=float(train_config.get("weight_decay", 0.0)),
    )
    epochs = int(train_config.get("epochs", 50))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=float(train_config.get("minimum_learning_rate", 1e-5)),
    )
    episodes_per_epoch = int(train_config.get("episodes_per_epoch", 100))
    meta_batch_size = int(train_config.get("meta_batch_size", 4))
    inner_steps = int(train_config.get("inner_steps", 1))
    inner_lr = float(train_config.get("inner_learning_rate", 0.01))
    first_order = bool(train_config.get("first_order", True))
    lambda_consistency = float(train_config.get("lambda_consistency", 0.1))
    lambda_edge = float(train_config.get("lambda_edge", 0.01))
    gradient_clip = float(train_config.get("gradient_clip", 5.0))
    validation_episodes = int(train_config.get("validation_episodes", 20))
    evaluation_episodes = int(train_config.get("evaluation_episodes", 100))
    patience = int(train_config.get("early_stopping_patience", 10))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    best_accuracy = -1.0
    best_epoch = -1
    stale_epochs = 0
    history = []
    start_time = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        episode_losses = []
        episode_accuracies = []
        completed = 0
        while completed < episodes_per_epoch:
            current_batch = min(meta_batch_size, episodes_per_epoch - completed)
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for _ in range(current_batch):
                episode = train_generator.sample()
                loss, accuracy = proto_maml_query_loss(
                    model,
                    episode,
                    device=device,
                    inner_steps=inner_steps,
                    inner_lr=inner_lr,
                    first_order=first_order,
                    lambda_consistency=lambda_consistency,
                    lambda_edge=lambda_edge,
                )
                losses.append(loss)
                episode_accuracies.append(accuracy)
            meta_loss = torch.stack(losses).mean()
            meta_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            episode_losses.append(float(meta_loss.item()))
            completed += current_batch
        scheduler.step()
        val_metrics = evaluate_prototype_episodes(
            model,
            val_generator,
            episodes=validation_episodes,
            device=device,
        )
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": {
                "loss": float(np.mean(episode_losses)),
                "query_accuracy": float(np.mean(episode_accuracies)),
            },
            "validation_prototype": val_metrics,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))

        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                _checkpoint_payload(
                    model,
                    config,
                    manifest,
                    source_conditions,
                    target_conditions,
                    "fewshot",
                    epoch,
                    record,
                ),
                output_dir / "best.pt",
            )
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break

    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    target_metrics = evaluate_prototype_episodes(
        model,
        target_generator,
        episodes=evaluation_episodes,
        device=device,
    )
    elapsed = time.perf_counter() - start_time
    summary = {
        "mode": "fewshot_proto_maml",
        "device": str(device),
        "best_epoch": best_epoch,
        "best_source_validation_accuracy": best_accuracy,
        "target_conditions": target_conditions,
        "target_prototype_evaluation": target_metrics,
        "episode": {"ways": ways, "shots": shots, "queries": queries},
        "elapsed_seconds": elapsed,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    torch.save(
        _checkpoint_payload(
            model,
            config,
            manifest,
            source_conditions,
            target_conditions,
            "fewshot",
            best_epoch,
            summary,
        ),
        output_dir / "last.pt",
    )
    _write_json(output_dir / "history.json", history)
    _write_json(output_dir / "summary.json", summary)
    return summary


def run_training(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config.get("seed", 42))
    set_seed(seed)
    device = select_device(str(config.get("device", "auto")))
    manifest, source, target, summary = prepare_experiment(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    mode = str(config["training"].get("mode", "fewshot")).lower()
    if mode == "supervised":
        return train_supervised(config, manifest, source, target, device)
    if mode in {"fewshot", "proto_maml", "fewshot_proto_maml"}:
        return train_fewshot(config, manifest, source, target, device)
    raise ValueError(f"Unknown training mode: {mode}")
