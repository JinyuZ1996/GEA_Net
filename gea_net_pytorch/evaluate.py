from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader

from gea_net.data import (
    CachedWindowDataset,
    DatasetFormatError,
    EpisodeGenerator,
    prepare_dataset,
)
from gea_net.engine import (
    evaluate_prototype_episodes,
    evaluate_supervised,
    select_device,
    set_seed,
)
from gea_net.model import GEANet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved GEA-Net checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--batch-size", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    device = select_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    set_seed(int(config.get("seed", 42)))
    try:
        manifest = prepare_dataset(config["dataset"])
    except DatasetFormatError as error:
        print(f"Dataset error: {error}", file=sys.stderr)
        raise SystemExit(2) from None
    model = GEANet(**checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    split_fractions = config["dataset"].get("split_fractions", [0.7, 0.15, 0.15])
    mode = checkpoint["mode"]
    if mode == "supervised":
        conditions = checkpoint["target_conditions"] or checkpoint["source_conditions"]
        dataset = CachedWindowDataset(
            manifest,
            split="test",
            conditions=conditions,
            split_fractions=split_fractions,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size or int(config["training"].get("batch_size", 256)),
            shuffle=False,
            num_workers=int(config["training"].get("num_workers", 0)),
            pin_memory=device.type == "cuda",
        )
        metrics = evaluate_supervised(model, loader, device, nn.CrossEntropyLoss())
    else:
        train_config = config["training"]
        support = CachedWindowDataset(
            manifest,
            split="train",
            conditions=checkpoint["target_conditions"],
            split_fractions=split_fractions,
        )
        query = CachedWindowDataset(
            manifest,
            split="test",
            conditions=checkpoint["target_conditions"],
            split_fractions=split_fractions,
        )
        generator = EpisodeGenerator(
            support,
            query_dataset=query,
            ways=int(train_config.get("ways", 5)),
            shots=int(train_config.get("shots", 5)),
            queries=int(train_config.get("queries", 15)),
            seed=int(config.get("seed", 42)) + 100,
        )
        metrics = evaluate_prototype_episodes(
            model,
            generator,
            episodes=args.episodes or int(train_config.get("evaluation_episodes", 100)),
            device=device,
        )
    result = {
        "checkpoint": str(checkpoint_path),
        "mode": mode,
        "device": str(device),
        "metrics": metrics,
    }
    output_path = checkpoint_path.parent / "evaluation.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
