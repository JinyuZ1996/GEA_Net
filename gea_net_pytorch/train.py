from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from gea_net.config import load_config
from gea_net.data import DatasetFormatError
from gea_net.engine import prepare_experiment, run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train corrected GEA-Net on MCC5-THU or SEU."
    )
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument(
        "--mode",
        choices=["supervised", "fewshot"],
        help="Override training.mode.",
    )
    parser.add_argument("--device", help="Override device: auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--epochs", type=int, help="Override number of epochs.")
    parser.add_argument("--output-dir", help="Override output directory.")
    parser.add_argument("--max-classes", type=int, help="Limit classes for diagnostics.")
    parser.add_argument(
        "--max-windows-per-recording",
        type=int,
        help="Limit cached windows per raw CSV.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Discover, validate and cache the dataset without training.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="One tiny epoch/episode on at most three classes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.mode:
        config["training"]["mode"] = args.mode
    if args.device:
        config["device"] = args.device
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.output_dir:
        config["output_dir"] = str(Path(args.output_dir).expanduser().resolve())
    if args.max_classes is not None:
        config["dataset"]["max_classes"] = args.max_classes
    if args.max_windows_per_recording is not None:
        config["dataset"]["max_windows_per_recording"] = args.max_windows_per_recording

    if args.smoke:
        config["dataset"]["max_classes"] = min(
            int(config["dataset"].get("max_classes") or 3),
            3,
        )
        config["dataset"]["max_windows_per_recording"] = min(
            int(config["dataset"].get("max_windows_per_recording") or 8),
            8,
        )
        config["training"].update(
            {
                "epochs": 1,
                "batch_size": 8,
                "ways": 2,
                "shots": 1,
                "queries": 1,
                "episodes_per_epoch": 1,
                "meta_batch_size": 1,
                "validation_episodes": 1,
                "evaluation_episodes": 2,
                "early_stopping_patience": 1,
            }
        )
        config["graph"]["prior_samples"] = 8
        config["output_dir"] = str(Path(config["output_dir"]) / "smoke")

    try:
        if args.prepare_only:
            _, _, _, summary = prepare_experiment(config)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return
        summary = run_training(config)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except DatasetFormatError as error:
        print(f"Dataset error: {error}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
