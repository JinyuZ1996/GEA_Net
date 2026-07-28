from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from gea_net.data import (
    CachedWindowDataset,
    EpisodeGenerator,
    estimate_adjacency_prior,
    prepare_dataset,
    resolve_condition_split,
)
from gea_net.engine import proto_maml_query_loss
from gea_net.model import GEANet


def write_seu_csv(path: Path, length: int = 96, sampling_rate: float = 16000.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(length) / sampling_rate
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["Time(s)", "CH 1", "CH 2", "CH 3", ""])
        for index, value in enumerate(time):
            writer.writerow(
                [
                    value,
                    np.sin(index * 0.1),
                    np.cos(index * 0.2),
                    np.sin(index * 0.3) + 0.1,
                    "",
                ]
            )


def write_mcc_csv(path: Path, length: int = 96) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "speed",
                "torque",
                "motor_x",
                "motor_y",
                "motor_z",
                "gearbox_x",
                "gearbox_y",
                "gearbox_z",
            ]
        )
        for index in range(length):
            writer.writerow(
                [
                    1.0,
                    2.0,
                    *[np.sin(index * (axis + 1) * 0.02) for axis in range(6)],
                ]
            )


class ModelSmokeTest(unittest.TestCase):
    def test_forward_backward_and_affinity(self) -> None:
        torch.manual_seed(3)
        model = GEANet(
            num_sensors=3,
            num_classes=4,
            embed_dim=8,
            memory_size=5,
            graph_layers=2,
            graph_beta=0.5,
            dropout=0.0,
        )
        x = torch.randn(6, 3, 32)
        logits, auxiliary = model(x, return_aux=True)
        self.assertEqual(tuple(logits.shape), (6, 4))
        self.assertEqual(tuple(auxiliary["attention_affinity"].shape), (6, 3, 3))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(
            torch.allclose(
                auxiliary["attention_affinity"],
                auxiliary["attention_affinity"].transpose(1, 2),
                atol=1e-6,
            )
        )
        loss = logits.square().mean() + auxiliary["edge_sparsity"]
        loss.backward()
        self.assertIsNotNone(model.external_attention.memory_key.grad)

    def test_proto_maml_episode_has_outer_gradient(self) -> None:
        model = GEANet(
            num_sensors=3,
            num_classes=2,
            embed_dim=4,
            memory_size=4,
            graph_layers=1,
            dropout=0.0,
        )
        from gea_net.data import Episode

        episode = Episode(
            support_x=torch.randn(4, 3, 24),
            support_y=torch.tensor([0, 0, 1, 1]),
            query_x=torch.randn(4, 3, 24),
            query_y=torch.tensor([0, 0, 1, 1]),
            global_classes=torch.tensor([0, 1]),
        )
        loss, accuracy = proto_maml_query_loss(
            model,
            episode,
            device=torch.device("cpu"),
            inner_steps=1,
            inner_lr=0.01,
            first_order=True,
            lambda_consistency=0.1,
            lambda_edge=0.01,
        )
        loss.backward()
        self.assertTrue(np.isfinite(float(loss.item())))
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)
        self.assertIsNotNone(model.temporal_stem.conv.weight.grad)


class DatasetSmokeTest(unittest.TestCase):
    def test_seu_discovery_cache_split_and_episode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "seu"
            for condition in ("1200", "1800"):
                write_seu_csv(root / "Bearing Dataset" / f"Normal_{condition}.csv")
                write_seu_csv(root / "Bearing Dataset" / f"IF_{condition}.csv")
            config = {
                "name": "seu",
                "root": str(root),
                "cache_dir": str(Path(temporary) / "cache"),
                "window_size": 16,
                "stride": 16,
                "normalization": "per_window",
                "include_variable_condition": False,
                "target_conditions": ["1800"],
                "max_windows_per_recording": None,
            }
            manifest = prepare_dataset(config)
            self.assertEqual(manifest["num_sensors"], 3)
            self.assertEqual(len(manifest["class_names"]), 2)
            self.assertAlmostEqual(manifest["inferred_sampling_rate_median"], 16000, delta=2)
            source, target = resolve_condition_split(manifest, config)
            self.assertEqual(source, ["1200"])
            self.assertEqual(target, ["1800"])
            train = CachedWindowDataset(manifest, "train", source)
            val = CachedWindowDataset(manifest, "val", source)
            generator = EpisodeGenerator(
                train,
                query_dataset=val,
                ways=2,
                shots=1,
                queries=1,
                seed=1,
            )
            episode = generator.sample()
            self.assertEqual(tuple(episode.support_x.shape), (2, 3, 16))
            prior = estimate_adjacency_prior(train, max_samples=4, top_k=1)
            self.assertEqual(prior.shape, (3, 3))
            self.assertTrue(np.allclose(prior, prior.T))
            train.close()
            val.close()

    def test_mcc_selects_six_vibration_channels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "mcc"
            for condition in ("0Nm-900rpm", "5Nm-1200rpm"):
                write_mcc_csv(root / f"normal_steady_{condition}_1.csv")
                write_mcc_csv(root / f"gear_pitting_steady_{condition}_1.csv")
            config = {
                "name": "mcc5_thu",
                "root": str(root),
                "cache_dir": str(Path(temporary) / "cache"),
                "window_size": 16,
                "stride": 16,
                "normalization": "per_window",
                "include_variable_condition": True,
                "target_conditions": None,
            }
            manifest = prepare_dataset(config)
            self.assertEqual(manifest["num_sensors"], 6)
            self.assertEqual(set(manifest["class_names"]), {"health", "pitting"})
            self.assertNotIn("speed", [name.lower() for name in manifest["channel_names"]])
            self.assertNotIn("torque", [name.lower() for name in manifest["channel_names"]])


if __name__ == "__main__":
    unittest.main()
