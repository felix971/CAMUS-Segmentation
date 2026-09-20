"""CPU-only tracer bullets for safe training continuation."""

import contextlib
import copy
import csv
import hashlib
import io
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import TensorDataset

from camus_segmentation import experiment, train
from camus_segmentation.loss import DiceCrossEntropyLoss


def tiny_model():
    return torch.nn.Conv2d(1, 4, 1)


def tiny_data():
    images = torch.arange(32, dtype=torch.float32).reshape(2, 1, 4, 4) / 32
    masks = torch.arange(32).reshape(2, 4, 4) % 4
    dataset = TensorDataset(images, masks)
    dataset.image_size = (4, 4)
    return dataset


class RandomDataset(torch.utils.data.Dataset):
    image_size = (4, 4)

    def __init__(self, fail_after=None):
        self.calls = 0
        self.fail_after = fail_after

    def __len__(self):
        return 6

    def __getitem__(self, index):
        if self.calls == self.fail_after:
            raise RuntimeError("synthetic interruption")
        self.calls += 1
        image, mask = tiny_data()[index % 2]
        noise = random.random() + float(np.random.random()) + torch.rand_like(image)
        return image + noise * 0.05, mask


class StochasticModel(torch.nn.Conv2d):
    def __init__(self):
        super().__init__(1, 4, 1)

    def forward(self, images):
        return torch.nn.functional.dropout(super().forward(images), 0.25, self.training)


def legacy_checkpoint(path):
    torch.manual_seed(7)
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    images, masks = tiny_data().tensors
    DiceCrossEntropyLoss()(model(images), masks).backward()
    optimizer.step()
    for state in optimizer.state.values():
        state["step"].fill_(2000)
    checkpoint = {
        "epoch": 10,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation_loss": 0.30922128617763517,
        "validation_dice": [0.9138287901878357, 0.8206327557563782, 0.8559705018997192],
        "mean_validation_dice": 0.8634774088859558,
        "training_config": {
            "epochs": 10, "batch_size": 2, "image_size": [4, 4],
            "loss": "DiceCrossEntropyLoss", "optimizer": "AdamW",
            "learning_rate": 0.001, "weight_decay": 0.0001,
            "betas": [0.9, 0.999], "eps": 1e-8,
        },
    }
    torch.save(checkpoint, path)
    return checkpoint


class TrainingResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "legacy.pt"
        self.legacy = legacy_checkpoint(self.source)

    def arguments(self, name="run", epochs=11, resume=True, extra=()):
        argv = ["train", "--run-dir", str(self.root / name),
                "--epochs", str(epochs), "--device", "cpu", *extra]
        if resume:
            argv += ["--resume", str(self.source)]
        with patch("sys.argv", argv):
            return train.parse_arguments()

    def run_tiny(self, args, dataset=None, model_factory=tiny_model):
        with contextlib.redirect_stdout(io.StringIO()):
            return train.run_training(args, dataset or tiny_data(), tiny_data(),
                                      model_factory=model_factory)

    def test_legacy_resume_restores_adam_state_at_epoch_eleven(self):
        self.assertTrue(callable(getattr(train, "run_training", None)),
                        "a reusable CPU training entry point is required")
        original = self.source.read_bytes()
        args = self.arguments()
        self.run_tiny(args)
        result = torch.load(args.run_dir / "last.pt", weights_only=True)
        self.assertEqual(result["epoch"], 11)
        self.assertEqual(result["global_step"], 2001)
        self.assertEqual(self.source.read_bytes(), original)
        self.assertEqual(result["training_config"]["batch_size"], 2)

        expected_model = tiny_model()
        expected_model.load_state_dict(self.legacy["model_state_dict"])
        expected_optimizer = torch.optim.AdamW(expected_model.parameters())
        expected_optimizer.load_state_dict(self.legacy["optimizer_state_dict"])
        images, masks = tiny_data().tensors
        DiceCrossEntropyLoss()(expected_model(images), masks).backward()
        expected_optimizer.step()
        for name, value in expected_model.state_dict().items():
            torch.testing.assert_close(result["model_state_dict"][name], value)
        for key, expected in expected_optimizer.state_dict()["state"].items():
            for field in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(result["optimizer_state_dict"]["state"][key][field],
                                           expected[field])
        with (args.run_dir / "history.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([int(row["epoch"]) for row in rows], [11])
        self.assertEqual(int(rows[0]["optimizer_steps"]), 2001)

    def test_resume_conflicts_are_rejected_not_silently_ignored(self):
        for flag, value in (("--batch-size", "3"), ("--learning-rate", "0.002"),
                            ("--weight-decay", "0.01")):
            with self.subTest(flag=flag):
                args = self.arguments(name=flag, extra=(flag, value))
                with self.assertRaisesRegex(ValueError, "conflict"):
                    self.run_tiny(args)
                self.assertFalse(args.run_dir.exists())
        args = self.arguments(extra=("--batch-size", "2", "--learning-rate", "0.001"))
        self.assertEqual(train.training_config(args, self.legacy)["batch_size"], 2)
        fresh = train.training_config(self.arguments(resume=False))
        self.assertEqual((fresh["batch_size"], fresh["learning_rate"], fresh["weight_decay"]),
                         (8, 0.001, 0.0001))

    def test_existing_directory_rejected_before_constructing_model(self):
        args = self.arguments()
        args.run_dir.mkdir()
        sentinel = args.run_dir / "best.pt"
        sentinel.write_bytes(b"do not touch")
        def unexpected_model():
            self.fail("collision must be checked before model/device allocation")
        with self.assertRaises(FileExistsError):
            self.run_tiny(args, model_factory=unexpected_model)
        self.assertEqual(sentinel.read_bytes(), b"do not touch")
        self.assertEqual(list(args.run_dir.iterdir()), [sentinel])

    def test_worse_validation_cannot_replace_legacy_best(self):
        args = self.arguments()
        self.run_tiny(args)
        self.assertTrue((args.run_dir / "baseline.pt").is_file(), "baseline must be retained")
        self.assertEqual((args.run_dir / "baseline.pt").read_bytes(), self.source.read_bytes())
        self.assertEqual((args.run_dir / "best.pt").read_bytes(), self.source.read_bytes())
        point = json.loads((args.run_dir / "baseline.json").read_text())
        for key in ("epoch", "validation_loss", "validation_dice", "mean_validation_dice"):
            self.assertEqual(point[key], self.legacy[key])
        self.assertEqual(point["sha256"], hashlib.sha256(self.source.read_bytes()).hexdigest())
        last = torch.load(args.run_dir / "last.pt", weights_only=True)
        self.assertLess(last["mean_validation_dice"], self.legacy["mean_validation_dice"])
        self.assertEqual(last["best_epoch"], 10)
        self.assertEqual(last["best_mean_validation_dice"], self.legacy["mean_validation_dice"])

    def test_rng_roundtrip_is_weights_only_loadable(self):
        self.assertTrue(callable(getattr(experiment, "capture_rng", None)),
                        "checkpoints need serializable RNG state")
        generators = {"train": torch.Generator().manual_seed(42),
                      "validation": torch.Generator().manual_seed(43)}
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        # Populate numpy's cached Gaussian as well as its integer state.
        np.random.normal()
        state = experiment.capture_rng(generators)
        path = self.root / "rng.pt"
        torch.save(state, path)
        loaded = torch.load(path, weights_only=True)
        def draws():
            return (random.random(), np.random.normal(), torch.rand(3),
                    torch.rand(3, generator=generators["train"]),
                    torch.rand(3, generator=generators["validation"]))
        expected = draws()
        draws()
        experiment.restore_rng(loaded, generators)
        actual = draws()
        self.assertEqual(expected[:2], actual[:2])
        for a, b in zip(expected[2:], actual[2:]):
            self.assertTrue(torch.equal(a, b))

    def test_interruption_restart_matches_uninterrupted_cpu_run(self):
        whole = self.arguments(name="whole", epochs=13)
        self.run_tiny(whole, RandomDataset(), StochasticModel)
        interrupted = self.arguments(name="interrupted", epochs=13)
        with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
            self.run_tiny(interrupted, RandomDataset(fail_after=6), StochasticModel)
        self.source = interrupted.run_dir / "last.pt"
        resumed = self.arguments(name="continued", epochs=13)
        self.run_tiny(resumed, RandomDataset(), StochasticModel)
        a = torch.load(whole.run_dir / "last.pt", weights_only=True)
        b = torch.load(resumed.run_dir / "last.pt", weights_only=True)
        for name in a["model_state_dict"]:
            self.assertTrue(torch.equal(a["model_state_dict"][name], b["model_state_dict"][name]),
                            "restarting must restore all stochastic state")
        self.assertEqual(a["global_step"], b["global_step"])
        for key, state in a["optimizer_state_dict"]["state"].items():
            for field in ("step", "exp_avg", "exp_avg_sq"):
                self.assertTrue(torch.equal(state[field], b["optimizer_state_dict"]["state"][key][field]))
        self.assertEqual((resumed.run_dir / "baseline.pt").read_bytes(),
                         (self.root / "legacy.pt").read_bytes())
        self.assertEqual((resumed.run_dir / "best.pt").read_bytes(),
                         (self.root / "legacy.pt").read_bytes())
        self.assertEqual(b["best_epoch"], 10)
        self.assertEqual(b["source"]["sha256"], experiment.sha256(self.source))
        self.assertEqual(b["source"]["path"], str(self.source.resolve()))
        self.assertIn("rng_state", b)
        with (whole.run_dir / "history.csv").open() as stream:
            full_rows = list(csv.DictReader(stream))
        with (resumed.run_dir / "history.csv").open() as stream:
            continued_rows = list(csv.DictReader(stream))
        for full, continued in zip(full_rows[1:], continued_rows):
            for key in full.keys() - {"epoch_seconds"}:
                self.assertEqual(full[key], continued[key])
        self.assertEqual([row["epoch"] for row in continued_rows], ["12", "13"])

    def test_nonfinite_training_loss_fails_before_optimizer_update(self):
        model = tiny_model()
        original = {key: value.clone() for key, value in model.state_dict().items()}
        optimizer = torch.optim.AdamW(model.parameters())
        data = tiny_data()
        data.tensors[0].fill_(float("nan"))
        loader = torch.utils.data.DataLoader(data, batch_size=2)
        with self.assertRaisesRegex(FloatingPointError, "training loss"):
            train.train_one_epoch(model, loader, DiceCrossEntropyLoss(), optimizer, torch.device("cpu"))
        self.assertEqual(len(optimizer.state), 0)
        for key, value in original.items():
            self.assertTrue(torch.equal(model.state_dict()[key], value))

    def test_nonfinite_validation_does_not_publish_checkpoint(self):
        args = self.arguments()
        validation = tiny_data()
        validation.tensors[0].fill_(float("nan"))
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(FloatingPointError, "validation"):
                train.run_training(args, tiny_data(), validation, model_factory=tiny_model)
        self.assertFalse((args.run_dir / "last.pt").exists())
        self.assertEqual((args.run_dir / "best.pt").read_bytes(), self.source.read_bytes())
        with (args.run_dir / "history.csv").open() as stream:
            self.assertEqual(list(csv.DictReader(stream)), [])

    def test_smoke_is_one_update_not_promotable_to_formal_resume(self):
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                args = self.arguments(epochs=50, extra=("--smoke-test", "--seed", "42"))
            except SystemExit:
                self.fail("--smoke-test is required")
        training, validation = RandomDataset(fail_after=2), RandomDataset(fail_after=2)
        with contextlib.redirect_stdout(io.StringIO()):
            train.run_training(args, training, validation, model_factory=tiny_model)
        self.assertEqual((training.calls, validation.calls), (2, 2))
        last = torch.load(args.run_dir / "last.pt", weights_only=True)
        self.assertEqual((last["epoch"], last["global_step"]), (11, 2001))
        self.assertIs(last["smoke_test"], True)
        self.assertIs(json.loads((args.run_dir / "config.json").read_text())["smoke_test"], True)
        for name in ("last.pt", "best.pt", "baseline.pt"):
            with self.subTest(source=name):
                # Even an artifact moved out of its run directory remains marked.
                self.source = self.root / ("copied_" + name)
                self.source.write_bytes((args.run_dir / name).read_bytes())
                formal = self.arguments(name="formal_" + name, epochs=50)
                with self.assertRaisesRegex(ValueError, "smoke"):
                    self.run_tiny(formal)
                self.assertFalse(formal.run_dir.exists())

    def test_failed_checkpoint_save_keeps_previous_complete_epoch(self):
        args = self.arguments(epochs=12)
        real_save = torch.save
        saved_bytes = None
        def failing_save(state, destination, *positional, **kwargs):
            nonlocal saved_bytes
            path = destination if isinstance(destination, Path) else Path(destination.name)
            if state.get("epoch") == 12 and "last.pt" in path.name:
                if isinstance(destination, Path):
                    destination.write_bytes(b"partial")
                else:
                    destination.write(b"partial")
                raise OSError("synthetic disk failure")
            result = real_save(state, destination, *positional, **kwargs)
            if state.get("epoch") == 11 and "last.pt" in path.name:
                if hasattr(destination, "flush"):
                    destination.flush()
                saved_bytes = path.read_bytes()
            return result
        with patch("torch.save", side_effect=failing_save):
            with self.assertRaisesRegex(OSError, "synthetic disk failure"):
                self.run_tiny(args)
        self.assertIsNotNone(saved_bytes)
        self.assertEqual((args.run_dir / "last.pt").read_bytes(), saved_bytes,
                         "failed serialization must never truncate last.pt")
        self.assertEqual(torch.load(args.run_dir / "last.pt", weights_only=True)["epoch"], 11)
        self.assertEqual((args.run_dir / "best.pt").read_bytes(), self.source.read_bytes())
        self.assertEqual(list(args.run_dir.glob("*.tmp")), [])

    def test_split_preflight_checks_ids_files_without_loading_test_data(self):
        self.assertTrue(callable(getattr(train, "preflight_data", None)),
                        "split/data preflight must happen before training")
        split_root, data_root = self.root / "splits", self.root / "images"
        split_root.mkdir()
        files = {name: split_root / f"subgroup_{name}.txt"
                 for name in ("training", "validation", "testing")}
        files["training"].write_text("patient0001\n")
        files["validation"].write_text("patient0002\n")
        files["testing"].write_text("patient0003\n")
        for name in ("training", "validation"):
            for sample in train.build_samples(data_root, files[name]):
                sample.image_path.parent.mkdir(parents=True, exist_ok=True)
                sample.image_path.write_bytes(b"existence only")
                sample.mask_path.write_bytes(b"existence only")
        training, validation, info = train.preflight_data(data_root, split_root)
        self.assertEqual((len(training), len(validation)), (4, 4))
        self.assertEqual(info["splits"]["training"]["sha256"], experiment.sha256(files["training"]))
        self.assertFalse((data_root / "patient0003").exists())
        for value, message in (("patient0001\npatient0001\n", "duplicate"),
                               ("patient0002\n", "overlap"), ("", "empty"),
                               ("patient9999\n", "Missing")):
            with self.subTest(value=value):
                files["training"].write_text(value)
                with self.assertRaisesRegex(ValueError, message):
                    train.preflight_data(data_root, split_root)
        files["training"].write_text("patient0001\n")
        files["testing"].write_text("patient0001\n")
        with self.assertRaisesRegex(ValueError, "overlap"):
            train.preflight_data(data_root, split_root)

    def test_run_artifacts_report_provenance_without_inventing_history(self):
        args = self.arguments(extra=("--seed", "42"))
        self.run_tiny(args)
        config = json.loads((args.run_dir / "config.json").read_text())
        self.assertIn("seed_meaning", config)
        self.assertEqual(config["rng_initialization"], "legacy_seed_no_historical_rng")
        self.assertEqual(config["training_config"]["seed"], 42)
        self.assertEqual(config["source"]["sha256"], experiment.sha256(self.source))
        self.assertRegex(config["git"]["head"], r"^[0-9a-f]{40}$")
        self.assertIsInstance(config["git"]["dirty"], bool)
        source_file = Path(train.__file__)
        self.assertEqual(config["source_hashes"]["camus_segmentation/train.py"], experiment.sha256(source_file))
        self.assertEqual(config["environment"]["device"], "cpu")
        self.assertIn("torch", config["environment"]["packages"])
        self.assertIn("cudnn_benchmark", config["environment"])
        self.assertEqual(config["data"]["counts"], {"training_samples": 2, "validation_samples": 2})
        self.assertEqual(config["history_segment"], {"start_epoch": 11, "target_epoch": 11})
        with (args.run_dir / "history.csv").open() as stream:
            rows = list(csv.reader(stream))
        self.assertEqual(rows[0], ["epoch", "optimizer_steps", "learning_rate", "train_loss", "val_loss",
                                  "val_dice_lv", "val_dice_myo", "val_dice_la", "val_mean_dice", "epoch_seconds"])
        self.assertEqual(len(rows), 2)
        self.assertIn("Epoch 11/11", (args.run_dir / "train.log").read_text())
        self.assertIn("Validation loss", (args.run_dir / "train.log").read_text())
        summary = json.loads((args.run_dir / "summary.json").read_text())
        self.assertEqual((summary["epoch"], summary["best_epoch"]), (11, 10))
        self.assertEqual(summary["best_mean_validation_dice"], self.legacy["mean_validation_dice"])
        self.assertGreaterEqual(summary["elapsed_seconds"], 0)
        self.assertNotIn("test_metrics", summary)
        last = torch.load(args.run_dir / "last.pt", weights_only=True)
        self.assertEqual(last["elapsed_seconds"], summary["elapsed_seconds"])
        self.source = args.run_dir / "last.pt"
        continuation = self.arguments(name="continuation", epochs=12)
        self.run_tiny(continuation)
        continued = json.loads((continuation.run_dir / "config.json").read_text())
        self.assertEqual(continued["rng_initialization"], "restored_checkpoint_rng")
        self.assertEqual(continued["source_history"]["sha256"], experiment.sha256(args.run_dir / "history.csv"))
        self.assertEqual((continuation.run_dir / "source_history.csv").read_bytes(),
                         (args.run_dir / "history.csv").read_bytes())
        self.assertEqual(continued["history_segment"], {"start_epoch": 12, "target_epoch": 12})

    def test_saved_optimizer_recipe_and_step_metadata_must_agree(self):
        mutations = (
            lambda cp: cp["training_config"].update(learning_rate=0.01),
            lambda cp: cp["training_config"].update(weight_decay=0.1),
            lambda cp: cp["training_config"].update(betas=[0.8, 0.999]),
            lambda cp: cp["training_config"].update(eps=1e-7),
            lambda cp: next(iter(cp["optimizer_state_dict"]["state"].values()))["step"].fill_(1999),
            lambda cp: cp.update(global_step=1),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                checkpoint = copy.deepcopy(self.legacy)
                mutate(checkpoint)
                torch.save(checkpoint, self.source)
                args = self.arguments(name=f"mismatch_{index}")
                with self.assertRaisesRegex(ValueError, "optimizer|step"):
                    self.run_tiny(args)
                self.assertFalse(args.run_dir.exists())

    def test_saved_model_loss_preprocessing_recipe_cannot_change(self):
        for key, value in (("loss", "CrossEntropyLoss"), ("optimizer", "SGD"),
                           ("model", "different model"), ("preprocessing", "different preprocessing")):
            with self.subTest(key=key):
                checkpoint = copy.deepcopy(self.legacy)
                checkpoint["training_config"][key] = value
                torch.save(checkpoint, self.source)
                args = self.arguments(name=key)
                with self.assertRaisesRegex(ValueError, "recipe"):
                    self.run_tiny(args)
                self.assertFalse(args.run_dir.exists())
        torch.save(self.legacy, self.source)
        data = tiny_data()
        data.image_size = (8, 8)
        args = self.arguments(name="size")
        with self.assertRaisesRegex(ValueError, "image_size"):
            self.run_tiny(args, data)
        self.assertFalse(args.run_dir.exists())

    def test_new_schema_must_restore_rng_not_reseed_silently(self):
        first = self.arguments(extra=("--seed", "123"))
        self.run_tiny(first)
        self.source = first.run_dir / "last.pt"
        args = self.arguments(name="conflicting_seed", epochs=12, extra=("--seed", "42"))
        with self.assertRaisesRegex(ValueError, "seed.*conflict|conflict.*seed"):
            self.run_tiny(args)
        self.assertFalse(args.run_dir.exists())
        checkpoint = torch.load(self.source, weights_only=True)
        del checkpoint["rng_state"]
        self.source = self.root / "missing_rng.pt"
        torch.save(checkpoint, self.source)
        args = self.arguments(name="missing_rng", epochs=12)
        with self.assertRaisesRegex(ValueError, "rng_state"):
            self.run_tiny(args)
        self.assertFalse(args.run_dir.exists())

    def test_nonfinite_source_metrics_rejected_before_artifact_creation(self):
        for field in ("validation_loss", "mean_validation_dice", "validation_dice"):
            with self.subTest(field=field):
                checkpoint = copy.deepcopy(self.legacy)
                checkpoint[field] = [0.5, float("nan"), 0.5] if field == "validation_dice" else float("inf")
                torch.save(checkpoint, self.source)
                args = self.arguments(name=field)
                with self.assertRaisesRegex(ValueError, "finite"):
                    self.run_tiny(args)
                self.assertFalse(args.run_dir.exists())

    def test_failed_baseline_copy_never_publishes_partial_best(self):
        args = self.arguments()
        def broken_copy(source, destination, *positional, **kwargs):
            path = Path(destination)
            path.write_bytes(b"partial copy")
            raise OSError("synthetic copy failure")
        with patch("shutil.copyfile", side_effect=broken_copy):
            with self.assertRaisesRegex(OSError, "synthetic copy failure"):
                self.run_tiny(args)
        self.assertFalse((args.run_dir / "best.pt").exists(), "partial best must not be published")
        self.assertFalse((args.run_dir / "last.pt").exists())
        self.assertEqual(list(args.run_dir.glob("*.tmp")), [])

    def test_continuation_requires_same_training_data_identity(self):
        first = self.arguments()
        split = self.root / "training_ids.txt"
        split.write_text("patient0001\n")
        data_info = {"splits": {"training": {"sha256": experiment.sha256(split)}}}
        with contextlib.redirect_stdout(io.StringIO()):
            train.run_training(first, tiny_data(), tiny_data(), model_factory=tiny_model, data_info=data_info)
        self.source = first.run_dir / "last.pt"
        args = self.arguments(name="changed_split", epochs=12)
        split.write_text("patient0002\n")
        changed = {"splits": {"training": {"sha256": experiment.sha256(split)}}}
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "split"):
                train.run_training(args, tiny_data(), tiny_data(), model_factory=tiny_model, data_info=changed)
        self.assertFalse(args.run_dir.exists())
        args = self.arguments(name="changed_count", epochs=12)
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "count"):
                train.run_training(args, RandomDataset(), tiny_data(), model_factory=tiny_model, data_info=data_info)
        self.assertFalse(args.run_dir.exists())

    def test_missing_true_best_is_rejected_before_new_run_creation(self):
        first = self.arguments()
        self.run_tiny(first)
        self.source = first.run_dir / "last.pt"
        (first.run_dir / "best.pt").unlink()
        args = self.arguments(name="missing_best", epochs=12)
        with self.assertRaisesRegex(ValueError, "linked checkpoint"):
            self.run_tiny(args)
        self.assertFalse(args.run_dir.exists(), "validate true best before creating a new run")

    def test_interruption_during_best_publish_recovers_best_from_last(self):
        self.legacy.update(validation_dice=[0.0, 0.0, 0.0], mean_validation_dice=0.0)
        torch.save(self.legacy, self.source)
        whole = self.arguments(name="whole", epochs=12)
        self.run_tiny(whole)
        interrupted = self.arguments(name="interrupted", epochs=12)
        real_save = torch.save
        def fail_best(state, destination, *positional, **kwargs):
            path = destination if isinstance(destination, Path) else Path(destination.name)
            if state.get("epoch") == 11 and "best.pt" in path.name:
                raise OSError("synthetic best publish failure")
            return real_save(state, destination, *positional, **kwargs)
        with patch("torch.save", side_effect=fail_best):
            with self.assertRaisesRegex(OSError, "best publish failure"):
                self.run_tiny(interrupted)
        self.assertTrue((interrupted.run_dir / "last.pt").is_file(),
                        "commit last before publishing its new best")
        with (interrupted.run_dir / "history.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([row["epoch"] for row in rows], ["11"],
                         "a committed checkpoint must not lose its measured history row")
        self.source = interrupted.run_dir / "last.pt"
        recovered = self.arguments(name="recovered", epochs=12)
        self.run_tiny(recovered)
        expected = torch.load(whole.run_dir / "best.pt", weights_only=True)
        actual = torch.load(recovered.run_dir / "best.pt", weights_only=True)
        self.assertEqual(actual["epoch"], expected["epoch"])
        self.assertEqual(actual["mean_validation_dice"], expected["mean_validation_dice"])
        for key, value in expected["model_state_dict"].items():
            self.assertTrue(torch.equal(value, actual["model_state_dict"][key]))
        self.assertEqual(torch.load(recovered.run_dir / "last.pt", weights_only=True)["epoch"], 12)

    def test_total_fifty_produces_only_epochs_eleven_through_fifty(self):
        args = self.arguments(epochs=50, extra=("--seed", "42"))
        self.run_tiny(args)
        with (args.run_dir / "history.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([int(row["epoch"]) for row in rows], list(range(11, 51)))
        self.assertEqual(len(rows), 40)
        last = torch.load(args.run_dir / "last.pt", weights_only=True)
        self.assertEqual((last["epoch"], last["global_step"]), (50, 2040))
        self.assertEqual(json.loads((args.run_dir / "summary.json").read_text())["epoch"], 50)

    def test_fresh_run_can_resume_from_its_own_best_checkpoint(self):
        first = self.arguments(name="fresh", epochs=1, resume=False)
        self.run_tiny(first)
        checkpoint = torch.load(first.run_dir / "best.pt", weights_only=True)
        self.assertEqual(checkpoint["epoch"], 1)
        self.assertIsNone(checkpoint["baseline"])
        self.assertEqual(checkpoint["global_step"], 1)
        self.source = first.run_dir / "best.pt"
        continued = self.arguments(name="fresh_continued", epochs=2)
        self.run_tiny(continued)
        last = torch.load(continued.run_dir / "last.pt", weights_only=True)
        self.assertEqual((last["epoch"], last["global_step"]), (2, 2))
        self.assertIsNone(last["baseline"])
        self.assertFalse((continued.run_dir / "baseline.pt").exists())

    def test_empty_validation_is_rejected_before_model_allocation(self):
        args = self.arguments()
        empty = tiny_data()
        empty.tensors = tuple(value[:0] for value in empty.tensors)
        def unexpected_model():
            self.fail("empty datasets must fail before model allocation")
        with self.assertRaisesRegex(ValueError, "nonempty"):
            train.run_training(args, tiny_data(), empty, model_factory=unexpected_model)
        self.assertFalse(args.run_dir.exists())

    def test_invalid_checkpoint_cannot_silently_become_fresh_training(self):
        torch.save({}, self.source)
        args = self.arguments()
        with self.assertRaisesRegex(ValueError, "checkpoint fields"):
            self.run_tiny(args)
        self.assertFalse(args.run_dir.exists())

    def test_total_target_must_exceed_saved_epoch(self):
        for target in (0, 9, 10):
            with self.subTest(target=target):
                args = self.arguments(epochs=target)
                with self.assertRaisesRegex(ValueError, "target.*epoch"):
                    train.training_config(args, self.legacy)
                self.assertFalse(args.run_dir.exists())

    def test_run_directory_is_required_instead_of_overwriting_baseline(self):
        with contextlib.redirect_stderr(io.StringIO()), patch("sys.argv", ["train"]):
            with self.assertRaises(SystemExit):
                train.parse_arguments()
        with tempfile.TemporaryDirectory() as directory:
            with patch("sys.argv", ["train", "--run-dir", directory]):
                args = train.parse_arguments()
            self.assertEqual(args.run_dir, Path(directory))
            self.assertIsNone(args.resume)
            self.assertEqual(args.epochs, 10)


if __name__ == "__main__":
    unittest.main()
