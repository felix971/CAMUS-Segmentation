"""CPU-only inference CLI checks using synthetic CAMUS-shaped NIfTI files."""

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader

from camus_segmentation import evaluate_test, visualize_prediction
from camus_segmentation.data import DEFAULT_IMAGE_SIZE, CamusDataset, build_samples
from camus_segmentation.evaluation import evaluate_model
from camus_segmentation.loss import DiceCrossEntropyLoss
from camus_segmentation.model import UNet2D


class InferenceCliTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        rng_state = torch.get_rng_state()
        self.addCleanup(torch.set_rng_state, rng_state)
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, self.old_threads)
        self.module_path = self.root / "camus_segmentation" / "evaluate_test.py"
        self.split = self.root / "data/splits/subgroup_validation.txt"
        self.split.parent.mkdir(parents=True)
        self.split.write_text("patient0001\n")
        (self.split.parent / "subgroup_testing.txt").write_text("patient0001\n")
        self.camus_root = self.root / "data/raw/camus"
        for sample in build_samples(self.camus_root, self.split):
            sample.image_path.parent.mkdir(parents=True, exist_ok=True)
            image = np.arange(16 * 32, dtype=np.float32).reshape(16, 32) % 256
            mask = np.indices((16, 32)).sum(axis=0).astype(np.uint8) % 4
            nib.save(nib.Nifti1Image(image, np.eye(4)), sample.image_path)
            nib.save(nib.Nifti1Image(mask, np.eye(4)), sample.mask_path)
        self.model = UNet2D(base_channels=2)
        with torch.no_grad():
            for parameter in self.model.parameters():
                parameter.zero_()
            self.model.segmentation_head.bias[2] = 3.0
        self.checkpoint = self.root / "chosen.pt"
        self.payload = {
            "epoch": 7,
            "model_state_dict": self.model.state_dict(),
            "mean_validation_dice": 0.625,
            "training_config": {"image_size": [16, 32], "batch_size": 3},
        }
        torch.save(self.payload, self.checkpoint)

    def invoke(self, module, arguments):
        stream = io.StringIO()
        with mock.patch.object(module, "__file__", str(self.module_path)), \
                mock.patch("sys.argv", [module.__name__, *arguments]), \
                contextlib.redirect_stdout(stream):
            module.main()
        return stream.getvalue()

    def test_evaluation_uses_selected_weights_and_checkpoint_geometry_on_cpu(self):
        dataset = CamusDataset(build_samples(self.camus_root, self.split), (16, 32))
        expected_loss, expected_dice = evaluate_model(
            self.model, DataLoader(dataset, batch_size=3),
            DiceCrossEntropyLoss(), torch.device("cpu"),
        )
        with mock.patch.object(evaluate_test, "UNet2D", side_effect=lambda: UNet2D(base_channels=2)), \
                mock.patch.object(evaluate_test, "DataLoader", wraps=DataLoader) as loader, \
                mock.patch.object(evaluate_test, "evaluate_model", wraps=evaluate_model) as evaluate, \
                mock.patch.object(evaluate_test.torch, "load", wraps=torch.load) as load:
            try:
                output = self.invoke(evaluate_test, [
                    "--checkpoint", str(self.checkpoint), "--split", "validation",
                    "--device", "cpu",
                ])
            except Exception as error:
                self.fail(f"Explicit-checkpoint CPU evaluation must work: {error}")
        self.assertEqual(load.call_args.args[0], self.checkpoint)
        self.assertEqual(load.call_args.kwargs, {"map_location": "cpu", "weights_only": True})
        self.assertEqual(loader.call_args.kwargs["batch_size"], 3)
        self.assertEqual(loader.call_args.args[0].image_size, (16, 32))
        self.assertFalse(loader.call_args.kwargs["shuffle"])
        self.assertEqual(evaluate.call_args.args[3], torch.device("cpu"))
        for name, weights in self.model.state_dict().items():
            torch.testing.assert_close(evaluate.call_args.args[0].state_dict()[name], weights)
        self.assertIn("Validation patients:", output)
        self.assertIn(f"{expected_loss:.6f}", output)
        self.assertIn(f"Mean foreground Dice: {expected_dice.mean().item():.4f}", output)

    def test_json_report_records_checkpoint_split_protocol_and_unrounded_metrics(self):
        output_path = self.root / "reports" / "validation.json"
        checkpoint_bytes = self.checkpoint.read_bytes()
        expected_loss, expected_dice = evaluate_model(
            self.model,
            DataLoader(CamusDataset(build_samples(self.camus_root, self.split), (16, 32)), batch_size=2),
            DiceCrossEntropyLoss(), torch.device("cpu"),
        )
        with mock.patch.object(evaluate_test, "UNet2D", side_effect=lambda: UNet2D(base_channels=2)):
            self.invoke(evaluate_test, [
                "--checkpoint", str(self.checkpoint), "--split", "validation",
                "--device", "cpu", "--batch-size", "2", "--output-json", str(output_path),
            ])
        report = json.loads(output_path.read_text())
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["checkpoint"], {
            "path": str(self.checkpoint.resolve()),
            "sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
            "epoch": 7,
            "mean_validation_dice": 0.625,
        })
        self.assertEqual(report["split"], {
            "name": "validation", "path": str(self.split.resolve()),
            "sha256": hashlib.sha256(self.split.read_bytes()).hexdigest(),
            "patients": 1, "samples": 4,
        })
        self.assertEqual(report["inference"], {
            "device": "cpu", "batch_size": 2, "batches": 2, "image_size": [16, 32],
        })
        self.assertEqual(report["metric_protocol"], {
            "grid": "resized", "prediction": "argmax", "foreground_class_ids": [1, 2, 3],
            "background_excluded": True, "smooth": 1e-6, "empty_class_score": 1.0,
            "reduction": "per-image Dice, mean over samples, then mean over foreground classes",
            "image_interpolation": "bilinear", "image_align_corners": False,
            "mask_interpolation": "nearest", "loss": "DiceCrossEntropyLoss",
            "loss_reduction": "sample-weighted mean of batch losses",
        })
        self.assertEqual(report["metrics"]["loss"], expected_loss)
        self.assertEqual(report["metrics"]["mean_foreground_dice"], expected_dice.mean().item())
        self.assertEqual(report["metrics"]["foreground_dice"], dict(zip(
            evaluate_test.FOREGROUND_CLASS_NAMES, expected_dice.tolist(),
        )))
        self.assertEqual(self.checkpoint.read_bytes(), checkpoint_bytes)

    def test_output_collisions_are_rejected_before_loading_checkpoint_or_data(self):
        existing = self.root / "existing.json"
        existing.write_text("curated result")
        symlink = self.root / "checkpoint-link.json"
        symlink.symlink_to(self.checkpoint)
        hardlink = self.root / "checkpoint-hardlink.json"
        os.link(self.checkpoint, hardlink)
        dangling = self.root / "dangling.json"
        dangling.symlink_to(self.root / "missing.json")
        before = self.checkpoint.read_bytes()
        for module, flag in ((evaluate_test, "--output-json"), (visualize_prediction, "--output")):
            for output in (existing, self.checkpoint, symlink, hardlink, dangling, self.split, self.root):
                with self.subTest(module=module.__name__, output=output.name), \
                        mock.patch.object(module.torch, "load") as load, \
                        mock.patch.object(module, "build_samples") as build:
                    with self.assertRaises(FileExistsError):
                        self.invoke(module, [
                            "--checkpoint", str(self.checkpoint), "--device", "cpu", flag, str(output),
                        ])
                    load.assert_not_called()
                    build.assert_not_called()
        self.assertEqual(self.checkpoint.read_bytes(), before)
        self.assertEqual(existing.read_text(), "curated result")
        self.assertFalse((self.root / "missing.json").exists())

    def test_invalid_checkpoint_config_is_rejected_before_data_or_model_work(self):
        invalid_configs = [None, [], "bad"]
        invalid_configs += [{"image_size": value} for value in (
            None, "16,32", [16], [16, 32, 48], [0, 32], [-16, 32], [8, 32],
            [17, 32], [16.0, 32], [True, 32], {"height": 16, "width": 32},
        )]
        invalid_configs += [{"batch_size": value} for value in (0, -2, True, 1.5, "2", None)]
        for config in invalid_configs:
            for module in (evaluate_test, visualize_prediction):
                with self.subTest(config=config, module=module.__name__):
                    torch.save({**self.payload, "training_config": config}, self.checkpoint)
                    with mock.patch.object(module, "build_samples", side_effect=AssertionError("data accessed")), \
                            mock.patch.object(module, "UNet2D", side_effect=AssertionError("model built")):
                        with self.assertRaisesRegex(ValueError, "training_config|image_size|batch_size"):
                            self.invoke(module, ["--checkpoint", str(self.checkpoint), "--device", "cpu"])

    def test_invalid_cli_batch_size_fails_before_checkpoint_load(self):
        for value in ("0", "-1", "1.5", "true"):
            with self.subTest(batch_size=value), \
                    mock.patch.object(evaluate_test.torch, "load", side_effect=AssertionError("checkpoint accessed")), \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    self.invoke(evaluate_test, ["--batch-size", value])
                self.assertEqual(error.exception.code, 2)


    def test_visualization_uses_selected_checkpoint_sample_and_transposed_grid(self):
        output = self.root / "figures/chosen.png"
        curated = self.root / "docs/assets/validation_prediction.png"
        curated.parent.mkdir(parents=True)
        curated.write_bytes(b"curated figure")
        captured = []
        original_subplots = visualize_prediction.plt.subplots

        def capture_figure(*args, **kwargs):
            figure, axes = original_subplots(*args, **kwargs)
            captured.append((figure, axes))
            return figure, axes

        with mock.patch.object(visualize_prediction, "UNet2D", side_effect=lambda: UNet2D(base_channels=2)), \
                mock.patch.object(visualize_prediction.plt, "subplots", side_effect=capture_figure), \
                mock.patch.object(visualize_prediction.torch, "load", wraps=torch.load) as load:
            try:
                console = self.invoke(visualize_prediction, [
                    "--checkpoint", str(self.checkpoint), "--output", str(output),
                    "--sample-index", "2", "--device", "cpu",
                ])
            except Exception as error:
                self.fail(f"Explicit-checkpoint CPU visualization must work: {error}")
        self.assertEqual(load.call_args.args[0], self.checkpoint)
        self.assertEqual(load.call_args.kwargs, {"map_location": "cpu", "weights_only": True})
        self.assertTrue(output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertEqual(curated.read_bytes(), b"curated figure")
        figure, axes = captured[0]
        dataset = CamusDataset(build_samples(self.camus_root, self.split), (16, 32))
        image, target = dataset[2]
        np.testing.assert_array_equal(axes[0].images[0].get_array(), image.squeeze(0).numpy().T)
        np.testing.assert_array_equal(axes[1].images[0].get_array(), target.numpy().T)
        np.testing.assert_array_equal(axes[2].images[0].get_array(), np.full((32, 16), 2))
        self.assertEqual(figure._suptitle.get_text(), "patient0001 | 4CH | ED")
        expected_dice = visualize_prediction.foreground_dice_scores(
            torch.full_like(target, 2).unsqueeze(0), target.unsqueeze(0),
        )[0].mean()
        self.assertEqual(axes[2].get_title(), f"Predicted mask\nMean foreground Dice: {expected_dice:.3f}")
        self.assertIn("(1, 1, 16, 32)", console)
        self.assertIn("patient0001 4CH ED", console)

    def test_visualization_default_is_a_new_non_overwriting_output_figure(self):
        curated = self.root / "docs/assets/validation_prediction.png"
        curated.parent.mkdir(parents=True)
        curated.write_bytes(b"curated figure")
        arguments = ["--checkpoint", str(self.checkpoint), "--device", "cpu"]
        with mock.patch.object(visualize_prediction, "UNet2D", side_effect=lambda: UNet2D(base_channels=2)):
            self.invoke(visualize_prediction, arguments)
        self.assertEqual(curated.read_bytes(), b"curated figure")
        output = self.root / "outputs/figures/validation_prediction.png"
        self.assertTrue(output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
        with mock.patch.object(visualize_prediction.torch, "load", side_effect=AssertionError("checkpoint accessed")):
            with self.assertRaises(FileExistsError):
                self.invoke(visualize_prediction, arguments)

    def test_visualization_rejects_invalid_indices_before_image_or_model_work(self):
        for value in ("-1", "4", "999"):
            with self.subTest(sample_index=value), \
                    mock.patch.object(visualize_prediction.torch, "load", wraps=torch.load) as load, \
                    mock.patch.object(visualize_prediction, "CamusDataset", side_effect=AssertionError("dataset built")), \
                    mock.patch.object(visualize_prediction, "UNet2D", side_effect=AssertionError("model built")), \
                    contextlib.redirect_stderr(io.StringIO()):
                expected_error = SystemExit if value == "-1" else ValueError
                with self.assertRaises(expected_error):
                    self.invoke(visualize_prediction, [
                        "--checkpoint", str(self.checkpoint), "--sample-index", value, "--device", "cpu",
                    ])
                if value == "-1":
                    load.assert_not_called()


    def test_legacy_defaults_are_preserved_and_selected_checkpoint_is_printed(self):
        legacy_checkpoint = self.root / "outputs/checkpoints/best_model.pt"
        legacy_checkpoint.parent.mkdir(parents=True)
        torch.save({key: value for key, value in self.payload.items() if key != "training_config"}, legacy_checkpoint)
        for module in (evaluate_test, visualize_prediction):
            with self.subTest(module=module.__name__), \
                    mock.patch.object(module, "__file__", str(self.module_path)), \
                    mock.patch("sys.argv", [module.__name__]):
                arguments = module.parse_arguments()
                self.assertEqual(arguments.checkpoint, legacy_checkpoint)
                self.assertEqual(arguments.device, "cuda")
                if module is evaluate_test:
                    self.assertEqual(arguments.split, "test")
                    self.assertIsNone(arguments.batch_size)
                    self.assertIsNone(arguments.output_json)
                else:
                    self.assertEqual(arguments.sample_index, 0)
        with mock.patch.object(evaluate_test, "UNet2D", side_effect=lambda: UNet2D(base_channels=2)), \
                mock.patch.object(evaluate_test, "DataLoader", wraps=DataLoader) as loader, \
                mock.patch.object(evaluate_test, "build_samples", wraps=build_samples) as build:
            output = self.invoke(evaluate_test, ["--device", "cpu"])
        self.assertEqual(loader.call_args.kwargs["batch_size"], 8)
        self.assertEqual(loader.call_args.args[0].image_size, DEFAULT_IMAGE_SIZE)
        self.assertEqual(build.call_args.args[1], self.root / "data/splits/subgroup_testing.txt")
        self.assertIn("Test patients:", output)
        self.assertIn(str(legacy_checkpoint), output)
        self.assertEqual(list((self.root / "outputs").iterdir()), [legacy_checkpoint.parent])

    def test_visualization_cannot_overwrite_a_checkpoint_alias_created_during_inference(self):
        output = self.root / "racing.png"
        before = self.checkpoint.read_bytes()

        def racing_model():
            output.symlink_to(self.checkpoint)
            return UNet2D(base_channels=2)

        with mock.patch.object(visualize_prediction, "UNet2D", side_effect=racing_model):
            with self.assertRaises(FileExistsError):
                self.invoke(visualize_prediction, [
                    "--checkpoint", str(self.checkpoint), "--output", str(output), "--device", "cpu",
                ])
        self.assertEqual(self.checkpoint.read_bytes(), before)
        self.assertEqual(visualize_prediction.plt.get_fignums(), [])

    def test_output_parent_must_be_a_directory_before_checkpoint_load(self):
        parent_file = self.root / "not-a-directory"
        parent_file.write_text("preserve me")
        for module, flag in ((evaluate_test, "--output-json"), (visualize_prediction, "--output")):
            with self.subTest(module=module.__name__), \
                    mock.patch.object(module.torch, "load", side_effect=AssertionError("checkpoint accessed")):
                with self.assertRaises(NotADirectoryError):
                    self.invoke(module, [flag, str(parent_file / "result.png"), "--device", "cpu"])
        self.assertEqual(parent_file.read_text(), "preserve me")

    def test_visualization_legacy_geometry_and_checkpoint_path_remain_visible(self):
        torch.save({key: value for key, value in self.payload.items() if key != "training_config"}, self.checkpoint)
        with mock.patch.object(visualize_prediction, "UNet2D", side_effect=lambda: UNet2D(base_channels=2)), \
                mock.patch.object(visualize_prediction, "CamusDataset", wraps=CamusDataset) as dataset, \
                mock.patch.object(visualize_prediction, "build_samples", wraps=build_samples) as build:
            console = self.invoke(visualize_prediction, ["--checkpoint", str(self.checkpoint), "--device", "cpu"])
        self.assertEqual(dataset.call_args.kwargs["image_size"], DEFAULT_IMAGE_SIZE)
        self.assertEqual(build.call_args.args[1], self.split)
        self.assertIn("patient0001 2CH ED", console)
        self.assertIn(str(self.checkpoint), console)

    def test_nonfinite_metrics_do_not_leave_a_partial_json_report(self):
        with torch.no_grad():
            self.model.segmentation_head.bias.fill_(float("nan"))
        torch.save({**self.payload, "model_state_dict": self.model.state_dict()}, self.checkpoint)
        output = self.root / "invalid.json"
        with mock.patch.object(evaluate_test, "UNet2D", side_effect=lambda: UNet2D(base_channels=2)):
            with self.assertRaises(ValueError):
                self.invoke(evaluate_test, [
                    "--checkpoint", str(self.checkpoint), "--device", "cpu", "--output-json", str(output),
                ])
        self.assertFalse(output.exists())

    def test_malformed_checkpoint_is_rejected_before_data_or_model_work(self):
        payloads = [[], None, {}, {**self.payload, "model_state_dict": []}]
        payloads.append({name: value for name, value in self.payload.items() if name != "epoch"})
        payloads.extend({**self.payload, "epoch": value} for value in (-1, True, "7", None))
        for payload in payloads:
            for module in (evaluate_test, visualize_prediction):
                with self.subTest(payload_type=type(payload).__name__, module=module.__name__):
                    torch.save(payload, self.checkpoint)
                    with mock.patch.object(module, "build_samples", side_effect=AssertionError("data accessed")), \
                            mock.patch.object(module, "UNet2D", side_effect=AssertionError("model built")):
                        with self.assertRaisesRegex(ValueError, "checkpoint|model_state_dict"):
                            self.invoke(module, ["--checkpoint", str(self.checkpoint), "--device", "cpu"])


    def test_visualization_accepts_legacy_checkpoints_without_evaluation_metadata(self):
        payload = {key: value for key, value in self.payload.items() if key != "mean_validation_dice"}
        torch.save(payload, self.checkpoint)
        with mock.patch.object(visualize_prediction, "UNet2D", side_effect=lambda: UNet2D(base_channels=2)):
            try:
                self.invoke(visualize_prediction, ["--checkpoint", str(self.checkpoint), "--device", "cpu"])
            except ValueError as error:
                self.fail(f"Visualization did not previously require a validation score: {error}")
        self.assertTrue((self.root / "outputs/figures/validation_prediction.png").is_file())

    def test_evaluation_requires_finite_validation_score_before_data_work(self):
        payloads = [{key: value for key, value in self.payload.items() if key != "mean_validation_dice"}]
        payloads.extend({**self.payload, "mean_validation_dice": value} for value in (
            None, "0.5", True, float("nan"), float("inf"),
        ))
        for payload in payloads:
            with self.subTest(score=payload.get("mean_validation_dice")):
                torch.save(payload, self.checkpoint)
                with mock.patch.object(evaluate_test, "build_samples", side_effect=AssertionError("data accessed")):
                    with self.assertRaisesRegex(ValueError, "mean_validation_dice"):
                        self.invoke(evaluate_test, ["--checkpoint", str(self.checkpoint), "--device", "cpu"])


if __name__ == "__main__":
    unittest.main()
