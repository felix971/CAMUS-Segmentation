"""Plotting tests use explicitly synthetic data, never research results."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ['epoch', 'optimizer_steps', 'learning_rate', 'train_loss', 'val_loss',
          'val_dice_lv', 'val_dice_myo', 'val_dice_la', 'val_mean_dice', 'epoch_seconds']


class PlotHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run_dir = Path(self.temp.name)
        self.rows = [dict(zip(FIELDS, [11, 2200, .001, .3, .4, .9, .8, .7, .8, 5])),
                     dict(zip(FIELDS, [12, 2400, .001, .2, .3, .93, .83, .73, .83, 6]))]
        self.write_history()
        (self.run_dir / 'baseline.json').write_text(json.dumps({
            'epoch': 10, 'validation_loss': .45, 'validation_dice': [.87, .77, .67],
            'mean_validation_dice': .77}))

    def write_history(self):
        with (self.run_dir / 'history.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)

    def command(self, *extra):
        return subprocess.run([sys.executable, '-m', 'camus_segmentation.plot_history',
                               '--run-dir', str(self.run_dir), *extra], cwd=ROOT,
                              capture_output=True, text=True)

    def test_cli_plots_recorded_history_without_changing_inputs(self):
        history = (self.run_dir / 'history.csv').read_bytes()
        baseline = (self.run_dir / 'baseline.json').read_bytes()
        result = self.command()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.run_dir / 'curves.png').read_bytes()[:8], b'\x89PNG\r\n\x1a\n')
        self.assertEqual((self.run_dir / 'history.csv').read_bytes(), history)
        self.assertEqual((self.run_dir / 'baseline.json').read_bytes(), baseline)

    def test_rejects_invalid_or_empty_recorded_history(self):
        invalid_rows = [[], [dict(self.rows[0], val_mean_dice='nan')],
                        [dict(self.rows[0], val_dice_lv=1.5)],
                        [self.rows[0], self.rows[0]],
                        [dict(self.rows[0], epoch=11.5)],
                        [dict(self.rows[0], epoch_seconds=-1)]]
        for rows in invalid_rows:
            with self.subTest(rows=rows):
                self.rows = rows
                self.write_history()
                result = self.command()
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((self.run_dir / 'curves.png').exists())

    def test_output_collision_requires_explicit_overwrite(self):
        output = self.run_dir / 'curves.png'
        output.write_bytes(b'KEEP EXISTING FIGURE')
        result = self.command()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(output.read_bytes(), b'KEEP EXISTING FIGURE')
        result = self.command('--overwrite')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(output.read_bytes()[:8], b'\x89PNG\r\n\x1a\n')

    def test_output_cannot_replace_history_file(self):
        history = self.run_dir / 'history.csv'
        original = history.read_bytes()
        result = self.command('--output', str(history), '--overwrite')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(history.read_bytes(), original)

    def test_single_smoke_measurement_is_visible_and_labeled_partial(self):
        from unittest.mock import patch
        from camus_segmentation.plot_history import plot_history
        self.rows = self.rows[:1]
        self.write_history()
        (self.run_dir / 'config.json').write_text(json.dumps({'smoke_test': True}))
        with patch('matplotlib.figure.Figure.savefig', autospec=True) as save:
            plot_history(self.run_dir)
        figure = save.call_args.args[0]
        for axis in figure.axes:
            for line in axis.lines:
                self.assertNotIn(line.get_marker(), (None, 'None', '', ' '))
        self.assertIn('SMOKE', figure._suptitle.get_text())
        self.assertIn('partial', figure._suptitle.get_text())


if __name__ == '__main__':
    unittest.main()
