"""Plot recorded epoch metrics; never reconstruct missing historical curves."""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_history(run_dir: Path, output: Path | None = None, *, overwrite: bool = False) -> Path:
    """Show loss, mean Dice, and class Dice for this recorded run segment."""
    output = output or run_dir / 'curves.png'
    if output.suffix.lower() != '.png':
        raise ValueError('Plot output must be a .png file, not a data or checkpoint file.')
    if output.is_symlink():
        raise ValueError('Refusing to write a figure through a symbolic link.')
    if output.exists() and not overwrite:
        raise FileExistsError(f'{output} already exists; use --overwrite to replace the figure.')
    with (run_dir / 'history.csv').open(newline='') as stream:
        rows = [{key: float(value) for key, value in row.items()}
                for row in csv.DictReader(stream)]

    if not rows:
        raise ValueError('history.csv has no completed epochs to plot.')
    previous_epoch = previous_step = 0
    for row in rows:
        if not all(math.isfinite(value) and value >= 0 for value in row.values()):
            raise ValueError('History values must be finite and nonnegative.')
        if any(not 0 <= row[key] <= 1 for key in
               ('val_dice_lv', 'val_dice_myo', 'val_dice_la', 'val_mean_dice')):
            raise ValueError('Dice values must be in [0, 1].')
        epoch, step = row['epoch'], row['optimizer_steps']
        if not epoch.is_integer() or not step.is_integer():
            raise ValueError('Epoch and update counts must be integers.')
        if epoch <= previous_epoch or step <= previous_step:
            raise ValueError('Epochs and update counts must strictly increase.')
        previous_epoch, previous_step = epoch, step
    epochs = [row['epoch'] for row in rows]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for key, label in [('train_loss', 'Training'), ('val_loss', 'Validation')]:
        axes[0].plot(epochs, [row[key] for row in rows], marker='.', label=label)
    axes[0].set(title='Loss', ylabel='Cross-entropy + foreground soft Dice loss')
    axes[1].plot(epochs, [row['val_mean_dice'] for row in rows], marker='.', label='Validation')
    axes[1].set(title='Mean foreground Dice', ylabel='Validation Dice')
    for key, label in [('val_dice_lv', 'LV cavity'), ('val_dice_myo', 'Myocardium'),
                       ('val_dice_la', 'Left atrium')]:
        axes[2].plot(epochs, [row[key] for row in rows], marker='.', label=label)
    axes[2].set(title='Per-class Dice', ylabel='Validation Dice')
    baseline_path = run_dir / 'baseline.json'
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text())
        axes[0].scatter([baseline['epoch']], [baseline['validation_loss']],
                        marker='D', color='black', label='Saved baseline (validation)')
        axes[1].scatter([baseline['epoch']], [baseline['mean_validation_dice']],
                        marker='D', color='black', label='Saved baseline')
        for index, score in enumerate(baseline['validation_dice']):
            axes[2].scatter([baseline['epoch']], [score], marker='D', color=f'C{index}')
    config_path = run_dir / 'config.json'
    smoke = config_path.exists() and json.loads(config_path.read_text()).get('smoke_test', False)
    for axis in axes:
        axis.set_xlabel('Attempted epoch (partial batch only)' if smoke else 'Completed epoch')
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    scope = 'SMOKE TEST: partial batch only, not a full epoch' if smoke else 'Recorded segment only'
    figure.suptitle(f'{run_dir.name}\n{scope}; no interpolation of missing earlier epochs',
                   fontsize=11)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--overwrite', action='store_true', help='Replace an existing PNG figure.')
    args = parser.parse_args()
    print(plot_history(args.run_dir, args.output, overwrite=args.overwrite))


if __name__ == '__main__':
    main()
