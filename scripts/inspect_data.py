"""Validate generated NPZ dataset splits and write an approval report."""

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Inspect generated dataset tiles.")
    parser.add_argument('--dataset-dir', default='outputs/dataset')
    parser.add_argument('--output', default='outputs/dataset_inspection_report.json')
    parser.add_argument('--config', default=None)
    args = parser.parse_args()

    quality_cfg = {}
    if args.config:
        quality_cfg = load_config(args.config).get('dataset_quality', {})

    counts = {split: 0 for split in ('train', 'val', 'test')}
    groups = {split: set() for split in counts}
    inputs = {split: {} for split in counts}
    nan_count = inf_count = invalid_masks = empty_tiles = positive_tiles = hard_negative_tiles = 0
    valid_pixel_count = center_positive_pixel_count = 0
    failures = []
    for split in counts:
        split_dir = os.path.join(args.dataset_dir, split)
        if not os.path.isdir(split_dir):
            failures.append(f'missing split directory: {split}')
            continue
        for filename in sorted(name for name in os.listdir(split_dir) if name.endswith('.npz')):
            counts[split] += 1
            with np.load(os.path.join(split_dir, filename), allow_pickle=False) as tile:
                image = tile['image']
                row_mask, center_mask = tile['row_mask'], tile['center_mask']
                orientation = np.stack((tile['orientation_sin'], tile['orientation_cos']))
                valid_mask = tile['valid_mask'] if 'valid_mask' in tile else np.ones_like(center_mask)
                input_name = str(tile['input_name'].item()) if 'input_name' in tile else filename.split('_tile_')[0]
                group_id = str(tile['group_id'].item()) if 'group_id' in tile else 'unknown'
                groups[split].add((input_name, group_id))
                inputs[split][input_name] = inputs[split].get(input_name, 0) + 1
                nan_count += int(np.isnan(image).sum() + np.isnan(orientation).sum())
                inf_count += int(np.isinf(image).sum() + np.isinf(orientation).sum())
                valid = image.ndim == 3 and image.shape[0] in (3, 4) and row_mask.shape == center_mask.shape
                valid &= set(np.unique(row_mask)).issubset({0, 1}) and set(np.unique(center_mask)).issubset({0, 1})
                if not valid:
                    invalid_masks += 1
                if row_mask.any() or center_mask.any():
                    positive_tiles += 1
                else:
                    empty_tiles += 1
                    sampling_class = str(tile['sampling_class'].item()) if 'sampling_class' in tile else 'unknown'
                    hard_negative_tiles += int(sampling_class == 'hard_negative')
                valid_pixels = valid_mask > 0
                valid_pixel_count += int(valid_pixels.sum())
                center_positive_pixel_count += int(
                    ((center_mask > 0) & valid_pixels).sum()
                )
    leakage = bool(groups['train'] & groups['val'] or groups['train'] & groups['test'] or groups['val'] & groups['test'])
    gsd_values = []
    report_csv = os.path.join(os.path.dirname(args.dataset_dir), 'dataset_gsd_report.csv')
    if os.path.exists(report_csv):
        with open(report_csv, newline='', encoding='utf-8') as handle:
            gsd_values = [float(row['output_gsd_m']) for row in csv.DictReader(handle)]
    minimums = {
        'train': int(quality_cfg.get('min_train_tiles', 1)),
        'val': int(quality_cfg.get('min_val_tiles', 1)),
        'test': int(quality_cfg.get('min_test_tiles', 1)),
    }
    for split, minimum in minimums.items():
        if counts[split] < minimum:
            failures.append(
                f'{split} has {counts[split]} tiles; minimum required is {minimum}'
            )
    source_names = set().union(*(set(value) for value in inputs.values()))
    minimum_sources = int(quality_cfg.get('min_source_images', 1))
    if len(source_names) < minimum_sources:
        failures.append(
            f'dataset has {len(source_names)} source images; minimum required is {minimum_sources}'
        )
    minimum_negatives = int(quality_cfg.get('min_hard_negative_tiles', 0))
    if hard_negative_tiles < minimum_negatives:
        failures.append(
            f'dataset has {hard_negative_tiles} hard-negative tiles; minimum required is {minimum_negatives}'
        )
    center_positive_fraction = (
        center_positive_pixel_count / valid_pixel_count if valid_pixel_count else 0.0
    )
    background_fraction = 1.0 - center_positive_fraction
    minimum_background = float(quality_cfg.get('min_background_pixel_fraction', 0.0))
    if background_fraction < minimum_background:
        failures.append(
            f'dataset background fraction is {background_fraction:.4f}; '
            f'minimum required is {minimum_background:.4f}'
        )
    approved = all(counts.values()) and not failures and not (nan_count or inf_count or invalid_masks or leakage)
    report = {
        'status': 'approved' if approved else 'rejected',
        'train_tiles': counts['train'], 'val_tiles': counts['val'], 'test_tiles': counts['test'],
        'nan_count': nan_count, 'inf_count': inf_count, 'invalid_masks': invalid_masks,
        'group_leakage': leakage, 'empty_tiles': empty_tiles, 'positive_tiles': positive_tiles,
        'hard_negative_tiles': hard_negative_tiles,
        'valid_pixel_count': valid_pixel_count,
        'center_positive_pixel_count': center_positive_pixel_count,
        'center_positive_pixel_fraction': center_positive_fraction,
        'background_pixel_fraction': background_fraction,
        'distribution_by_input': inputs, 'distribution_by_group': {key: len(value) for key, value in groups.items()},
        'gsd_min_m': min(gsd_values) if gsd_values else None,
        'gsd_max_m': max(gsd_values) if gsd_values else None,
        'gsd_median_m': float(np.median(gsd_values)) if gsd_values else None,
        'failures': failures,
    }
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    if not approved:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
