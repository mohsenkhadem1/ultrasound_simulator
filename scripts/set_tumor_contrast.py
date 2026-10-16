"""Create an imaging variant of recorded mechanics with stronger tumour echoes.

No image pixels are painted; RF and IQ are regenerated from modified scatterers.
"""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('episode', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tumor-backscatter', type=float, default=6.)
    args = p.parse_args()
    if args.output.exists():
        p.error('choose a new output directory')
    if not (args.episode/'complete.json').exists():
        p.error('source mechanics must be complete')
    if not np.isfinite(args.tumor_backscatter) or args.tumor_backscatter <= 0:
        p.error('backscatter must be finite and positive')
    reference = dict(np.load(args.episode/'reference.npz'))
    if 'scatter_amplitudes' not in reference or int(reference.get('body_count', 2)) < 2:
        p.error('source must record the material scatterer amplitudes')
    amplitudes = reference['scatter_amplitudes'].copy()
    old = float(amplitudes[1])
    if old <= 0:
        p.error('source tumour amplitude must be positive')
    coefficients = reference['reflectivity'].copy()
    tumor = reference['body'] == 1
    if not tumor.any():
        p.error('source has no tumour scatterers')
    coefficients[tumor] *= args.tumor_backscatter/old
    amplitudes[1] = args.tumor_backscatter
    reference.update(reflectivity=coefficients, scatter_amplitudes=amplitudes)
    args.output.mkdir(parents=True)
    for name in ['state', 'scene_frames']:
        if (args.episode/name).exists():
            shutil.copytree(args.episode/name, args.output/name)
    for name in ['scene.usda', 'complete.json', 'validation.json', 'anatomy_source.json', 'tumor_press.json']:
        if (args.episode/name).exists():
            shutil.copy2(args.episode/name, args.output/name)
    np.savez_compressed(args.output/'reference.npz', **reference)
    metadata = json.loads((args.episode/'metadata.json').read_text())
    change = dict(source_episode=str(args.episode.resolve()), region=1,
                  previous_backscatter_amplitude=old, new_backscatter_amplitude=args.tumor_backscatter,
                  amplitude_ratio=args.tumor_backscatter/old,
                  change_db=float(20*np.log10(args.tumor_backscatter/old)),
                  meaning='Illustrative ultrasound scattering contrast; not a measured mass density or patient calibration.',
                  mechanics='Recorded geometry, mechanics and probe poses copied unchanged.')
    metadata.update(scatter_amplitudes=amplitudes.tolist(), acoustic_contrast=change)
    (args.output/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    (args.output/'contrast.json').write_text(json.dumps(change, indent=2)+'\n')
    print(json.dumps(change, indent=2))


if __name__ == '__main__':
    main()
