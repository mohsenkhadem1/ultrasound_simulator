"""Render independent Isaac snapshots concurrently using the SIMUS RF model."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import multiprocessing
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from PIL import Image
from deform_sim.phantom import BodyMesh, advect, acoustic_scatterers, load_bodies
from deform_sim.ultrasound import Acquisition, Ultrasound, log_compress, sequence_display


def initialize(reference_path):
    global bodies, scatter, renderer
    reference = np.load(reference_path)
    bodies = load_bodies(reference)
    scatter = {k: reference[k] for k in ('body', 'tet', 'weights', 'reflectivity')}
    renderer = Ultrasound(Acquisition(elements=64, angles_deg=(0.,), depth_max=.05 if 'scatter_bounds' in reference else .03,
                                    elevation_limit=float(reference.get('acoustic_elevation_limit', 0.))))


def render_snapshot(path):
    state = np.load(path)
    world = advect(bodies, [state[f'positions_{i}'] for i in range(len(bodies))], scatter)
    xyz, coefficients = acoustic_scatterers(world, scatter['reflectivity'], state)
    iq, rf = renderer.render(xyz, coefficients)
    pose = {k: state[k] for k in ('probe_position', 'probe_rotation') if k in state}
    return path.stem, float(state['time']), float(state['command']), iq, rf, pose


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('episode', type=Path)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    files = sorted(p for p in (args.episode/'state').glob('*.npz') if p.stem.isdecimal())
    if not files or not (args.episode/'complete.json').exists():
        parser.error('a complete Isaac episode is required')
    output = args.episode/'ultrasound'
    output.mkdir(exist_ok=False)
    reference = np.load(args.episode/'reference.npz')
    config = Acquisition(elements=64, angles_deg=(0.,), depth_max=.05 if 'scatter_bounds' in reference else .03,
                         elevation_limit=float(reference.get('acoustic_elevation_limit', 0.)))
    (output/'acquisition.json').write_text(json.dumps(asdict(config), indent=2)+'\n')
    gain = None
    images = []
    timestamps = []
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'), initializer=initialize, initargs=(args.episode/'reference.npz',)) as pool:
        for index, (stem, timestamp, command, iq, rf, pose) in enumerate(pool.map(render_snapshot, files)):
            if gain is None:
                gain = float(np.max(np.abs(iq)))
                (output/'display.json').write_text(json.dumps(dict(reference_amplitude=gain, source='first_frame'))+'\n')
            image = Image.fromarray(log_compress(iq, gain, config.dynamic_range_db))
            image.save(output/f'{stem}.png')
            np.savez_compressed(output/f'{stem}.npz', iq=iq, time=timestamp, command=command, **pose, **{f'rf_{i}': r for i, r in enumerate(rf)})
            images.append(image)
            timestamps.append(timestamp)
            print(f'{index+1}/{len(files)} acoustic frames, elapsed {time.monotonic()-started:.1f}s', flush=True)
    duration = round(1000*np.median(np.diff(timestamps))) if len(timestamps) > 1 else 100
    images[0].save(output/'sequence.gif', save_all=True, append_images=images[1:], duration=duration, loop=0)
    metadata = json.loads((args.episode/'metadata.json').read_text())
    if metadata.get('probe_motion') == 'scan':
        sequence_display(output)


if __name__ == '__main__':
    main()
