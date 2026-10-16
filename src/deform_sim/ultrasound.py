"""SIMUS acoustic RF simulation, coherent DAS reconstruction, B-mode display.

Runs outside Isaac's environment to isolate its pinned scientific dependencies.
"""
import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
import pymust

from .phantom import BodyMesh, advect, acoustic_scatterers, load_bodies


@dataclass
class Acquisition:
    elements: int = 128
    frequency: float = 5e6
    pitch: float = 0.0003
    element_width: float = 0.00027
    element_height: float = 0.005
    elevation_focus: float = 0.02
    sound_speed: float = 1540.0
    attenuation: float = 0.5
    angles_deg: tuple = (-6.0, 0.0, 6.0)
    width_pixels: int = 192
    depth_pixels: int = 192
    dynamic_range_db: float = 60.0
    depth_min: float = .002
    depth_max: float = .03
    elevation_limit: float = 0.


class Ultrasound:
    def __init__(self, config):
        self.c = config
        p = pymust.utils.Param()
        p.Nelements = config.elements
        p.fc = config.frequency
        p.fs = 4*p.fc
        p.pitch = config.pitch
        p.width = config.element_width
        p.height = config.element_height
        p.focus = config.elevation_focus
        p.bandwidth = 70.0
        p.c = config.sound_speed
        p.attenuation = config.attenuation
        p.radius = np.inf
        p.TXnow = 2
        p.fnumber = 1.5
        self.param = p
        self.x, self.z = np.meshgrid(np.linspace(-0.02, 0.02, config.width_pixels),
                                     np.linspace(config.depth_min, config.depth_max, config.depth_pixels))

    def render(self, xyz, reflectivity):
        if not np.all(np.isfinite(xyz)) or np.any(xyz[:, 2] <= 0):
            raise ValueError('scatterers must be finite and in front of fixed probe')
        if self.c.elevation_limit:
            keep = np.abs(xyz[:, 1]) <= self.c.elevation_limit
            xyz, reflectivity = xyz[keep], reflectivity[keep]
        iq_images, rf_signals = [], []
        for angle in self.c.angles_deg:
            delays = pymust.txdelay(self.param, np.deg2rad(angle))
            options = pymust.utils.Options()
            options.ParPool = False
            options.dBThresh = -60
            rf, _ = pymust.simus(xyz[:, 0], xyz[:, 1], xyz[:, 2], reflectivity,
                                 delays, self.param, options)
            iq = pymust.rf2iq(rf, self.param)
            matrix = pymust.dasmtx(iq, self.x, self.z, delays, self.param)
            beamformed = (matrix @ iq.flatten(order='F')).reshape(self.x.shape, order='F')
            # Fixed analytic time gain compensation, independent of observed anatomy.
            gain = 10**(2*self.c.attenuation*(self.c.frequency/1e6)*(self.z*100)/20)
            iq_images.append(beamformed*gain)
            rf_signals.append(rf.astype(np.float32))
        return np.mean(iq_images, axis=0), rf_signals


def log_compress(iq, reference, dynamic_range):
    envelope = np.abs(iq)
    db = 20*np.log10(np.maximum(envelope, 1e-30)/max(reference, 1e-30))
    return np.clip((db+dynamic_range)/dynamic_range*255, 0, 255).astype(np.uint8)


def sequence_display(output):
    """One shared gain for a scan that can begin mostly outside the tissue."""
    files = sorted(p for p in output.glob('*.npz') if p.stem.isdecimal())
    gain = max(float(np.abs(np.load(path)['iq']).max()) for path in files)
    config = json.loads((output/'acquisition.json').read_text())
    images, timestamps = [], []
    for path in files:
        data = np.load(path)
        image = Image.fromarray(log_compress(data['iq'], gain, config['dynamic_range_db']))
        image.save(output/(path.stem+'.png'))
        images.append(image)
        timestamps.append(float(data['time']))
    (output/'display.json').write_text(json.dumps(dict(reference_amplitude=gain, source='sequence_peak',
        note='One constant gain across the entire scan; raw RF and IQ unchanged.'))+'\n')
    duration = round(1000*np.median(np.diff(timestamps))) if len(timestamps) > 1 else 100
    images[0].save(output/'sequence.gif', save_all=True, append_images=images[1:], duration=duration, loop=0)


def iter_snapshots(episode, watch=False, limit=0):
    """Drain frames written during a slow render before accepting completion."""
    seen = set()
    while True:
        available = sorted(p for p in (episode/'state').glob('*.npz') if p.stem.isdecimal())
        pending = [path for path in available if path not in seen]
        if not pending:
            if not watch or (episode/'complete.json').exists():
                return
            time.sleep(0.1)
            continue
        for path in pending:
            seen.add(path)
            yield path
            if limit and len(seen) >= limit:
                return
        if not watch:
            return


def main():
    parser = argparse.ArgumentParser(description='Render Isaac tissue snapshots using acoustic RF simulation')
    parser.add_argument('episode', type=Path)
    parser.add_argument('--output', type=Path, help='new render directory; defaults to EPISODE/ultrasound')
    parser.add_argument('--quick', action='store_true', help='16 elements, one angle, 64x64 for integration checks')
    parser.add_argument('--preview', action='store_true', help='64 elements, one angle; faster acoustic preview')
    parser.add_argument('--frames', type=int, default=0, help='0 means all recorded frames')
    parser.add_argument('--watch', action='store_true', help='render snapshots as Isaac writes them')
    args = parser.parse_args()
    reference = np.load(args.episode/'reference.npz')
    bodies = load_bodies(reference)
    scatter = {k: reference[k] for k in ('body', 'tet', 'weights', 'reflectivity')}
    config = Acquisition(depth_max=.05 if 'scatter_bounds' in reference else .03)
    config.elevation_limit = float(reference.get('acoustic_elevation_limit', 0.))
    if args.quick:
        config.elements, config.angles_deg = 16, (0.0,)
        config.width_pixels = config.depth_pixels = 64
    elif args.preview:
        config.elements, config.angles_deg = 64, (0.0,)
    renderer = Ultrasound(config)
    output = args.output or args.episode/'ultrasound'
    output.mkdir(exist_ok=False)
    (output/'acquisition.json').write_text(json.dumps(asdict(config), indent=2)+'\n')
    reference_gain = None
    frames = []
    for snapshot in iter_snapshots(args.episode, args.watch, args.frames):
        start = time.monotonic()
        state = np.load(snapshot)
        world = advect(bodies, [state[f'positions_{i}'] for i in range(len(bodies))], scatter)
        xyz, coefficients = acoustic_scatterers(world, scatter['reflectivity'], state)
        iq, rfs = renderer.render(xyz, coefficients)
        if reference_gain is None:
            reference_gain = float(np.max(np.abs(iq)))
            (output/'display.json').write_text(json.dumps(dict(reference_amplitude=reference_gain, source='first_frame'))+'\n')
        image = Image.fromarray(log_compress(iq, reference_gain, config.dynamic_range_db))
        image.save(output/f'{snapshot.stem}.png')
        np.savez_compressed(output/f'{snapshot.stem}.npz', iq=iq, x=renderer.x, z=renderer.z,
                            time=state['time'], command=state['command'],
                            **{k: state[k] for k in ('probe_position', 'probe_rotation') if k in state},
                            **{f'rf_{i}': rf for i, rf in enumerate(rfs)})
        frames.append(image)
        print(f'{snapshot.stem}: RF -> IQ -> DAS -> B-mode, {time.monotonic()-start:.1f}s', flush=True)
    if not frames:
        raise ValueError('no Isaac state frames found')
    frames[0].save(output/'sequence.gif', save_all=True, append_images=frames[1:], duration=120, loop=0)
    metadata = json.loads((args.episode/'metadata.json').read_text())
    if metadata.get('probe_motion') == 'scan':
        sequence_display(output)


if __name__ == '__main__':
    main()
