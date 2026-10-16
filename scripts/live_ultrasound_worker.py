"""Independent forward-imaging / observation-only inverse workers for Isaac."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from deform_sim.phantom import load_bodies, advect, acoustic_scatterers
from deform_sim.reconstruction import Observation, TrackedVolume, advance_material_tracks
from deform_sim.fast_ultrasound import FastUltrasound


def atomic_npz(path, **data):
    temporary = path.with_name(path.stem+'.tmp.npz')
    np.savez(temporary, **data)
    os.replace(temporary, path)


def atomic_json(path, data):
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(data)+'\n')
    os.replace(temporary, path)


def forward(args):
    reference = dict(np.load(args.episode/'reference.npz'))
    bodies = load_bodies(reference)
    scatter = {key: reference[key] for key in ('body', 'tet', 'weights', 'reflectivity')}
    renderer = FastUltrasound()
    last = None
    destination = args.episode/'live_ultrasound'
    while not (args.episode/'live_complete.json').exists():
        path = destination/'physics.npz'
        if not path.exists() or path.stat().st_mtime_ns == last:
            time.sleep(.05)
            continue
        last = path.stat().st_mtime_ns
        state = np.load(path)
        started = time.monotonic()
        world = advect(bodies, [state[f'positions_{i}'] for i in range(len(bodies))], scatter)
        xyz, coefficients = acoustic_scatterers(world, scatter['reflectivity'], state)
        image = renderer.render(xyz, coefficients)
        identifier = int(state['frame'])
        image_path = destination/f'us_{identifier:08d}.png'
        Image.fromarray((image*255).astype(np.uint8)).save(image_path)
        metadata = dict(frame=identifier, simulation_time=float(state['time']), capture_wall_time=float(state['capture_wall_time']),
            published_wall_time=time.time(), epoch=int(state['epoch']), image=image_path.name,
            position=state['probe_position'].tolist(), rotation=state['probe_rotation'].tolist(),
            image_model='coherent beamformed-IQ PSF preview; not channel RF/SIMUS', render_ms=(time.monotonic()-started)*1000)
        atomic_npz(destination/'observation.npz', image=image, position=state['probe_position'], rotation=state['probe_rotation'],
            time=state['time'], frame=identifier, epoch=state['epoch'], capture_wall_time=state['capture_wall_time'])
        atomic_json(destination/'image.json', metadata)
        old = sorted(destination.glob('us_*.png'))
        for stale in old[:-256]:
            stale.unlink(missing_ok=True)


def inverse(args):
    from deform_sim.inverse_elasticity import ElasticInverse
    from deform_sim.gaussians import pack_gaussian_frame
    initial = dict(np.load(args.prior))
    model = dict(np.load(args.gaussians))
    elastic = ElasticInverse(initial)
    destination = args.episode/'live_ultrasound'
    previous = None
    anchor = None
    stable_start = None
    epoch = None
    volume = None
    last = None
    last_publish = time.monotonic()
    while not (args.episode/'live_complete.json').exists():
        path = destination/'observation.npz'
        if not path.exists() or path.stat().st_mtime_ns == last:
            time.sleep(.05)
            continue
        last = path.stat().st_mtime_ns
        data = np.load(path)
        started = time.monotonic()
        observation = Observation(data['image'], data['position'], data['rotation'], float(data['time']))
        if epoch != int(data['epoch']):
            epoch = int(data['epoch'])
            volume = TrackedVolume([-.04, -.045, .07], [.04, .045, .155], spacing=.0015)
            previous = anchor = None
            stable_start = None
        volume.add(observation)
        identifier = int(data['frame'])
        volume_name = f'volume_{identifier:08d}.bin'
        estimate_name = f'estimate_{identifier:08d}.bin'
        temporary = destination/'volume.tmp.bin'
        volume.splats(limit=25000).tofile(temporary)
        os.replace(temporary, destination/volume_name)
        stable = previous is not None and abs(float((observation.position-previous.position)@observation.rotation[:, 1]))<.0003 and np.dot(observation.rotation[:, 1], previous.rotation[:, 1])>np.cos(np.deg2rad(.5))
        if not stable:
            stable_start = None
            anchor = None
        else:
            if stable_start is None:
                stable_start = observation.time
            if anchor is None and observation.time-stable_start > .5:
                anchor = observation
                rr, cc = np.mgrid[12:180:10, 12:180:10]
                rows, cols = rr.ravel(), cc.ravel()
                selection = anchor.image[rows, cols] > .25
                rows, cols = rows[selection], cols[selection]
                if len(rows) >= 20:
                    directions = anchor.rotation[:, [0, 2]].T
                    material_world = anchor.world_points(rows, cols)
                    cumulative = np.zeros((len(rows), 3))
                    tracking_previous = anchor
                    elastic.prepare(material_world, directions,
                        np.clip((anchor.image[rows, cols]-.2)/.5, .1, 1))
                else:
                    anchor = None
        estimated = [body.points for body in elastic.bodies]
        report = dict(observable=False, message='Hold the scan plane still to estimate incremental tissue motion.')
        if anchor is not None:
            tracks = advance_material_tracks(tracking_previous, observation, material_world,
                elastic.point_displacements(), cumulative, reference=anchor)
            if tracks is not None:
                cumulative, confidence = tracks
                measured = cumulative@directions.T
                estimated, fit = elastic.solve(measured, confidence=confidence)
                tracking_previous = observation
                report = dict(fit, observable=fit['accepted'] and fit['measurement_count']>=20, anchor_time=anchor.time,
                    message='US image motion + elastic prior; normal motion is prior-driven.' if fit['accepted'] else fit['rejection_reason'])
                if fit['min_volume_jacobian'] <= 0:
                    report = dict(observable=False, message='Image-derived estimate rejected: inverted element.')
                    estimated = [body.points for body in elastic.bodies]
        rotation = observation.rotation@np.diag([1., -1., -1.])
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        pose = np.r_[observation.position, 0, 0, np.sin(yaw/2), np.cos(yaw/2)]
        header = [1, observation.time, int(data['frame']), len(model['means']), *observation.position, yaw, 0, 0]
        packet, info = pack_gaussian_frame(model, estimated, pose, header)
        temporary = destination/'estimate.tmp.bin'
        temporary.write_bytes(packet)
        os.replace(temporary, destination/estimate_name)
        published = time.monotonic()
        metadata = dict(frame=int(data['frame']), simulation_time=observation.time, epoch=epoch,
            capture_wall_time=float(data['capture_wall_time']), published_wall_time=time.time(),
            image=f'us_{int(data["frame"]):08d}.png', inference_ms=(time.monotonic()-started)*1000,
            update_hz=1/max(1e-6, published-last_publish),
            estimate_file=estimate_name, volume_file=volume_name,
            volume_frames=volume.frames, volume_spacing_m=volume.spacing, **report, **info)
        atomic_json(destination/'estimate.json', metadata)
        last_publish = published
        for pattern in ['estimate_*.bin', 'volume_*.bin']:
            for stale in sorted(destination.glob(pattern))[:-8]:
                stale.unlink(missing_ok=True)
        previous = observation


def main():
    p = argparse.ArgumentParser()
    p.add_argument('episode', type=Path)
    p.add_argument('--mode', choices=['forward', 'inverse'], required=True)
    p.add_argument('--prior', type=Path)
    p.add_argument('--gaussians', type=Path)
    args = p.parse_args()
    (args.episode/'live_ultrasound').mkdir(exist_ok=True)
    (forward if args.mode == 'forward' else inverse)(args)


if __name__ == '__main__':
    main()
