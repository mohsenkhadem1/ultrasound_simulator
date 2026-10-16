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
from deform_sim.reconstruction import Observation, TrackedVolume, advance_material_tracks, advance_material_tracks_lk
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
    renderer = FastUltrasound(size=256, depth_max=.08)
    last = None
    destination = args.episode/'live_ultrasound'
    truth_dir = args.episode/'comparison_truth'
    truth_dir.mkdir(exist_ok=True)
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
        # Evaluation-only state, never included in observation.npz or read by the inverse.
        atomic_npz(truth_dir/f'truth_{identifier:08d}.npz', **dict(state))
        image_path = destination/f'us_{identifier:08d}.png'
        Image.fromarray((image*255).astype(np.uint8)).save(image_path)
        metadata = dict(frame=identifier, simulation_time=float(state['time']), capture_wall_time=float(state['capture_wall_time']),
            published_wall_time=time.time(), epoch=int(state['epoch']), image=image_path.name,
            position=state['probe_position'].tolist(), rotation=state['probe_rotation'].tolist(),
            image_model='coherent beamformed-IQ PSF preview; not channel RF/SIMUS', render_ms=(time.monotonic()-started)*1000)
        metadata['calibration'] = renderer.calibration
        atomic_npz(destination/'observation.npz', image=image, position=state['probe_position'], rotation=state['probe_rotation'],
            time=state['time'], frame=identifier, epoch=state['epoch'], capture_wall_time=state['capture_wall_time'], **renderer.calibration)
        if bool(state.get('record', False)):
            recording = args.episode/'recording'
            (recording/'observations').mkdir(parents=True, exist_ok=True)
            (recording/'truth').mkdir(exist_ok=True)
            atomic_npz(recording/'observations'/f'{identifier:08d}.npz', image=image,
                position=state['probe_position'], rotation=state['probe_rotation'],time=state['time'],
                frame=identifier,epoch=state['epoch'],capture_wall_time=state['capture_wall_time'], **renderer.calibration)
            atomic_npz(recording/'truth'/f'{identifier:08d}.npz', **dict(state))
        atomic_json(destination/'image.json', metadata)
        old = sorted(destination.glob('us_*.png'))
        for stale in old[:-256]:
            stale.unlink(missing_ok=True)
        for stale in sorted(truth_dir.glob('truth_*.npz'))[:-256]:
            stale.unlink(missing_ok=True)


def inverse(args):
    from deform_sim.inverse_elasticity import ElasticInverse
    from deform_sim.gaussians import pack_gaussian_frame
    initial = dict(np.load(args.prior))
    model = dict(np.load(args.gaussians))
    elastic = ElasticInverse(initial, data_weight=args.inverse_data_weight)
    tracker = advance_material_tracks_lk if args.tracker == 'lk' else advance_material_tracks
    destination = args.episode/'live_ultrasound'
    prediction_dir = args.episode/'comparison_predictions'
    prediction_dir.mkdir(exist_ok=True)
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
        calibration = {key: float(data[key]) for key in ('lateral_min', 'lateral_max', 'depth_min', 'depth_max') if key in data}
        observation = Observation(data['image'], data['position'], data['rotation'], float(data['time']), **calibration)
        if epoch != int(data['epoch']):
            epoch = int(data['epoch'])
            height = float(initial.get('probe_height', .144))
            volume = TrackedVolume([-.04, -.045, height-.085], [.04, .045, height+.015], spacing=.0015)
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
                rr, cc = np.mgrid[12:observation.image.shape[0]-12:16, 12:observation.image.shape[1]-12:16]
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
            tracks = tracker(tracking_previous, observation, material_world,
                elastic.point_displacements(), cumulative, reference=anchor)
            if tracks is not None:
                proposed, confidence = tracks
                measured = proposed@directions.T
                estimated, fit = elastic.solve(measured, confidence=confidence)
                if fit['accepted']:
                    cumulative = proposed
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
        atomic_npz(prediction_dir/f'prediction_{identifier:08d}.npz',
            positions=np.concatenate(estimated), frame=identifier, epoch=epoch)
        published = time.monotonic()
        metadata = dict(frame=int(data['frame']), simulation_time=observation.time, epoch=epoch,
            capture_wall_time=float(data['capture_wall_time']), published_wall_time=time.time(),
            image=f'us_{int(data["frame"]):08d}.png', inference_ms=(time.monotonic()-started)*1000,
            update_hz=1/max(1e-6, published-last_publish),
            estimate_file=estimate_name, volume_file=volume_name,
            volume_frames=volume.frames, volume_spacing_m=volume.spacing, calibration=calibration,
            image_shape=list(observation.image.shape), tracking_method=args.tracker,
            inverse_data_weight=args.inverse_data_weight, **report, **info)
        atomic_json(destination/'estimate.json', metadata)
        last_publish = published
        for pattern in ['estimate_*.bin', 'volume_*.bin']:
            for stale in sorted(destination.glob(pattern))[:-8]:
                stale.unlink(missing_ok=True)
        previous = observation
        for stale in sorted(prediction_dir.glob('prediction_*.npz'))[:-32]:
            stale.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('episode', type=Path)
    p.add_argument('--mode', choices=['forward', 'inverse', 'comparison'], required=True)
    p.add_argument('--prior', type=Path)
    p.add_argument('--gaussians', type=Path)
    p.add_argument('--tracker', choices=['lk', 'tvl1'], default='lk')
    p.add_argument('--inverse-data-weight', type=float, default=10.)
    args = p.parse_args()
    if not np.isfinite(args.inverse_data_weight) or args.inverse_data_weight <= 0:
        p.error('--inverse-data-weight must be finite and positive')
    (args.episode/'live_ultrasound').mkdir(exist_ok=True)
    if args.mode == 'comparison':
        from deform_sim.comparison import run_comparison
        run_comparison(args)
    else:
        (forward if args.mode == 'forward' else inverse)(args)


if __name__ == '__main__':
    main()
