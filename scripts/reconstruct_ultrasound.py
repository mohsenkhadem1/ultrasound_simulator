"""Reconstruct tracked ultrasound and infer incremental motion from image data."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from deform_sim.reconstruction import Observation, TrackedVolume, advance_material_tracks
from deform_sim.inverse_elasticity import ElasticInverse
from deform_sim.gaussians import advect_gaussians


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ultrasound', type=Path, default=Path('outputs/tumor_scan_contrast/ultrasound'))
    p.add_argument('--prior', type=Path, default=Path('data/remind-001/anatomy.npz'))
    p.add_argument('--gaussians', type=Path, default=Path('data/remind-001/gaussians.npz'))
    p.add_argument('--output', type=Path, default=Path('outputs/ultrasound_reconstruction'))
    args = p.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    started = time.perf_counter()
    config = json.loads((args.ultrasound/'acquisition.json').read_text())
    gain = json.loads((args.ultrasound/'display.json').read_text())['reference_amplitude']
    observations, hashes = [], []
    for path in sorted(args.ultrasound.glob('*.npz')):
        if not path.stem.isdecimal():
            continue
        data = np.load(path)
        envelope = np.abs(data['iq'])
        image = np.clip((20*np.log10(np.maximum(envelope, 1e-30)/gain)+config['dynamic_range_db'])/config['dynamic_range_db'], 0, 1).astype(np.float32)
        observations.append(Observation(image, data['probe_position'], data['probe_rotation'], float(data['time']),
            depth_min=config['depth_min'], depth_max=config['depth_max']))
        hashes.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    if len(observations) < 6:
        raise ValueError('at least six calibrated ultrasound frames are required')
    # Hold-out intensity values are excluded from the validation reconstruction.
    heldout = [i for i in range(len(observations)) if i % 5 == 2]
    validation = TrackedVolume.from_observations(observations)
    for i, observation in enumerate(observations):
        if i not in heldout:
            validation.add(observation)
    metrics = []
    for i in heldout:
        observation = observations[i]
        predicted, confidence = validation.sample(observation.world_points())
        known = confidence >= .08
        foreground = known&(observation.image > .15)
        mse = float(np.mean((predicted[foreground]-observation.image[foreground])**2)) if foreground.any() else None
        Image.fromarray((predicted*255).astype(np.uint8)).save(args.output/f'heldout_{i:05d}.png')
        metrics.append(dict(frame=i, coverage_fraction=float(known.mean()), foreground_pixels=int(foreground.sum()),
            foreground_mae=float(np.abs(predicted[foreground]-observation.image[foreground]).mean()) if foreground.any() else None,
            foreground_psnr_db=float(-10*np.log10(max(mse, 1e-12))) if mse is not None else None))
    volume = TrackedVolume.from_observations(observations)
    frames = []
    for i, observation in enumerate(observations):
        volume.add(observation)
        volume.splats().tofile(args.output/f'volume_{i:05d}.bin')
        Image.fromarray((observation.image*255).astype(np.uint8)).save(args.output/f'us_{i:05d}.png')
        prediction, coverage = validation.sample(observation.world_points())
        if i in heldout:
            Image.fromarray((prediction*255).astype(np.uint8)).save(args.output/f'reslice_{i:05d}.png')
        else:
            full_temp = volume.sample(observation.world_points())[0]
            Image.fromarray((full_temp*255).astype(np.uint8)).save(args.output/f'reslice_{i:05d}.png')
        frames.append(dict(index=i, time=observation.time, position=observation.position.tolist(),
            rotation=observation.rotation.tolist(), held_out=i in heldout,
            image=f'us_{i:05d}.png', reslice=f'reslice_{i:05d}.png', volume=f'volume_{i:05d}.bin',
            flow=None, estimate=None))
    np.savez_compressed(args.output/'reconstructed_volume.npz', intensity=volume.intensity, weight=volume.weight,
        origin_m=volume.origin, spacing_m=volume.spacing, sampled_frame_times=[o.time for o in observations])
    print('Image-only tracked volume complete', volume.shape, flush=True)

    # Find the longest nearly coplanar run using poses only, never load commands.
    eligible_runs = []
    for i, anchor in enumerate(observations):
        run = [j for j in range(i, len(observations)) if
            abs(float((observations[j].position-anchor.position)@anchor.rotation[:, 1])) < .0005 and
            np.dot(anchor.rotation[:, 1], observations[j].rotation[:, 1]) > np.cos(np.deg2rad(1))]
        eligible_runs.append(run)
    run = max(eligible_runs, key=len)
    anchor_index = run[0]
    anchor = observations[anchor_index]
    print('Image motion anchor', anchor_index, 'eligible', run, flush=True)
    inverse = ElasticInverse(dict(np.load(args.prior)))
    model = dict(np.load(args.gaussians))
    static = np.column_stack((model['colors'], model['opacity'], model['semantic'], model['youngs'], model['means'])).astype(np.float32)
    static.tofile(args.output/'prior.bin')
    rows, cols = np.mgrid[10:anchor.image.shape[0]-10:8, 10:anchor.image.shape[1]-10:8]
    rows, cols = rows.ravel(), cols.ravel()
    selected = anchor.image[rows, cols] > .25
    rows, cols = rows[selected], cols[selected]
    world = anchor.world_points(rows, cols)
    directions = anchor.rotation[:, [0, 2]].T
    baseline_confidence = np.clip((anchor.image[rows, cols]-.2)/.5, .1, 1)
    inverse.prepare(world, directions, baseline_confidence)
    rest = [b.points for b in inverse.bodies]
    last_positions = rest
    previous_observation = anchor
    cumulative = np.zeros((len(world), 3))
    for i, observation in enumerate(observations):
        start = time.perf_counter()
        tracks = (advance_material_tracks(previous_observation, observation, world, inverse.point_displacements(), cumulative, reference=anchor)
                  if i in run else None)
        if tracks is not None:
            cumulative, confidence = tracks
            # Unreliable image measurements receive less weight; they must not
            # become artificial zero-displacement constraints.
            measured = cumulative@directions.T
            estimated, report = inverse.solve(measured, confidence=confidence)
            if report['min_volume_jacobian'] <= 0:
                raise RuntimeError(f'inverse estimate inverted a tetrahedron: {report}')
            last_positions = estimated
            pixels = measured/np.array([(anchor.lateral_max-anchor.lateral_min)/(anchor.image.shape[1]-1),
                (anchor.depth_max-anchor.depth_min)/(anchor.image.shape[0]-1)])
            flow = np.column_stack((rows, cols, pixels, confidence)).astype(np.float32)
            flow.tofile(args.output/f'flow_{i:05d}.bin')
            np.savez_compressed(args.output/f'inferred_state_{i:05d}.npz',
                **{f'positions_{j}': positions for j, positions in enumerate(estimated)},
                image_anchor=anchor_index, image_frame=i)
            frames[i].update(flow=f'flow_{i:05d}.bin', estimate=report)
            previous_observation = observation
            print('Image-driven inverse', i, report, 'seconds', time.perf_counter()-start, flush=True)
        # Outside the observable run do not extrapolate a motion trajectory.
        estimated = last_positions if i in run else rest
        rotation = observation.rotation@np.diag([1., -1., -1.])
        angle = np.arctan2(rotation[1, 0], rotation[0, 0])
        pose = np.r_[observation.position, 0, 0, np.sin(angle/2), np.cos(angle/2)]
        centers, cov, displacement, stress, j = advect_gaussians(model, estimated, pose)
        values = np.column_stack((centers, cov[:, [0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]], displacement, stress)).astype(np.float32)
        values.tofile(args.output/f'prior_{i:05d}.bin')
        frames[i]['prior'] = f'prior_{i:05d}.bin'
        frames[i]['inverse_seconds'] = time.perf_counter()-start
    manifest = dict(version=1, source='recorded synthetic SIMUS RF -> IQ -> B-mode', frames=frames,
        width_pixels=anchor.image.shape[1], depth_pixels=anchor.image.shape[0],
        lateral_min_m=anchor.lateral_min, lateral_max_m=anchor.lateral_max,
        depth_min_m=anchor.depth_min, depth_max_m=anchor.depth_max,
        volume_spacing_m=volume.spacing, anchor_frame=anchor_index, coplanar_frames=run,
        prior_gaussians=len(model['means']), inference_measurements=['B-mode image intensities', 'tracked probe poses', 'pixel calibration'],
        prior_inputs=[str(args.prior), str(args.gaussians)],
        forbidden_inputs_not_read=['deformed FEM states', 'loading commands', 'deformed/evaluation tumour masks'],
        observation_files=hashes, heldout_metrics=metrics, elapsed_seconds=time.perf_counter()-started,
        limitations=['Synthetic acquisition; not validated against clinical images.',
            'Only incremental in-plane motion is measured; 3D extension and initial tumour shape come from the elastic/MRI prior.',
            'Echo compounding mixes acquisition times while tissue is deforming.',
            'Unscanned anatomy and out-of-plane tissue motion are not recovered from these images.'])
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('Completed', args.output, 'seconds', manifest['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    main()
