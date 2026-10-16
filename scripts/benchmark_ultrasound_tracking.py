"""Replay image-only estimators, then independently score actual Isaac truth.

The inference function has no truth-directory argument. Every method sees the
same calibrated images, poses and initial MRI prior. Truth is read only by score.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from deform_sim.reconstruction import Observation, advance_material_tracks, advance_material_tracks_lk
from deform_sim.inverse_elasticity import ElasticInverse
from deform_sim.phantom import load_bodies, surface_positions
from deform_sim.comparison import slice_surface, contour_metrics


def infer(observations, prior, output, methods):
    data = [dict(np.load(p)) for p in observations]
    images = [Observation(d['image'], d['position'], d['rotation'], float(d['time']),
        **{k: float(d[k]) for k in ('lateral_min','lateral_max','depth_min','depth_max')}) for d in data]
    anchor = images[0]
    rr, cc = np.mgrid[12:anchor.image.shape[0]-12:16, 12:anchor.image.shape[1]-12:16]
    rows, cols = rr.ravel(), cc.ravel()
    select = anchor.image[rows, cols]>.25
    rows, cols = rows[select], cols[select]
    world = anchor.world_points(rows, cols)
    directions = anchor.rotation[:, [0, 2]].T
    inverse = ElasticInverse(prior)
    inverse.prepare(world, directions, np.clip((anchor.image[rows, cols]-.2)/.5, .1, 1))
    cached = inverse.cached
    for name in methods:
        inverse.cached = cached
        inverse.data_weight = float(name[2:]) if name.startswith('lk') and name[2:].isdigit() else 5000.
        cumulative = np.zeros((len(world), 3))
        previous = anchor
        folder = output/name; folder.mkdir(parents=True, exist_ok=True)
        tracker = advance_material_tracks_lk if name.startswith('lk') else advance_material_tracks
        for i, (path, observation) in enumerate(zip(observations, images)):
            started = time.monotonic()
            tracks = tracker(previous, observation, world, inverse.point_displacements(), cumulative, reference=anchor)
            if tracks is None:
                raise RuntimeError('Benchmark left the observable plane')
            proposed, confidence = tracks
            positions, report = inverse.solve(proposed@directions.T, confidence=confidence)
            if name == 'tvl1' or report['accepted']:
                cumulative = proposed
            np.savez(folder/path.name, **{f'positions_{j}':p for j,p in enumerate(positions)})
            report.update(frame=int(data[i]['frame']), time=observation.time,
                inference_ms=(time.monotonic()-started)*1000, image_only=True)
            (folder/path.with_suffix('.json').name).write_text(json.dumps(report)+'\n')
            previous = observation
            if i%20 == 0:
                print(name, i, '/', len(images), 'accepted', report['accepted'], flush=True)


def score(observations, truth_dir, prior, output, methods):
    bodies = load_bodies(prior)
    faces = prior['display_faces_1']
    rows = []
    for path in observations:
        observation = dict(np.load(path))
        calibration = {k:float(observation[k]) for k in ('lateral_min','lateral_max','depth_min','depth_max')}
        truth = np.load(truth_dir/path.name)
        assert int(truth['frame']) == int(observation['frame']) and int(truth['epoch']) == int(observation['epoch'])
        actual = surface_positions(prior, [truth[f'positions_{j}'] for j in range(len(bodies))], 1)
        actual_contour = slice_surface(actual, faces, observation['position'], observation['rotation'])
        row = dict(frame=int(observation['frame']), time=float(observation['time']), methods={})
        for name in ['static']+methods:
            if name == 'static':
                points = prior['display_points_1']; report = dict(accepted=True, inference_ms=0)
            else:
                prediction = np.load(output/name/path.name)
                points = surface_positions(prior, [prediction[f'positions_{j}'] for j in range(len(bodies))], 1)
                report = json.loads((output/name/path.with_suffix('.json').name).read_text())
            contour = slice_surface(points, faces, observation['position'], observation['rotation'])
            metrics = contour_metrics(contour, actual_contour, calibration, observation['image'].shape)
            metrics.update(surface_vertex_rmse_mm=float(np.sqrt(np.mean(np.sum((points-actual)**2,axis=1)))*1000),
                accepted=report['accepted'], inference_ms=report['inference_ms'])
            row['methods'][name] = metrics
        rows.append(row)
    summary = {}
    for name in ['static']+methods:
        values = [r['methods'][name] for r in rows]
        summary[name] = {key:float(np.mean([v[key] for v in values if v[key] is not None]))
            for key in ['surface_vertex_rmse_mm','mean_contour_distance_mm','dice','inference_ms']}
        summary[name]['accepted_fraction'] = float(np.mean([v['accepted'] for v in values]))
    result = dict(scope='Synthetic tracked in-plane scan, all methods use identical images and static MRI prior.',
        no_truth_during_inference=True, frames=len(rows), summary=summary, frames_detail=rows)
    (output/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('episode', type=Path)
    p.add_argument('--prior', type=Path, default=Path('data/remind-041/anatomy.npz'))
    p.add_argument('--output', type=Path, default=Path('outputs/small_tumour_benchmark'))
    p.add_argument('--methods', nargs='+', default=['tvl1','lk10'])
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--score-only', action='store_true')
    args=p.parse_args()
    observations=sorted((args.episode/'recording/observations').glob('*.npz'))[::args.stride]
    if len(observations)<10:raise ValueError('Record at least ten observations first')
    prior=dict(np.load(args.prior))
    args.output.mkdir(parents=True,exist_ok=True)
    if not args.score_only:infer(observations,prior,args.output,args.methods)
    score(observations,args.episode/'recording/truth',prior,args.output,args.methods)


if __name__=='__main__':main()
