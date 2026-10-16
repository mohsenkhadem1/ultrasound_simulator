"""Evaluation-only, timestamp-matched Isaac/ultrasound contour comparison.

Truth is consumed here, after prediction. It never becomes an inverse input.
Contours are triangle/plane intersections of the same MRI surface embedding
transported by the initial, estimated and actual Isaac nodal positions.
"""
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
from scipy.spatial import cKDTree
from .phantom import load_bodies, surface_positions
from .gaussians import pack_gaussian_frame
from .comparison_mesh import ComparisonMesh


def slice_surface(points, faces, position, rotation):
    """Return unoriented lateral/depth line segments in acoustic metres.

    A half-open sign convention avoids double-counting vertices on the plane.
    Triangles entirely on the plane are omitted (not a contour crossing).
    """
    local = (points-position)@rotation
    tri = local[faces]
    signs = tri[:, :, 1] >= 0
    tri = tri[np.any(signs, axis=1) & ~np.all(signs, axis=1)]
    if not len(tri):
        return np.empty((0, 2, 2))
    hits = np.empty((len(tri), 2, 2))
    count = np.zeros(len(tri), dtype=int)
    for a, b in [(0, 1), (1, 2), (2, 0)]:
        valid = (tri[:, a, 1] >= 0) != (tri[:, b, 1] >= 0)
        ids = np.flatnonzero(valid)
        t = -tri[ids, a, 1]/(tri[ids, b, 1]-tri[ids, a, 1])
        p = tri[ids, a]+t[:, None]*(tri[ids, b]-tri[ids, a])
        hits[ids, count[ids]] = p[:, [0, 2]]
        count[ids] += 1
    return hits[(count == 2) & (np.linalg.norm(hits[:, 0]-hits[:, 1], axis=1) > 1e-10)]


def contour_mask(segments, width=192, height=192, lateral_min=-.02, lateral_max=.02, depth_min=.002, depth_max=.05):
    """Even/odd scanline fill on the acquired image FOV, including holes."""
    x = np.linspace(lateral_min, lateral_max, width)
    z = np.linspace(depth_min, depth_max, height)
    mask = np.zeros((height, width), bool)
    if not len(segments):
        return mask
    a, b = segments[:, 0], segments[:, 1]
    for row, depth in enumerate(z):
        crossing = (a[:, 1] > depth) != (b[:, 1] > depth)
        aa, bb = a[crossing], b[crossing]
        xs = aa[:, 0]+(depth-aa[:, 1])*(bb[:, 0]-aa[:, 0])/(bb[:, 1]-aa[:, 1])
        mask[row] = np.count_nonzero(xs[:, None] > x[None], axis=0) % 2 == 1
    return mask


def contour_samples(segments, spacing=.0001):
    if not len(segments):
        return np.empty((0, 2))
    samples = []
    for a, b in segments:
        n = max(1, int(np.ceil(np.linalg.norm(b-a)/spacing)))
        samples.append(a+(b-a)*((np.arange(n)+.5)/n)[:, None])
    return np.concatenate(samples)


def contour_metrics(candidate, truth, calibration=None, shape=(192, 192)):
    """Dice is image-grid overlap; distances use contours inside the image FOV."""
    calibration = calibration or dict(lateral_min=-.02, lateral_max=.02, depth_min=.002, depth_max=.05)
    a, b = [contour_mask(c, width=shape[1], height=shape[0], **calibration) for c in (candidate, truth)]
    denom = int(a.sum()+b.sum())
    # No tumour in the acquired field must not appear as a perfect Dice score.
    dice = float(2*np.count_nonzero(a & b)/denom) if denom and b.any() else None
    sa, sb = contour_samples(candidate), contour_samples(truth)
    def visible(p):
        return p[(p[:, 0] >= calibration['lateral_min']) & (p[:, 0] <= calibration['lateral_max']) &
                 (p[:, 1] >= calibration['depth_min']) & (p[:, 1] <= calibration['depth_max'])]
    sa, sb = visible(sa), visible(sb)
    if len(sa) and len(sb):
        distances = np.r_[cKDTree(sa).query(sb)[0], cKDTree(sb).query(sa)[0]]*1000
        mean, hd95 = float(distances.mean()), float(np.percentile(distances, 95))
    else:
        mean = hd95 = None
    return dict(dice=dice, mean_contour_distance_mm=mean, contour_hd95_mm=hd95,
                truth_pixels=int(b.sum()), candidate_pixels=int(a.sum()))


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value, allow_nan=False)+'\n')
    os.replace(temporary, path)


def run_comparison(args):
    prior = dict(np.load(args.prior))
    model = dict(np.load(args.gaussians))
    bodies = load_bodies(prior)
    rest = [b.points for b in bodies]
    offsets = np.cumsum([len(b.points) for b in bodies])[:-1]
    faces = prior['display_faces_1']
    initial_tumour = prior['display_points_1']
    destination = args.episode/'comparison'
    destination.mkdir(exist_ok=True)
    packet, _ = pack_gaussian_frame(model, rest, np.array([0., 0., 0., 0., 0., 0., 1.]), [])
    (destination/'static.bin').write_bytes(packet)
    mesh = ComparisonMesh(prior, bodies)
    mesh.faces.tofile(destination/'mesh_indices.bin')
    atomic_json(destination/'mesh.json', mesh.metadata)
    previous = None
    while not (args.episode/'live_complete.json').exists():
        path = args.episode/'live_ultrasound/estimate.json'
        if not path.exists():
            time.sleep(.1)
            continue
        metadata = json.loads(path.read_text())
        identifier, epoch = metadata['frame'], metadata['epoch']
        if (epoch, identifier) == previous:
            time.sleep(.05)
            continue
        truth_path = args.episode/'comparison_truth'/f'truth_{identifier:08d}.npz'
        prediction_path = args.episode/'comparison_predictions'/f'prediction_{identifier:08d}.npz'
        if not truth_path.exists() or not prediction_path.exists():
            time.sleep(.05)
            continue
        with np.load(truth_path) as truth, np.load(prediction_path) as prediction:
            if int(truth['epoch']) != epoch or int(prediction['epoch']) != epoch:
                time.sleep(.05)
                continue
            assert int(truth['frame']) == int(prediction['frame']) == identifier
            assert abs(float(truth['time'])-metadata['simulation_time']) < 1e-6
            truth_nodes = [truth[f'positions_{i}'] for i in range(len(bodies))]
            predicted_nodes = np.split(prediction['positions'], offsets)
            position, rotation = truth['probe_position'], truth['probe_rotation']
            yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
            pose = np.r_[position, 0., 0., np.sin(yaw/2), np.cos(yaw/2)]
            header = [1, float(truth['time']), identifier, len(model['means']), *position, yaw]
            packet = mesh.pack(truth_nodes, pose, header)
            (destination/f'truth_{identifier:08d}.bin').write_bytes(packet)
            truth_tumour = surface_positions(prior, truth_nodes, 1)
            predicted_tumour = surface_positions(prior, predicted_nodes, 1)
            surfaces = dict(static=initial_tumour, predicted=predicted_tumour, truth=truth_tumour)
            contours = {name: slice_surface(points, faces, position, rotation) for name, points in surfaces.items()}
            metrics = {}
            for name in ('static', 'predicted'):
                metrics[name] = contour_metrics(contours[name], contours['truth'],
                    calibration=metadata.get('calibration'), shape=metadata.get('image_shape', (192, 192)))
                metrics[name]['surface_vertex_rmse_mm'] = float(np.sqrt(np.mean(
                    np.sum((surfaces[name]-truth_tumour)**2, axis=1)))*1000)
            frame_data = dict(frame=identifier, epoch=epoch,
                contours={k: v.tolist() for k, v in contours.items()},
                position=position.tolist(), rotation=rotation.tolist(), metrics=metrics,
                tumour_center=initial_tumour.mean(axis=0).tolist(),
                interpretation='MRI-prior contour transported by image-inferred or actual Isaac nodal displacement; not US segmentation')
        atomic_json(destination/f'contours_{identifier:08d}.json', frame_data)
        predicted_name = f'predicted_{identifier:08d}.bin'
        shutil.copyfile(args.episode/'live_ultrasound'/metadata['estimate_file'], destination/predicted_name)
        shutil.copyfile(args.episode/'live_ultrasound'/metadata['image'], destination/metadata['image'])
        report = dict(metadata, truth_file=f'truth_{identifier:08d}.bin', predicted_file=predicted_name,
            contours_file=f'contours_{identifier:08d}.json', truth_frame=identifier, prediction_frame=identifier,
            truth_representation='conventional triangle mesh; no Gaussian splats', metrics=metrics,
            source='actual Isaac FEM state at the image timestamp; evaluation only')
        atomic_json(destination/'comparison.json', report)
        previous = epoch, identifier
        for pattern in ('truth_*.bin', 'predicted_*.bin', 'contours_*.json', 'us_*.png'):
            for stale in sorted(destination.glob(pattern))[:-32]:
                stale.unlink(missing_ok=True)
