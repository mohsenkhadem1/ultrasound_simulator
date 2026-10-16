"""Convert the patient mesh prior into material-bound Gaussian surface splats."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from deform_sim.phantom import load_bodies


def sample_surface(points, faces, count, rng):
    triangles = points[faces]
    normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    area = np.linalg.norm(normals, axis=1)/2
    ids = rng.choice(len(faces), count, p=area/area.sum())
    weights = rng.dirichlet([1., 1., 1.], count)
    centers = np.einsum('ni,nij->nj', weights, triangles[ids])
    normal = normals[ids]/np.maximum(2*area[ids, None], 1e-15)
    tangent_sigma = np.clip(np.sqrt(area.sum()/count)*.65, .00035, .0018)
    covariance = tangent_sigma**2*np.eye(3)[None]+((.25*tangent_sigma)**2-tangent_sigma**2)*normal[:, :, None]*normal[:, None, :]
    return centers, covariance, normal


def bind(points, catalog, candidates):
    """Fixed tetrahedral embedding; allow explicit extrapolation at fine skins."""
    tets, rest, inverse, _ = catalog
    _, local = cKDTree(rest[candidates].mean(axis=1)).query(points, k=min(32, len(candidates)))
    ids = candidates[local]
    tail = np.einsum('nkij,nkj->nki', inverse[ids], points[:, None]-rest[ids, 0])
    weights = np.concatenate((1-tail.sum(axis=2, keepdims=True), tail), axis=2)
    penalty = np.minimum(weights, 0.)**2
    score = penalty.sum(axis=2)+.0001*np.sum((points[:, None]-rest[ids].mean(axis=2))**2, axis=2)/.005**2
    best = score.argmin(axis=1)
    chosen = ids[np.arange(len(points)), best]
    return chosen, weights[np.arange(len(points)), best]


def box_surface(lower, upper):
    points = np.array([[x, y, z] for x in [lower[0], upper[0]] for y in [lower[1], upper[1]] for z in [lower[2], upper[2]]])
    quads = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]
    faces = np.array([tri for a, b, c, d in quads for tri in [[a, b, c], [a, c, d]]])
    return points, faces


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--anatomy', type=Path, default=Path('data/remind-001/anatomy.npz'))
    p.add_argument('--output', type=Path, default=Path('data/remind-001/gaussians.npz'))
    p.add_argument('--brain', type=int, default=28000)
    p.add_argument('--tumor', type=int, default=10000)
    p.add_argument('--ventricles', type=int, default=4000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--depth-mm', type=float, default=50., help='acoustic plane depth for this prior')
    args = p.parse_args()
    reference = np.load(args.anatomy)
    bodies = load_bodies(reference)
    offsets = np.r_[0, np.cumsum([len(b.points) for b in bodies])]
    nodes = np.concatenate([b.points for b in bodies])
    tets = np.concatenate([b.tetrahedra+offsets[i] for i, b in enumerate(bodies)])
    region = np.repeat(np.arange(len(bodies)), [len(b.tetrahedra) for b in bodies])
    rest = nodes[tets]
    inverse = np.linalg.inv((rest[:, 1:]-rest[:, :1]).transpose(0, 2, 1))
    catalog = tets, rest, inverse, region
    rng = np.random.default_rng(args.seed)
    result = {k: [] for k in ['means', 'covariance', 'colors', 'opacity', 'semantic', 'nodes', 'weights', 'inverse_rest_edges', 'youngs']}
    light = np.array([-.3, -.5, .8]); light /= np.linalg.norm(light)
    for semantic, count, color in [(0, args.brain, [.83, .47, .47]), (1, args.tumor, [.88, .19, .46]), (2, args.ventricles, [.15, .63, .93])]:
        points, cov, normal = sample_surface(reference[f'display_points_{semantic}'], reference[f'display_faces_{semantic}'], count, rng)
        # Acoustic and display surfaces share the complete FEM material map.
        # Restricting a fine skin to a coarse labelled region extrapolates it.
        candidates = np.arange(len(tets))
        ids, weights = bind(points, catalog, candidates)
        shade = .45+.55*np.abs(normal@light)
        result['means'].append(points); result['covariance'].append(cov)
        result['colors'].append(np.asarray(color)[None]*shade[:, None])
        result['opacity'].append(np.full(count, .85)); result['semantic'].append(np.full(count, semantic))
        result['nodes'].append(tets[ids]); result['weights'].append(weights)
        result['inverse_rest_edges'].append(inverse[ids])
        result['youngs'].append(reference['youngs_modulus'][region[ids]])
    soft_count = sum(len(x) for x in result['means'])
    # The rigid head and handle are also splatted so depth compositing stays correct.
    for low, high, count, color in [([-.02, -.004, -.003], [.02, .004, .003], 1200, [.98, .62, .08]),
                                   ([-.008, -.006, .003], [.008, .006, .053], 1800, [.82, .86, .91])]:
        points, cov, _ = sample_surface(*box_surface(low, high), count, rng)
        result['means'].append(points); result['covariance'].append(cov)
        result['colors'].append(np.tile(color, (count, 1)))
        result['opacity'].append(np.full(count, .95)); result['semantic'].append(np.full(count, 3))
    depth = args.depth_mm*.001
    corners = np.array([[-.02, 0, -.002], [.02, 0, -.002], [.02, 0, -depth], [-.02, 0, -depth], [-.02, 0, -.002]])
    line = np.concatenate([a+(b-a)*np.linspace(0, 1, 48)[:, None] for a, b in zip(corners[:-1], corners[1:])])
    result['means'].append(line); result['covariance'].append(np.tile(np.eye(3)*.00022**2, (len(line), 1, 1)))
    result['colors'].append(np.tile([.03, .85, .90], (len(line), 1)))
    result['opacity'].append(np.full(len(line), .85)); result['semantic'].append(np.full(len(line), 4))
    n_rigid = 3000+len(line)
    result['nodes'].append(np.zeros((n_rigid, 4), int)); result['weights'].append(np.zeros((n_rigid, 4)))
    result['inverse_rest_edges'].append(np.tile(np.eye(3), (n_rigid, 1, 1))); result['youngs'].append(np.zeros(n_rigid))
    payload = {k: np.concatenate(v).astype(np.int32 if k in ('semantic', 'nodes') else np.float32) for k, v in result.items()}
    payload['soft_count'] = soft_count
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    info = dict(source=str(args.anatomy), count=len(payload['means']), soft_count=soft_count, seed=args.seed,
                source_method='Area-sampled segmented surfaces with fixed tetrahedral material coordinates',
                deformed_covariance='F @ Sigma_rest @ F.T',
                extrapolated_skin_fraction=float(np.mean(payload['weights'][:soft_count].min(axis=1) < -1e-5)),
                minimum_barycentric_weight=float(payload['weights'][:soft_count].min()),
                limitations=['Semantic mesh-prior appearance; not photometrically trained 3DGS.',
                             'Fine skins can extrapolate beyond the coarse FEM boundary.',
                             'Physics-driven Gaussian transport; not an ultrasound-inferred or PIDG-trained reconstruction.'])
    args.output.with_suffix('.json').write_text(json.dumps(info, indent=2)+'\n')
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
