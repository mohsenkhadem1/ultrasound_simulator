"""Conforming tetrahedral phantom and material-embedded scatterers (metres)."""
from dataclasses import dataclass
import itertools
import numpy as np


@dataclass
class BodyMesh:
    points: np.ndarray
    tetrahedra: np.ndarray
    global_ids: np.ndarray


def make_phantom(n=9):
    """Matrix with a cavity and a conforming bonded inclusion, no overlap."""
    if n < 5:
        raise ValueError('n >= 5 required for inclusion geometry')
    shape = np.array([0.04, 0.02, 0.04])
    origin = np.array([-0.02, -0.01, 0.0])
    points = np.array(list(itertools.product(*[np.linspace(0, s, n) for s in shape]))) + origin
    index = lambda i, j, k: (i*n+j)*n+k
    regions = [[], []]
    for i, j, k in itertools.product(range(n-1), repeat=3):
        center = origin + shape * (np.array([i, j, k])+0.5)/(n-1)
        region = int(np.linalg.norm(center-np.array([0, 0, 0.021])) < 0.006)
        # Freudenthal subdivision: compatible shared faces across cells.
        for order in itertools.permutations(range(3)):
            v = np.array([i, j, k])
            tet = [index(*v)]
            for axis in order:
                v = v.copy()
                v[axis] += 1
                tet.append(index(*v))
            if np.linalg.det((points[tet[1:]]-points[tet[0]]).T) < 0:
                tet[1], tet[2] = tet[2], tet[1]
            regions[region].append(tet)
    bodies = []
    for region in regions:
        tets = np.asarray(region, dtype=np.int32)
        ids, inverse = np.unique(tets, return_inverse=True)
        bodies.append(BodyMesh(points[ids], inverse.reshape(-1, 4).astype(np.int32), ids))
    shared, a, b = np.intersect1d(bodies[0].global_ids, bodies[1].global_ids, return_indices=True)
    return bodies, (a.astype(np.int32), b.astype(np.int32))


def load_bodies(reference):
    """Read old two-region episodes and arbitrary segmented anatomy alike."""
    count = int(reference['body_count']) if 'body_count' in reference else 2
    return [BodyMesh(reference[f'points_{i}'], reference[f'tets_{i}'], reference[f'ids_{i}']) for i in range(count)]


def surface_positions(reference, positions, i):
    """Move a detailed anatomical skin with fixed interpolation of FEM motion."""
    key = f'display_points_{i}'
    if key not in reference:
        return positions[i]
    # The skin has its own geometry; interpolate displacement, not rest shape.
    displacement = np.concatenate([p-reference[f'points_{j}'] for j, p in enumerate(positions)])
    return reference[key] + np.einsum('nk,nkj->nj', reference[f'display_weights_{i}'],
                                    displacement[reference[f'display_nodes_{i}']])


def sample_scatterers(bodies, count=120000, seed=42, amplitudes=None, bounds=None):
    """Uniform samples by rest volume, fixed signed reflectivities and embeddings."""
    rng = np.random.default_rng(seed)
    volumes = [np.abs(np.linalg.det((b.points[b.tetrahedra[:, 1:]] - b.points[b.tetrahedra[:, :1]]).transpose(0, 2, 1)))/6 for b in bodies]
    if bounds is not None:
        bounds = np.asarray(bounds)
        for b, volume in zip(bodies, volumes):
            center = b.points[b.tetrahedra].mean(axis=1)
            volume[~np.all((center >= bounds[0]) & (center <= bounds[1]), axis=1)] = 0
    flat_volumes = np.concatenate(volumes)
    choice = rng.choice(len(flat_volumes), count, p=flat_volumes/flat_volumes.sum())
    boundaries = np.r_[0, np.cumsum([len(v) for v in volumes])]
    body = np.searchsorted(boundaries[1:], choice, side='right').astype(np.int32)
    tet = choice - boundaries[body]
    weights = rng.dirichlet(np.ones(4), count)
    if amplitudes is None:
        amplitudes = [1.0] + [0.18]*(len(bodies)-1)
    rc = rng.normal(size=count) * np.asarray(amplitudes)[body]
    return dict(body=body, tet=tet.astype(np.int32), weights=weights, reflectivity=rc)


def advect(bodies, positions, scatter):
    result = np.empty((len(scatter['body']), 3))
    for region, body in enumerate(bodies):
        mask = scatter['body'] == region
        vertices = positions[region][body.tetrahedra[scatter['tet'][mask]]]
        result[mask] = np.einsum('ni,nij->nj', scatter['weights'][mask], vertices)
    return result


def probe_coordinates(world, position=None, rotation=None):
    """Map material scatterers into the acoustic frame at the current probe pose.

    Rotation columns are the lateral, elevation and depth axes in world space.
    Omitting the pose preserves the original fixed side-probe convention.
    """
    if position is not None:
        if rotation is None:
            raise ValueError('probe rotation is required with a probe position')
        return (world-np.asarray(position)) @ np.asarray(rotation)
    # Fixed probe on the -y face: lateral=world x, elevation=world z-20mm,
    # acoustic depth=world y+15mm (5mm coupling path before the phantom).
    return np.column_stack((world[:, 0], world[:, 2]-0.02, world[:, 1]+0.015))


def acoustic_scatterers(world, reflectivity, state):
    """Pose-aware scatterers in front of the aperture; preserve material IDs."""
    if 'probe_position' in state:
        xyz = probe_coordinates(world, state['probe_position'], state['probe_rotation'])
    else:
        xyz = probe_coordinates(world)
    visible = xyz[:, 2] > 1e-4
    if not np.any(visible):
        raise ValueError('no tissue scatterers in front of the ultrasound aperture')
    return xyz[visible], reflectivity[visible]
