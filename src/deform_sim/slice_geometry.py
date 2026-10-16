"""Reference tumour intersections in the recorded acoustic plane."""
import numpy as np
from .phantom import surface_positions, probe_coordinates


def plane_segments(points, faces):
    """Intersect triangular surfaces with acoustic elevation y=0; return x,z."""
    triangles = points[faces]
    a, b = triangles, triangles[:, [1, 2, 0]]
    crossing = (a[:, :, 1] > 0) != (b[:, :, 1] > 0)
    a, b = a[crossing], b[crossing]
    if not len(a):
        return np.empty((0, 2, 2))
    cuts = a + (b-a)*(-a[:, 1]/(b[:, 1]-a[:, 1]))[:, None]
    return cuts.reshape(-1, 2, 3)[:, :, [0, 2]]


def tumor_segments(reference, state):
    count = int(reference['body_count']) if 'body_count' in reference else 2
    positions = [state[f'positions_{i}'] for i in range(count)]
    points = surface_positions(reference, positions, 1)
    local = probe_coordinates(points, state['probe_position'], state['probe_rotation'])
    return plane_segments(local, reference['display_faces_1'])
