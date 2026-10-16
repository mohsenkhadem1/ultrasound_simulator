"""Prescribed, surface-following tumour scan, in metres and seconds."""
import numpy as np


def contact_height_function(bodies):
    """Rest top surface under the actual rectangular head, not a wide scan envelope.

    Prepared phantoms are Cartesian occupied cubes split into tetrahedra. Group
    their columns to query the maximum occupied height intersecting the head.
    This avoids leaving the probe above steep anatomy because of distant nodes.
    """
    vertices = np.concatenate([b.points[b.tetrahedra] for b in bodies])
    lower, upper = vertices.min(axis=1), vertices.max(axis=1)
    rectangles, inverse = np.unique(np.column_stack((lower[:, :2], upper[:, :2])), axis=0, return_inverse=True)
    height = np.full(len(rectangles), -np.inf)
    np.maximum.at(height, inverse, upper[:, 2])
    def query(x, y, yaw=0.):
        c, s = abs(np.cos(yaw)), abs(np.sin(yaw))
        half_x, half_y = .02*c+.004*s, .02*s+.004*c
        overlap = ((rectangles[:, 0]<x+half_x-1e-9)&(rectangles[:, 2]>x-half_x+1e-9)&
                   (rectangles[:, 1]<y+half_y-1e-9)&(rectangles[:, 3]>y-half_y+1e-9))
        if not overlap.any():
            raise ValueError('probe footprint leaves the prepared brain')
        return float(height[overlap].max())
    return query


def scan_terrain(points):
    """Conservative contact envelope for a 40 x 8 mm head and modest yaw."""
    central = points[np.abs(points[:, 0]) <= .028]
    lower = max(-.033, float(central[:, 1].min())+.005)
    upper = min(.033, float(central[:, 1].max())-.005)
    if lower >= upper:
        raise ValueError('prepared brain has no supported scan interval')
    y = np.linspace(lower, upper, 133)
    heights = []
    for center in y:
        nearby = (np.abs(points[:, 0]) <= .028) & (np.abs(points[:, 1]-center) <= .010)
        if not nearby.any():
            raise ValueError('scan leaves the prepared brain mesh')
        heights.append(points[nearby, 2].max())
    # Round the staircase of the coarse FEM surface without lowering its envelope.
    padded = np.pad(heights, 5, mode='edge')
    envelope = np.max(np.lib.stride_tricks.sliding_window_view(padded, 11), axis=1)
    rounded = np.convolve(np.pad(envelope, 3, mode='edge'), np.ones(7)/7, mode='valid')
    return y, rounded


def scan_pose(time, duration, terrain):
    """Approach, cross the lesion, press 2 mm farther, scan past it and lift."""
    u = np.clip(time/duration, 0., 1.)
    blend = lambda a, b: (lambda q: q*q*(3-2*q))(np.clip((u-a)/(b-a), 0., 1.))
    progress = .5*blend(.08, .36)+.5*blend(.60, .88)
    left, right = max(-.032, float(terrain[0][0])+.001), min(.032, float(terrain[0][-1])-.001)
    x, y = -.004+.008*progress, left+(right-left)*progress
    stroke = .003*blend(0., .08)+.002*blend(.36, .44)-.002*blend(.52, .60)-.003*blend(.88, 1.)
    contact = float(np.interp(y, *terrain))
    z = contact+.004-stroke  # 3 mm half-head plus 1 mm initial clearance
    yaw = np.deg2rad(8)*np.sin(2*np.pi*progress)
    phases = ['APPROACH', 'FIND TUMOUR', 'GENTLE PRESS', 'HOLD', 'EASE PRESSURE', 'SCAN PAST', 'LIFT']
    phase = phases[np.searchsorted([.08, .36, .44, .52, .60, .88], u, side='right')]
    return np.array([x, y, z]), yaw, stroke, contact, phase
