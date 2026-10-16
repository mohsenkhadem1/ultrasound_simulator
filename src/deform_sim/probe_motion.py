"""Prescribed, surface-following tumour scan, in metres and seconds."""
import numpy as np


def scan_terrain(points):
    """Conservative contact envelope for a 40 x 8 mm head and modest yaw."""
    y = np.linspace(-.033, .033, 133)
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
    x, y = -.004+.008*progress, -.032+.064*progress
    stroke = .003*blend(0., .08)+.002*blend(.36, .44)-.002*blend(.52, .60)-.003*blend(.88, 1.)
    contact = float(np.interp(y, *terrain))
    z = contact+.004-stroke  # 3 mm half-head plus 1 mm initial clearance
    yaw = np.deg2rad(8)*np.sin(2*np.pi*progress)
    phases = ['APPROACH', 'FIND TUMOUR', 'GENTLE PRESS', 'HOLD', 'EASE PRESSURE', 'SCAN PAST', 'LIFT']
    phase = phases[np.searchsorted([.08, .36, .44, .52, .60, .88], u, side='right')]
    return np.array([x, y, z]), yaw, stroke, contact, phase
