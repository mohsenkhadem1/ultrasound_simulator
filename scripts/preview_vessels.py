"""Plot the two real TopBrain mesh groups in their shared source frame."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data/topbrain-001'))
    parser.add_argument('--output', type=Path, default=Path('outputs/anatomy_sources/vessels.png'))
    args = parser.parse_args()
    fig = plt.figure(figsize=(10, 9), facecolor='#101824')
    ax = fig.add_subplot(projection='3d', facecolor='#101824')
    points = []
    for name, color in [('arteries', '#ed6464'), ('veins', '#668eff')]:
        mesh = np.load(args.data/(name+'.npz'))
        v = mesh['points']*1000
        points.append(v)
        surface = Poly3DCollection(v[mesh['faces']], facecolors=color, linewidths=0, shade=True, lightsource=matplotlib.colors.LightSource(azdeg=230, altdeg=40))
        ax.add_collection3d(surface)
    all_points = np.concatenate(points)
    center = (all_points.min(axis=0)+all_points.max(axis=0))/2
    radius = np.ptp(all_points, axis=0).max()*.53
    ax.set(xlim=(center[0]-radius, center[0]+radius), ylim=(center[1]-radius, center[1]+radius), zlim=(center[2]-radius, center[2]+radius))
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=24, azim=-65)
    ax.set_axis_off()
    fig.text(.5, .94, 'Real segmented cerebral vessels', ha='center', color='white', fontsize=22)
    fig.text(.5, .89, 'Red: arteries     Blue: veins and venous sinuses', ha='center', color='#b7c5dc', fontsize=14)
    fig.text(.5, .075, 'TopBrain CT subject 001 • Separate subject from the brain simulation', ha='center', color='#b7c5dc', fontsize=12)
    fig.text(.5, .04, 'Source: TopBrain Challenge Organizers • doi.org/10.5281/zenodo.16878417', ha='center', color='#b7c5dc', fontsize=10)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, facecolor=fig.get_facecolor())
    print(args.output)


if __name__ == '__main__':
    main()
