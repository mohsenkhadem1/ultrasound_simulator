"""Check that an exported Isaac episode has valid, bonded, deformed volume state."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('episode', type=Path)
    args = parser.parse_args()
    reference = np.load(args.episode/'reference.npz')
    count = int(reference['body_count']) if 'body_count' in reference else 2
    interfaces = []
    for i in range(count):
        for j in range(i+1, count):
            _, ia, ib = np.intersect1d(reference[f'ids_{i}'], reference[f'ids_{j}'], return_indices=True)
            if len(ia):
                interfaces.append((i, j, ia, ib))
    det = lambda p: np.linalg.det((p[:, 1:]-p[:, :1]).transpose(0, 2, 1))
    rest = [det(reference[f'points_{i}'][reference[f'tets_{i}']]) for i in range(count)]
    max_gap, max_motion, min_jacobian = 0., 0., 1.
    files = sorted(p for p in (args.episode/'state').glob('*.npz') if p.stem.isdecimal())
    assert files and (args.episode/'complete.json').exists(), 'episode incomplete'
    for file in files:
        state = np.load(file)
        p = [state[f'positions_{i}'] for i in range(count)]
        assert all(np.isfinite(x).all() for x in p)
        for i, j, ia, ib in interfaces:
            max_gap = max(max_gap, float(np.linalg.norm(p[i][ia]-p[j][ib], axis=1).max()))
        for i in range(count):
            jacobian = det(p[i][reference[f'tets_{i}']])/rest[i]
            min_jacobian = min(min_jacobian, float(jacobian.min()))
            max_motion = max(max_motion, float(np.linalg.norm(p[i]-reference[f'points_{i}'], axis=1).max()))
    assert max_gap < .0002, f'broken inclusion bond: {max_gap} m'
    assert min_jacobian > 0, f'inverted tetrahedron: Jacobian {min_jacobian}'
    assert max_motion > .0001, 'no meaningful deformation'
    metadata = json.loads((args.episode/'metadata.json').read_text())
    if metadata.get('probe_motion') in ('press', 'sweep', 'scan'):
        probe_z = np.array([float(np.load(f)['probe_body_pose'][2]) for f in files])
        tool_z = np.array([float(np.load(f)['tool_pose'][2]) for f in files])
        assert np.ptp(probe_z) > .003, 'probe did not move through its pressing stroke'
        assert np.ptp(tool_z) < 1e-6 and tool_z.min() > .06, 'separate tool must stay parked'
    report = dict(frames=len(files), max_interface_gap_m=max_gap,
                  max_displacement_m=max_motion, min_volume_jacobian=min_jacobian)
    if metadata.get('probe_motion') in ('press', 'sweep', 'scan'):
        report['probe_stroke_m'] = float(np.ptp(probe_z))
        report['separate_tool_parked'] = True
    if metadata.get('probe_motion') == 'sweep':
        xy = np.array([np.load(f)['probe_body_pose'][:2] for f in files])
        assert np.ptp(xy[:, 0]) > .005, 'probe did not sweep laterally'
        report['probe_lateral_stroke_m'] = np.ptp(xy, axis=0).tolist()
    if metadata.get('probe_motion') == 'scan':
        xy = np.array([np.load(f)['probe_body_pose'][:2] for f in files])
        command = np.array([float(np.load(f)['command']) for f in files])
        assert np.ptp(xy[:, 1]) > .05, 'probe did not cross the tumour'
        assert command.max() >= .0049, 'extra pressing motion was not recorded'
        from deform_sim.slice_geometry import tumor_segments
        counts = [len(tumor_segments(reference, np.load(f))) for f in files]
        assert counts[0] == 0 and counts[-1] == 0 and max(counts) > 0, 'scan must enter and leave the tumour'
        report.update(probe_scan_distance_m=float(np.ptp(xy[:, 1])), max_press_travel_m=float(command.max()),
                      tumor_intersection_frames=int(np.count_nonzero(counts)), tumor_absent_frames=int(len(counts)-np.count_nonzero(counts)))
    (args.episode/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
