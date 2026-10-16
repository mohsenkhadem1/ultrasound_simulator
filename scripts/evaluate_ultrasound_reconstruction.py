"""Independent synthetic evaluation; ground truth never enters reconstruction."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('reconstruction', type=Path)
    p.add_argument('--truth-episode', type=Path, required=True)
    p.add_argument('--prior', type=Path, default=Path('data/remind-001/anatomy.npz'))
    args = p.parse_args()
    manifest = json.loads((args.reconstruction/'manifest.json').read_text())
    anchor_index = manifest['anchor_frame']
    anchor_frame = manifest['frames'][anchor_index]
    anchor = np.load(args.truth_episode/'state'/f'{anchor_index:05d}.npz')
    prior = np.load(args.prior)
    rotation = np.asarray(anchor_frame['rotation'])
    directions = rotation[:, [0, 2]]
    reports = []
    for frame in manifest['coplanar_frames']:
        truth = np.load(args.truth_episode/'state'/f'{frame:05d}.npz')
        inferred = np.load(args.reconstruction/f'inferred_state_{frame:05d}.npz')
        actual, estimated = [], []
        for body in range(int(prior['body_count'])):
            points = prior[f'points_{body}']
            local = (points-np.asarray(anchor_frame['position']))@rotation
            inside = ((np.abs(local[:, 1])<.003)&(local[:, 0]>manifest['lateral_min_m']+.002)&
                (local[:, 0]<manifest['lateral_max_m']-.002)&(local[:, 2]>manifest['depth_min_m']+.002)&
                (local[:, 2]<manifest['depth_max_m']-.002))
            actual.append((truth[f'positions_{body}']-anchor[f'positions_{body}'])[inside]@directions)
            estimated.append((inferred[f'positions_{body}']-points)[inside]@directions)
        actual, estimated = np.concatenate(actual), np.concatenate(estimated)
        reports.append(dict(frame=frame, roi_nodes=len(actual),
            zero_increment_rmse_mm=float(np.sqrt(np.mean(actual**2))*1000),
            image_inverse_rmse_mm=float(np.sqrt(np.mean((actual-estimated)**2))*1000)))
    result = dict(scope='Independent evaluation of incremental in-plane motion in the observed slab only.',
        no_ground_truth_used_during_inference=True, frame_results=reports)
    (args.reconstruction/'independent_evaluation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
