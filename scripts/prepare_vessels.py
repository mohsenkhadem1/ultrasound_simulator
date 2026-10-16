"""Retrieve one real TopBrain segmentation using HTTP ranges, export labelled meshes.

This is a different subject from ReMIND. No invented patient registration is applied.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

import meshio
import nibabel as nib
import numpy as np
from remotezip import RemoteZip
from skimage.measure import marching_cubes

URL = 'https://zenodo.org/api/records/16878417/files/TopBrain_Data_Release_Batches1n2_081425.zip/content'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--subject', default='001')
    p.add_argument('--output', type=Path, default=Path('data/topbrain-001'))
    args = p.parse_args()
    raw = args.output/'raw'
    raw.mkdir(parents=True, exist_ok=True)
    volume_name = f'topcow_ct_{args.subject}.nii.gz'
    if not (raw/volume_name).exists():
        with RemoteZip(URL, timeout=60) as z:
            for entry in z.infolist():
                name = entry.filename
                if name.endswith('/'+volume_name) and '/labelsTr_topbrain_ct/' in name or name.endswith('/License.txt') or name.endswith('/labelmap_topbrain_ct.txt'):
                    print('Downloading', name, flush=True)
                    (raw/Path(name).name).write_bytes(z.read(name))
    image = nib.load(raw/volume_name)
    volume = np.asarray(image.dataobj)
    names = {}
    for line in (raw/'labelmap_topbrain_ct.txt').read_text().splitlines():
        match = re.match(r'\s*(\d+)\s+.*"([^"]+)"', line)
        if match:
            names[int(match[1])] = match[2]
    info = []
    for group, ids in [('arteries', range(1, 35)), ('veins', range(35, 41))]:
        # Group includes true segmented vessel surfaces, not procedural tubes.
        mask = np.isin(volume, list(ids))
        vertices, faces, _, _ = marching_cubes(np.pad(mask.astype(np.uint8), 1), .5)
        points = nib.affines.apply_affine(image.affine, vertices-1)*.001
        meshio.write(args.output/(group+'.ply'), meshio.Mesh(points, [('triangle', faces)]))
        np.savez_compressed(args.output/(group+'.npz'), points=points, faces=faces,
                            coordinate_system='NIfTI RAS, metres; TopBrain CT subject '+args.subject)
        info.append(dict(group=group, structures=[names[i] for i in ids if i in names and np.any(volume == i)],
                         vertices=len(vertices), triangles=len(faces)))
    provenance = dict(dataset='TopBrain Challenge Data Release', version='v2, 14 August 2025', subject=args.subject,
                      source='https://zenodo.org/records/16878417', downloaded_archive_member=volume_name,
                      published_archive_md5='b703ea31cd1f0e7115a5d3e6e61f59b3',
                      member_sha256=hashlib.sha256((raw/volume_name).read_bytes()).hexdigest(),
                      license='Non-commercial use with attribution; commercial use requires data owner permission. See raw/License.txt.',
                      authors='TopBrain Challenge Organizers; Kaiyuan Yang, Pengcheng Shi and colleagues',
                      meshes=info, registration='Separate subject from ReMIND; not aligned, not included in the patient simulation.')
    (args.output/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main()
