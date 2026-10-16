"""Download real ReMIND segmentations and build a bonded, heterogeneous FEM mesh.

Run in .venv-ultrasound. The resulting NPZ needs only NumPy in Isaac Sim.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
import urllib.parse
import urllib.request
import zipfile

import numpy as np
import pydicom
from scipy.ndimage import map_coordinates, gaussian_filter, label
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes
import meshio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from deform_sim.phantom import BodyMesh

API = 'https://services.cancerimagingarchive.net/nbia-api/services/v1/'


def download_segmentations(destination, patient):
    destination.mkdir(parents=True, exist_ok=True)
    url = API+'getSeries?'+urllib.parse.urlencode(dict(Collection='ReMIND', PatientID=patient, Modality='SEG', format='json'))
    series = json.load(urllib.request.urlopen(url, timeout=60))
    (destination/'series.json').write_text(json.dumps(series, indent=2)+'\n')
    sources = []
    for structure in ('cerebrum', 'tumor', 'ventricles'):
        candidates = [s for s in series if s['SeriesDescription'].lower().startswith(structure+' seg') and s.get('StudyDesc') == 'Preop']
        if not candidates:
            raise ValueError(f'{patient} has no preoperative {structure} segmentation')
        s = candidates[0]
        archive = destination/(structure+'.zip')
        if not archive.exists():
            print('Downloading', patient, structure, flush=True)
            partial = archive.with_suffix('.partial')
            urllib.request.urlretrieve(API+'getImage?SeriesInstanceUID='+s['SeriesInstanceUID'], partial)
            partial.rename(archive)
        with zipfile.ZipFile(archive) as z:
            names = [name for name in z.namelist() if name.lower().endswith('.dcm')]
            if len(names) != 1:
                raise ValueError('expected one DICOM SEG per series')
            (destination/(structure+'.dcm')).write_bytes(z.read(names[0]))
            if 'LICENSE' in z.namelist():
                (destination/'LICENSE').write_bytes(z.read('LICENSE'))
        sources.append(dict(structure=structure, series_uid=s['SeriesInstanceUID'],
                            description=s['SeriesDescription'], sha256=hashlib.sha256(archive.read_bytes()).hexdigest()))
    return sources


def read_seg(path):
    """DICOM SEG frames -> [frame,row,column] with physical LPS millimetres."""
    d = pydicom.dcmread(path)
    if len(d.SegmentSequence) != 1:
        raise ValueError('this importer requires one anatomical segment per SEG')
    shared = d.SharedFunctionalGroupsSequence[0]
    orient = np.asarray(shared.PlaneOrientationSequence[0].ImageOrientationPatient, float).reshape(2, 3)
    spacing = np.asarray(shared.PixelMeasuresSequence[0].PixelSpacing, float)
    positions = np.array([f.PlanePositionSequence[0].ImagePositionPatient for f in d.PerFrameFunctionalGroupsSequence], float)
    data = d.pixel_array.astype(np.uint8)
    step = (positions[-1]-positions[0])/(len(positions)-1)
    expected = positions[0]+np.arange(len(positions))[:, None]*step
    if np.max(np.linalg.norm(positions-expected, axis=1)) > .01:
        raise ValueError('irregular or multi-segment DICOM frame positions')
    affine = np.eye(4)
    affine[:3, :3] = np.column_stack((step, orient[1]*spacing[0], orient[0]*spacing[1]))
    affine[:3, 3] = positions[0]
    return data, affine


def transform(points, affine):
    return points @ affine[:3, :3].T+affine[:3, 3]


def sample_mask(volume, affine, points):
    coords = transform(points, np.linalg.inv(affine))
    return map_coordinates(volume, coords.T, order=0, mode='constant', cval=0) > 0


def surface(volume, affine, stride=1):
    coarse = volume[::stride, ::stride, ::stride]
    # Smooth the segmentation by less than one downsampled voxel; retain sulci.
    field = gaussian_filter(np.pad(coarse.astype(float), 1), .6)
    v, faces, _, _ = marching_cubes(field, .5)
    return transform((v-1)*stride, affine), faces.astype(np.int32)


def embed_skin(points, bodies):
    nodes = np.concatenate([b.points for b in bodies])
    distance, ids = cKDTree(nodes).query(points, k=4)
    weights = 1/np.maximum(distance, 1e-8)**2
    weights /= weights.sum(axis=1, keepdims=True)
    return ids.astype(np.int32), weights


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--patient', default='ReMIND-001')
    p.add_argument('--output', type=Path, default=Path('data/remind-001'))
    p.add_argument('--spacing-mm', type=float, default=5.)
    p.add_argument('--brain-youngs', type=float, default=3000.)
    p.add_argument('--tumor-ratio', type=float, default=.7, help='illustrative, not a measured patient property')
    p.add_argument('--variation', type=float, default=.3, help='fractional spatial variation of brain stiffness')
    args = p.parse_args()
    if args.spacing_mm <= 0 or args.brain_youngs <= 0 or args.tumor_ratio <= 0 or not 0 <= args.variation < 1:
        p.error('invalid mesh spacing or material parameters')
    sources = download_segmentations(args.output/'raw', args.patient)
    volumes = {name: read_seg(args.output/'raw'/(name+'.dcm')) for name in ('cerebrum', 'tumor', 'ventricles')}
    tumor_indices = np.argwhere(volumes['tumor'][0] > 0)[::8]
    tumor_lps = transform(tumor_indices, volumes['tumor'][1])
    overlap = float(sample_mask(*volumes['cerebrum'], tumor_lps).mean())
    if overlap < .8:
        raise ValueError(f'tumour/cerebrum physical-frame overlap only {overlap:.1%}; inspect registration')
    skins = {name: surface(*volumes[name], stride=3 if name == 'cerebrum' else 1) for name in volumes}
    tumor_center = skins['tumor'][0].mean(axis=0)
    brain_points = skins['cerebrum'][0]
    _, nearest = cKDTree(brain_points).query(tumor_center)
    entry = brain_points[nearest]
    z = entry-tumor_center
    z /= np.linalg.norm(z)
    # Rotate the approach to point down from +z; preserve metric scale exactly.
    x = np.cross([0., 0., 1.], z)
    if np.linalg.norm(x) < .1:
        x = np.cross([0., 1., 0.], z)
    x /= np.linalg.norm(x)
    rotation = np.stack((x, np.cross(z, x), z))
    world = np.eye(4)
    world[:3, :3] = rotation*.001
    world[:3, 3] = -rotation@entry*.001
    rotated = transform(brain_points, world)
    world[2, 3] -= rotated[:, 2].min()
    for name, (vertices, faces) in skins.items():
        skins[name] = (transform(vertices, world), faces)
        meshio.write(args.output/(name+'.ply'), meshio.Mesh(skins[name][0], [('triangle', faces)]))
    brain_points = skins['cerebrum'][0]
    spacing = args.spacing_mm*.001
    origin = np.floor(brain_points.min(axis=0)/spacing)*spacing-spacing
    shape = np.ceil((brain_points.max(axis=0)-origin)/spacing).astype(int)+1
    ijk = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing='ij'), axis=-1).reshape(-1, 3)
    nodes = origin+ijk*spacing
    cells = np.stack(np.meshgrid(*[np.arange(s-1) for s in shape], indexing='ij'), axis=-1).reshape(-1, 3)
    centers = origin+(cells+.5)*spacing
    lps = transform(centers, np.linalg.inv(world))
    inside = sample_mask(*volumes['cerebrum'], lps)
    tumor = sample_mask(*volumes['tumor'], lps)
    ventricle = sample_mask(*volumes['ventricles'], lps)
    occupied = ((inside | tumor) & ~ventricle).reshape(tuple(shape-1))
    # Discard any isolated fragments generated by coarse sampling.
    components, _ = label(occupied)
    counts = np.bincount(components.ravel()); counts[0] = 0
    occupied = (components == np.argmax(counts)).ravel()
    field = (np.sin(centers[:, 0]/.018)+np.cos(centers[:, 1]/.023)+np.sin(centers[:, 2]/.021))/3
    bands = np.digitize(field, [-.2, .2])
    # Region 1 is tumour; other regions are spatial stiffness bands, not GM/WM labels.
    regions = np.choose(bands, [2, 0, 3]); regions[tumor] = 1
    tets, labels = [], []
    for order in itertools.permutations(range(3)):
        corners = np.zeros((4, 3), int)
        for k, axis in enumerate(order):
            corners[k+1] = corners[k]; corners[k+1, axis] += 1
        ids = np.ravel_multi_index((cells[occupied, None, :]+corners).transpose(2, 0, 1), tuple(shape))
        if np.linalg.det(corners[1:]-corners[0]) < 0:
            ids[:, [1, 2]] = ids[:, [2, 1]]
        tets.append(ids); labels.append(regions[occupied])
    tets = np.concatenate(tets).astype(np.int32); labels = np.concatenate(labels)
    bodies, payload = [], {}
    for i in range(4):
        ids, inverse = np.unique(tets[labels == i], return_inverse=True)
        if not len(ids):
            raise ValueError('empty material region; reduce grid spacing')
        body = BodyMesh(nodes[ids], inverse.reshape(-1, 4).astype(np.int32), ids)
        bodies.append(body)
        payload.update({f'points_{i}': body.points, f'tets_{i}': body.tetrahedra, f'ids_{i}': body.global_ids})
    for i, name in enumerate(('cerebrum', 'tumor', 'ventricles')):
        points, faces = skins[name]
        ids, weights = embed_skin(points, bodies)
        payload.update({f'display_points_{i}': points, f'display_faces_{i}': faces,
                        f'display_nodes_{i}': ids, f'display_weights_{i}': weights})
    # Probe starts above the highest FEM point in its swept 40 x 8 mm footprint.
    all_points = np.concatenate([b.points for b in bodies])
    footprint = (np.abs(all_points[:, 0]) <= .027) & (np.abs(all_points[:, 1]) <= .012)
    probe_height = float(all_points[footprint, 2].max()+.004)
    moduli = args.brain_youngs*np.array([1, args.tumor_ratio, 1-args.variation, 1+args.variation])
    np.savez_compressed(args.output/'anatomy.npz', body_count=4, **payload,
                        youngs_modulus=moduli, scatter_amplitudes=[1., 1.5, 1., 1.],
                        scatter_bounds=[[-.035, -.015, probe_height-.05], [.035, .018, probe_height]],
                        probe_height=probe_height, lps_mm_to_world=world)
    info = dict(dataset='ReMIND', patient=args.patient, source='https://www.cancerimagingarchive.net/collection/remind/',
                doi='10.7937/3RAG-D070', license='CC BY 4.0', source_series=sources,
                material_regions=['brain middle band', 'glioma', 'brain softer band', 'brain firmer band'],
                youngs_modulus_pa=moduli.tolist(), spacing_mm=args.spacing_mm,
                tumor_inside_cerebrum_fraction=overlap,
                nodes=[len(b.points) for b in bodies], tetrahedra=[len(b.tetrahedra) for b in bodies],
                probe_height_m=probe_height, tumor_depth_m=float(probe_height-skins['tumor'][0][:, 2].mean()),
                limitations=['Stiffness values and spatial bands are illustrative, not measured patient properties.',
                             'Acoustic scatterers are sampled only in the local swept imaging region; far-field scattering is omitted.',
                             'Coarse Cartesian tetrahedral mesh; fine segmentation surfaces use inverse-distance displacement interpolation.',
                             'Ventricles are cavities with a visual skin; no CSF fluid dynamics, skull or meninges.',
                             'No vessels in this case segmentation. TopBrain vessels are a different subject and kept separate.'])
    (args.output/'provenance.json').write_text(json.dumps(info, indent=2)+'\n')
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
