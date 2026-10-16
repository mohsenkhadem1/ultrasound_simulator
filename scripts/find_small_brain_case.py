"""Find genuinely small ReMIND tumours; never scale a patient's segmentation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import sys
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
import numpy as np
import pydicom
sys.path.insert(0, str(Path(__file__).parent))
from prepare_brain import API, read_seg, transform


def clinical_metadata(path):
    """Read the public clinical workbook without an Excel dependency."""
    ns = {'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(path) as archive:
        strings = [''.join(t.itertext()) for t in ET.fromstring(archive.read('xl/sharedStrings.xml')).findall('m:si', ns)]
        rows = []
        for row in ET.fromstring(archive.read('xl/worksheets/sheet1.xml')).findall('.//m:sheetData/m:row', ns):
            values = ['']*32
            for cell in row:
                column = 0
                for char in cell.attrib['r']:
                    if not char.isalpha():break
                    column = column*26+ord(char)-ord('A')+1
                value = cell.find('m:v', ns)
                if value is not None:
                    values[column-1] = strings[int(value.text)] if cell.get('t') == 's' else value.text
            rows.append(values)
    header = rows[0]
    return {'ReMIND-'+r[0]:dict(histopathology=r[header.index('Histopathology')],
        clinical_whole_tumour_volume_ml=float(r[header.index('Preoperative Tumor Volume')])) for r in rows[1:] if r[0]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=Path('data/remind_case_search'))
    p.add_argument('--candidates', type=int, default=20)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output/'series.json'
    if not path.exists():
        data = json.load(urllib.request.urlopen(API+'getSeries?Collection=ReMIND&Modality=SEG&format=json', timeout=60))
        path.write_text(json.dumps(data, indent=2)+'\n')
    series = json.loads(path.read_text())
    clinical_path = args.output/'clinical.xlsx'
    if not clinical_path.exists():
        urllib.request.urlretrieve('https://www.cancerimagingarchive.net/wp-content/uploads/ReMIND-Dataset-Clinical-Data-September-2023.xlsx', clinical_path)
    clinical = clinical_metadata(clinical_path)
    cases = {}
    for s in series:
        if s.get('StudyDesc') == 'Preop':
            kind = s['SeriesDescription'].lower().split(' seg')[0]
            cases.setdefault(s['PatientID'], {}).setdefault(kind, []).append(s)
    candidates = []
    for patient, kinds in cases.items():
        if not all(k in kinds for k in ('tumor', 'cerebrum', 'ventricles')) or 'previous_resection_cavity' in kinds:
            continue
        # Prefer the same MRI reference as cerebrum over an ambiguous alternate tumour mask.
        brain_ref = kinds['cerebrum'][0]['SeriesDescription'].split('MR ref:')[-1].strip()
        tumour = sorted(kinds['tumor'], key=lambda s: (not s['SeriesDescription'].endswith(brain_ref), int(s['FileSize'])))[0]
        candidates.append(tumour)
    candidates.sort(key=lambda s: int(s['FileSize']))

    def inspect(s):
        folder = args.output/s['PatientID']; folder.mkdir(exist_ok=True)
        archive = folder/'tumor.zip'
        if not archive.exists():
            urllib.request.urlretrieve(API+'getImage?SeriesInstanceUID='+s['SeriesInstanceUID'], archive)
        with zipfile.ZipFile(archive) as z:
            name = next(n for n in z.namelist() if n.lower().endswith('.dcm'))
            (folder/'tumor.dcm').write_bytes(z.read(name))
        volume, affine = read_seg(folder/'tumor.dcm')
        occupied = np.argwhere(volume > 0)
        physical = transform(occupied, affine)
        report = dict(patient=s['PatientID'], volume_ml=float(len(occupied)*abs(np.linalg.det(affine[:3, :3]))/1000),
            extent_lps_mm=np.ptp(physical, axis=0).tolist(), series_uid=s['SeriesInstanceUID'],
            description=s['SeriesDescription'], segmentation_bytes=int(s['FileSize']))
        report.update(clinical.get(s['PatientID'], {}))
        clinical_volume = report.get('clinical_whole_tumour_volume_ml')
        report['matches_clinical_whole_tumour_volume'] = bool(clinical_volume and abs(report['volume_ml']/clinical_volume-1)<.2)
        report['pathology_is_tumour'] = not any(s in report.get('histopathology', '').lower() for s in ['non-tumor','reactive brain'])
        print(json.dumps(report), flush=True)
        return report

    with ThreadPoolExecutor(max_workers=4) as pool:
        reports = list(pool.map(inspect, candidates[:args.candidates]))
    reports.sort(key=lambda r: r['volume_ml'])
    result = dict(source='https://www.cancerimagingarchive.net/collection/remind/',
        selection='Preoperative SEG with cerebrum and ventricles, no prior resection; measure physical voxel volume and audit against clinical whole-tumour volume and pathology before choosing.', cases=reports)
    (args.output/'candidates.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Ranked cases:', [(r['patient'], round(r['volume_ml'], 3)) for r in reports])


if __name__ == '__main__':
    main()
