"""Compare the regenerated echoes at identical pose with one common display gain."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from deform_sim.ultrasound import log_compress


def main():
    p = argparse.ArgumentParser()
    p.add_argument('before', type=Path)
    p.add_argument('after', type=Path)
    p.add_argument('--frame', type=int, default=8)
    args = p.parse_args()
    stem = f'{args.frame:05d}'
    gain = max(json.loads((root/'ultrasound/display.json').read_text())['reference_amplitude'] for root in [args.before, args.after])
    font = lambda n: ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', n)
    image = Image.new('RGB', (1200, 900), '#101824')
    pen = ImageDraw.Draw(image)
    pen.text((40, 22), 'Tumour echo contrast: before and after', font=font(30), fill='white')
    pen.text((40, 68), 'Same tissue state, probe pose and display gain. No reference outlines.', font=font(22), fill='#a5b5ca')
    for index, root in enumerate([args.before, args.after]):
        data = np.load(root/'ultrasound'/(stem+'.npz'))
        config = json.loads((root/'ultrasound/acquisition.json').read_text())
        amplitudes = np.load(root/'reference.npz')['scatter_amplitudes']
        gray = log_compress(data['iq'], gain, config['dynamic_range_db'])
        frame = Image.fromarray(gray).convert('RGB').resize((520, 624), Image.Resampling.BICUBIC)
        x = 40+index*600
        pen.text((x, 126), ('Before: ' if index == 0 else 'After: ')+f'{amplitudes[1]:g}x brain', font=font(27), fill='white')
        image.paste(frame, (x, 174))
    pen.text((40, 824), 'Tumour scattering increased by 12 dB. This is an illustrative simulation setting.', font=font(22), fill='#a5b5ca')
    output = args.after/'video'/'contrast_comparison.png'
    image.save(output)
    print(output)


if __name__ == '__main__':
    main()
