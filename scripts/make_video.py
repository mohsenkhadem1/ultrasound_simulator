"""Compose matching Isaac and acoustic captures into an annotated MP4/GIF."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('episode', type=Path)
    parser.add_argument('--no-tumor-outline', action='store_true', help='show ultrasound contrast without the reference boundary overlay')
    args = parser.parse_args()
    episode = args.episode
    output = episode/'video'
    output.mkdir(exist_ok=False)
    state_files = sorted(p for p in (episode/'state').glob('*.npz') if p.stem.isdecimal())
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    bold_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
    font = lambda n, bold=False: ImageFont.truetype(bold_path if bold else font_path, n)
    title, heading, text, small = font(35, True), font(22, True), font(21), font(17)
    bg, panel, fg, muted = '#0b1220', '#131f31', '#f1f5fc', '#a5b5ca'
    teal, orange, blue = '#30c59b', '#eea03e', '#69a4ff'
    reference = np.load(episode/'reference.npz')
    metadata = json.loads((episode/'metadata.json').read_text())
    pressing = metadata.get('probe_motion') in ('press', 'sweep', 'scan')
    scanning = metadata.get('probe_motion') == 'scan'
    show_outline = scanning and not args.no_tumor_outline
    sweeping = metadata.get('probe_motion') == 'sweep'
    anatomy = bool(metadata.get('anatomy'))
    count = int(reference['body_count']) if 'body_count' in reference else 2
    acquisition = json.loads((episode/'ultrasound'/'acquisition.json').read_text())
    depth_min, depth_max = acquisition.get('depth_min', .002)*1000, acquisition.get('depth_max', .03)*1000
    duration = float(metadata.get('duration_seconds', 2.))
    outlines = {}
    if scanning:
        from deform_sim.slice_geometry import tumor_segments
        outlines = {path.stem: tumor_segments(reference, np.load(path)) for path in state_files}
        (output/'tumor_slices.json').write_text(json.dumps([
            dict(frame=path.stem, time=float(np.load(path)['time']), intersection_segments=len(outlines[path.stem]))
            for path in state_files], indent=2)+'\n')

    def phase(t):
        if t < .5:
            return 'PROBE PRESS' if pressing else 'COMPRESS'
        if t < 1.2:
            return 'PROBE SWEEP' if sweeping else 'HOLD'
        if t < 1.6:
            return 'PROBE LIFT' if pressing else 'RELEASE'
        return 'RECOVERY'

    def compose(index, cycle):
        path = state_files[index]
        state = np.load(path)
        t = float(state['time'])
        canvas = Image.new('RGB', (1600, 900), bg)
        d = ImageDraw.Draw(canvas)
        d.text((40, 25), 'Patient brain anatomy + moving ultrasound' if anatomy else ('Probe-driven compression + synchronized ultrasound' if pressing else 'Tissue deformation + synchronized ultrasound'), font=title, fill=fg)
        subtitle = 'Scan into the tumour, press gently, then scan past it. Ultrasound matches the moving probe.' if scanning else ('ReMIND-001 brain and glioma. Illustrative heterogeneous stiffness; simulated B-mode.' if anatomy else ('Ultrasound head moves down, compresses, and lifts away.' if pressing else 'Compression tool moves. Ultrasound probe stays fixed.'))
        if metadata.get('acoustic_contrast'):
            value = metadata['acoustic_contrast']['new_backscatter_amplitude']
            subtitle = f'Visibility demonstration: tumour echo amplitude {value:g}x brain. Same recorded scan and gentle press.'
        d.text((42, 80), subtitle, font=text, fill=muted)
        d.rounded_rectangle((1310, 38, 1560, 87), radius=12, fill='#19483f')
        d.text((1330, 49), str(state['phase']) if scanning else phase(t), font=heading, fill=teal)
        d.rounded_rectangle((30, 134, 994, 783), radius=16, fill=panel)
        d.rounded_rectangle((1014, 134, 1570, 783), radius=16, fill=panel)
        d.text((52, 150), 'ISAAC SIM  /  3D CUTAWAY', font=heading, fill=fg)
        scene = Image.open(episode/'scene_frames'/f'{path.stem}.png').convert('RGB')
        scene.thumbnail((922, 548), Image.Resampling.LANCZOS)
        canvas.paste(scene, (50+(922-scene.width)//2, 192+(548-scene.height)//2))
        legend = [(52, orange, 'Moving ultrasound probe'), (420, '#bc70c7', 'Patient glioma'), (704, blue, 'Ventricles')] if anatomy else ([(52, orange, 'Moving ultrasound probe'), (520, blue, 'Stiff inclusion')] if pressing else [(52, teal, 'Compression tool'), (380, orange, 'Fixed US probe'), (684, blue, 'Stiff inclusion')])
        if scanning:
            legend = [(52, orange, 'Probe'), (210, '#bc70c7', 'Tumour'), (380, blue, 'Ventricles'), (610, '#19d2da', 'Imaging plane')]
        for x, color, label in legend:
            d.ellipse((x, 750, x+13, 763), fill=color)
            d.text((x+22, 744), label, font=small, fill=fg)

        d.text((1036, 150), 'MATCHING B-MODE', font=heading, fill=fg)
        us = Image.open(episode/'ultrasound'/f'{path.stem}.png').convert('RGB')
        # Preserve the physical image aspect: lateral 40 mm, depth range 28 mm.
        pixels_per_mm = min(460/40, 340/(depth_max-depth_min))
        image_width, image_height = round(40*pixels_per_mm), round((depth_max-depth_min)*pixels_per_mm)
        image_left = 1080+(460-image_width)//2
        us = us.resize((image_width, image_height), Image.Resampling.BICUBIC)
        if show_outline:
            pen = ImageDraw.Draw(us)
            for segment in outlines[path.stem]:
                pixel = np.column_stack(((segment[:, 0]+.02)/.04*(image_width-1),
                                         (segment[:, 1]*1000-depth_min)/(depth_max-depth_min)*(image_height-1)))
                pen.line([tuple(p) for p in pixel], fill='#ffe162', width=2)
        canvas.paste(us, (image_left, 250))
        for value in (-20, -10, 0, 10, 20):
            x = image_left+(value+20)/40*image_width
            d.line((x, 238, x, 247), fill=muted, width=1)
            d.text((x-15, 214), str(value), font=small, fill=muted)
        d.text((1205, 186), 'Lateral position (mm)', font=small, fill=muted)
        for depth in [depth_min]+list(range(10, round(depth_max)+1, 10)):
            y = 250+(depth-depth_min)/(depth_max-depth_min)*image_height
            d.line((image_left-10, y, image_left-3, y), fill=muted, width=1)
            d.text((image_left-43, y-9), f'{depth:g}', font=small, fill=muted)
        d.text((1080, 250+image_height+15), f'Depth: {depth_min:g}-{depth_max:g} mm | 5 MHz linear array', font=small, fill=muted)
        motion = max(np.linalg.norm(state[f'positions_{i}']-reference[f'points_{i}'], axis=1).max() for i in range(count))
        if scanning:
            visible = len(outlines[path.stem]) > 0
            d.text((1040, 646), 'Tumour in slice: '+('YES' if visible else 'NO'), font=text, fill='#ffe162' if visible else muted)
        else:
            d.text((1040, 646), f'Simulation time     {t:4.2f} s', font=text, fill=fg)
        travel_label = 'Probe travel' if pressing else 'Tool travel'
        d.text((1040, 684), f'{travel_label:<24} {float(state["command"])*1000:4.2f} mm', font=text, fill=fg)
        d.text((1040, 722), f'Peak tissue motion  {motion*1000:4.2f} mm', font=text, fill=fg)
        footer = f'0.5x playback  |  Replay {cycle}/2  |  Time {t:.2f} s  |  Paired frame {path.stem}'
        if show_outline:
            footer += '  |  Yellow outline: simulation reference, not an ultrasound prediction'
        elif metadata.get('acoustic_contrast'):
            footer += '  |  Stronger tumour echoes; one constant display gain; no boundary overlay'
        d.text((40, 808), footer, font=small, fill=muted)
        segments = [(0, .5, 'PROBE PRESS' if pressing else 'COMPRESS'), (.5, 1.2, 'SWEEP' if sweeping else 'HOLD'), (1.2, 1.6, 'PROBE LIFT' if pressing else 'RELEASE'), (1.6, 2., 'RECOVERY')]
        if scanning:
            segments = [(0, .08*duration, 'APPROACH'), (.08*duration, .36*duration, 'SCAN IN'),
                        (.36*duration, .60*duration, 'PRESS / EASE'), (.60*duration, .88*duration, 'SCAN OUT'),
                        (.88*duration, duration, 'LIFT')]
        for start, end, label in segments:
            x0, x1 = 40+start/duration*1520, 40+end/duration*1520
            d.rounded_rectangle((x0+2, 847, x1-2, 861), radius=4, fill='#2b3b51')
            d.text((x0+8, 870), label, font=small, fill=muted)
        x = 40+t/duration*1520
        d.ellipse((x-7, 847-1, x+7, 861+1), fill=teal)
        return canvas

    for path in state_files:
        assert (episode/'scene_frames'/f'{path.stem}.png').exists(), f'missing scene {path.stem}'
        assert (episode/'ultrasound'/f'{path.stem}.png').exists(), f'missing ultrasound {path.stem}'
        acoustic = np.load(episode/'ultrasound'/f'{path.stem}.npz')
        state = np.load(path)
        assert abs(float(acoustic['time'])-float(state['time'])) < 1e-9, 'unmatched frame times'
        if 'probe_position' in state:
            np.testing.assert_allclose(acoustic['probe_position'], state['probe_position'])
            np.testing.assert_allclose(acoustic['probe_rotation'], state['probe_rotation'])
    fps = 30
    timestamps = [float(np.load(path)['time']) for path in state_files]
    sample_dt = float(np.median(np.diff(timestamps))) if len(timestamps) > 1 else .05
    frames_per_sample = fps*sample_dt/.5
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
               '-s', '1600x900', '-pix_fmt', 'rgb24', '-r', str(fps), '-i', '-', '-an',
               '-vcodec', 'libx264', '-preset', 'medium', '-crf', '19', '-pix_fmt', 'yuv420p',
               '-movflags', '+faststart', str(output/'simulation_ultrasound.mp4')]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    preview = []
    frame_count = 0
    try:
        for cycle in (1, 2):
            for index in range(len(state_files)):
                canvas = compose(index, cycle)
                rgb = canvas.tobytes()
                # Alternate repeat counts when sample periods are fractional
                # at the encoded rate, preserving the labelled playback speed.
                repeats = max(1, round((index+1)*frames_per_sample)-round(index*frames_per_sample))
                for _ in range(repeats):
                    process.stdin.write(rgb)
                    frame_count += 1
                if cycle == 1:
                    preview.append(canvas.resize((960, 540), Image.Resampling.LANCZOS))
        process.stdin.close()
        error = process.stderr.read().decode()
        status = process.wait()
        if status:
            raise RuntimeError(error)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    preview[0].save(output/'preview.gif', save_all=True, append_images=preview[1:], duration=round(sample_dt/.5*1000), loop=0)
    compose(len(state_files)//2, 1).save(output/'poster.png')
    (output/'video.json').write_text(json.dumps(dict(output_fps=fps, encoded_frames=frame_count,
        duration_seconds=frame_count/fps, source_frames=len(state_files), source_fps=1/sample_dt,
        playback_speed=.5, replay_cycles=2, tumor_reference_outline=show_outline,
        paired_by='identical snapshot id and simulation timestamp'), indent=2)+'\n')
    print(output/'simulation_ultrasound.mp4', flush=True)


if __name__ == '__main__':
    main()
