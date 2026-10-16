"""Verify and record the observed-image / reconstructed-geometry user flows."""
import argparse
import json
import os
from pathlib import Path
import urllib.request
from playwright.sync_api import sync_playwright


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--url', default='http://127.0.0.1:8765')
    p.add_argument('--output', type=Path, default=Path('outputs/ultrasound_reconstruction/verification'))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('DISPLAY', ':1')
    errors = []
    with sync_playwright() as driver:
        browser = driver.chromium.launch(headless=False, args=['--no-sandbox', '--use-gl=angle', '--use-angle=gl',
            '--disable-gpu-vsync', '--disable-frame-rate-limit'])
        context = browser.new_context(viewport=dict(width=1600, height=1050), record_video_dir=str(args.output/'recording'),
            record_video_size=dict(width=1600, height=1050))
        page = context.new_page()
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(args.url+'/reconstruction.html')
        page.wait_for_function('window.reconstructionStats && reconstructionStats.estimated', timeout=45000)
        page.locator('#magnify').select_option('4')
        page.wait_for_timeout(8000)
        reconstructed = page.evaluate('reconstructionStats')
        assert reconstructed['glError'] == 0
        assert reconstructed['renderFPS'] >= 15, reconstructed
        page.screenshot(path=str(args.output/'rf_reconstruction.png'))
        page.locator('#clip').select_option('scan')
        page.locator('#layer').select_option('volume')
        page.wait_for_timeout(1000)
        page.locator('#play').click()
        page.wait_for_timeout(500)
        page.locator('#frame').fill('30')
        page.locator('#frame').dispatch_event('input')
        page.wait_for_function('reconstructionStats.frame===30')
        page.wait_for_timeout(3000)
        page.screenshot(path=str(args.output/'image_only_volume.png'))
        page.locator('#frame').fill('12')
        page.locator('#frame').dispatch_event('input')
        page.wait_for_function('reconstructionStats.frame===12')
        image_pixels = page.evaluate("Array.from(document.getElementById('observed').getContext('2d').getImageData(0,0,320,384).data).filter((_,i)=>i%4===0).reduce((a,v)=>a+(v>30),0)")
        assert image_pixels > 100, image_pixels
        assert 'Held out' in page.locator('#reslice-label').inner_text()
        page.screenshot(path=str(args.output/'heldout_slice.png'))

        page.goto(args.url)
        page.wait_for_function('window.viewerStats && window.ultrasoundStats && ultrasoundStats.observable', timeout=120000)
        page.wait_for_timeout(1000)
        baseline = page.evaluate('({viewer:viewerStats, ultrasound:ultrasoundStats})')
        page.locator('#us-press').click()
        samples = []
        for _ in range(14):
            page.wait_for_timeout(1000)
            samples.append(page.evaluate('({viewer:viewerStats, ultrasound:ultrasoundStats})'))
        page.wait_for_function("viewerStats.sequence===ultrasoundStats.imageFrame && viewerStats.geometrySource==='estimate'", timeout=15000)
        paired = page.evaluate('({viewer:viewerStats, ultrasound:ultrasoundStats})')
        assert paired['ultrasound']['imageFrame'] == paired['ultrasound']['estimatedFrame']
        assert page.locator('#us-image').evaluate('(img)=>img.complete && img.naturalWidth>0')
        assert all(s['viewer']['glError'] == 0 for s in samples)
        assert max(s['viewer']['renderFPS'] for s in samples) >= 15
        page.screenshot(path=str(args.output/'live_ultrasound_estimate.png'))
        page.locator('#magnify').select_option('4')
        page.wait_for_timeout(1500)
        page.screenshot(path=str(args.output/'live_ultrasound_magnified.png'))
        page.locator('#geometry-source').select_option('simulation')
        page.wait_for_function("viewerStats.geometrySource==='simulation' && viewerStats.streamHZ>1")
        page.wait_for_timeout(1000)
        page.screenshot(path=str(args.output/'simulation_reference.png'))
        page.locator('#geometry-source').select_option('estimate')
        page.wait_for_timeout(1500)
        assert not errors, errors
        with urllib.request.urlopen(args.url+'/ultrasound/estimate.json') as response:
            inference = json.load(response)
        assert inference['min_volume_jacobian'] >= .3 and inference['max_inferred_motion_mm'] <= 4.001
        result = dict(recorded_reconstruction=reconstructed, live_baseline=baseline, live_samples=samples,
            paired_frame=paired, latest_inverse=inference, browser_errors=errors)
        (args.output/'checks.json').write_text(json.dumps(result, indent=2)+'\n')
        video = page.video.path()
        context.close()
        browser.close()
        (args.output/'video_path.txt').write_text(str(video)+'\n')
        print(json.dumps(dict(paired_frame=paired, latest_inverse=inference, browser_errors=errors)), flush=True)


if __name__ == '__main__':
    main()
