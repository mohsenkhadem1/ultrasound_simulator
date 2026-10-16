"""Exercise the live, paired ultrasound/Isaac tumour comparison and record it."""
import argparse
import json
import os
from pathlib import Path
import time
import urllib.request
from playwright.sync_api import sync_playwright


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--url', default='http://127.0.0.1:8765')
    p.add_argument('--output', type=Path, default=Path('outputs/tumour_comparison'))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    url = args.url.rstrip('/')
    for _ in range(90):
        try:
            with urllib.request.urlopen(url+'/comparison/static.bin', timeout=1) as response:
                if response.status == 200:
                    break
        except Exception:
            time.sleep(1)
    else:
        raise RuntimeError('Isaac comparison worker did not start')
    os.environ.setdefault('DISPLAY', ':1')
    errors = []
    with sync_playwright() as driver:
        browser = driver.chromium.launch(headless=False, args=['--no-sandbox', '--use-gl=angle', '--use-angle=gl',
            '--disable-gpu-vsync', '--disable-frame-rate-limit'])
        context = browser.new_context(viewport=dict(width=1600, height=1050),
            record_video_dir=str(args.output/'recording'), record_video_size=dict(width=1600, height=1050))
        page = context.new_page()
        page.on('pageerror', lambda error: errors.append(str(error)))
        try:
            page.goto(url+'/comparison.html')
            page.wait_for_function('window.comparisonStats && comparisonStats.observable', timeout=120000)
            epoch = page.evaluate('comparisonStats.epoch')
            page.locator('#reset').click()
            page.wait_for_timeout(1500)
            page.wait_for_function('(epoch)=>comparisonStats.observable && comparisonStats.epoch>epoch', arg=epoch, timeout=120000)
            baseline = page.evaluate('comparisonStats')
            page.screenshot(path=str(args.output/'initial.png'))
            page.locator('#demo').click()
            samples = []
            for i in range(24):
                page.wait_for_timeout(1000)
                sample = page.evaluate('comparisonStats')
                samples.append(sample)
                if i == 8:
                    page.screenshot(path=str(args.output/'pressing.png'))
                if i == 12:
                    page.locator('#focus').click()
                if i == 14:
                    page.screenshot(path=str(args.output/'tumour_closeup.png'))
                if i == 18:
                    page.locator('#focus').click()
            page.screenshot(path=str(args.output/'comparison.png'))
            assert not errors, errors
            assert all(s['frame'] == s['imageFrame'] == s['predictionFrame'] == s['truthFrame'] for s in samples)
            assert all(s['glErrors'] == [0, 0] for s in samples)
            assert all(s['truthRepresentation'] == 'triangle mesh' for s in samples)
            assert max(s['renderFPS'] for s in samples) >= 15
            assert len({s['frame'] for s in samples}) >= 8, 'Comparison is frozen'
            assert max(s['metrics']['static']['surface_vertex_rmse_mm'] for s in samples) > .1, 'Isaac tumour did not deform'
            assert any(s['observable'] and abs(s['metrics']['predicted']['surface_vertex_rmse_mm']-
                s['metrics']['static']['surface_vertex_rmse_mm']) > .005 for s in samples), 'Prediction did not change from initial contour'
            with urllib.request.urlopen(url+'/api/status') as response:
                physics = json.load(response)
            evidence = dict(baseline=baseline, samples=samples, physics=physics, browser_errors=errors)
            (args.output/'checks.json').write_text(json.dumps(evidence, indent=2)+'\n')
            print(json.dumps(dict(frames=len({s['frame'] for s in samples}),
                peak_tumour_motion_mm=max(s['metrics']['static']['surface_vertex_rmse_mm'] for s in samples),
                inference_hz=samples[-1]['inferenceHZ'], render_fps=samples[-1]['renderFPS'], browser_errors=errors)))
        finally:
            video=page.video.path()
            context.close()
            browser.close()
            (args.output/'video_path.txt').write_text(str(video)+'\n')


if __name__ == '__main__':
    main()
