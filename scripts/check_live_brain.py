"""Exercise the real running Isaac viewer; capture a browser video and evidence.

Optional verification dependency: playwright==1.55.0 and its Chromium browser.
Run this using .venv-ultrasound, while run_live_brain.sh is running.
"""
import argparse
import json
import os
from pathlib import Path
import urllib.request
from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8765')
    parser.add_argument('--output', type=Path, default=Path('outputs/live_gaussians_validation'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('DISPLAY', ':1')

    def status():
        return json.load(urllib.request.urlopen(args.url+'/api/status', timeout=5))

    errors, samples = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, args=['--no-sandbox', '--use-gl=angle', '--use-angle=gl'])
        context = browser.new_context(viewport=dict(width=1440, height=1000),
            record_video_dir=str(args.output/'recording'), record_video_size=dict(width=1440, height=1000))
        page = context.new_page()
        page.on('pageerror', lambda error: errors.append(str(error)))
        try:
            page.goto(args.url)
            page.wait_for_function('window.viewerStats && viewerStats.received>10', timeout=30000)
            page.locator('#reset').click()
            page.wait_for_timeout(1800)
            rest = status()
            assert rest['max_displacement_mm'] < .25, rest
            page.screenshot(path=str(args.output/'rest.png'))
            page.locator('#press').fill('5')
            page.locator('#press').dispatch_event('input')
            page.wait_for_timeout(2200)
            pressed = status()
            assert pressed['max_displacement_mm'] > 1.5, pressed
            assert pressed['min_jacobian'] > 0, pressed
            page.screenshot(path=str(args.output/'pressed.png'))
            page.locator('#pause').click()
            page.wait_for_timeout(200)
            paused_time = status()['simulation_time_s']
            page.wait_for_timeout(400)
            assert status()['simulation_time_s'] == paused_time
            page.locator('#pause').click()
            page.locator('#colour').select_option('1')
            page.wait_for_timeout(800)
            page.screenshot(path=str(args.output/'displacement.png'))
            page.locator('#colour').select_option('0')
            page.locator('#auto').click()
            for _ in range(14):
                page.wait_for_timeout(1000)
                samples.append(dict(solver=status(), viewer=page.evaluate('viewerStats')))
            page.locator('#tumor-only').check()
            page.wait_for_timeout(500)
            page.screenshot(path=str(args.output/'tumor.png'))
            page.locator('#tumor-only').uncheck()
            page.locator('#reset').click()
            page.wait_for_timeout(1800)
            reset = status()
            assert reset['max_displacement_mm'] < .25, reset
            assert not errors, errors
            assert all(s['viewer']['glError'] == 0 for s in samples)
            assert all(s['solver']['min_jacobian'] > 0 for s in samples)
            assert max(s['solver']['probe_position_m'][1] for s in samples)-min(s['solver']['probe_position_m'][1] for s in samples) > .04
            evidence = dict(rest=rest, pressed=pressed, reset=reset, samples=samples, browser_errors=errors)
            (args.output/'checks.json').write_text(json.dumps(evidence, indent=2)+'\n')
            print(json.dumps(dict(pressed_motion_mm=pressed['max_displacement_mm'],
                update_hz=[s['solver']['update_hz'] for s in samples],
                render_fps=[s['viewer']['renderFPS'] for s in samples], renderer=samples[-1]['viewer']['renderer'])))
        finally:
            video_path = page.video.path()
            context.close()
            browser.close()
            (args.output/'video_path.txt').write_text(str(video_path)+'\n')


if __name__ == '__main__':
    main()
