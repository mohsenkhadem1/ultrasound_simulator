"""Open the viewer with explicit GPU presentation flags on this desktop."""
import argparse
import os
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--url', default='http://127.0.0.1:8765')
    p.add_argument('--reconstruction', action='store_true')
    args = p.parse_args()
    url = args.url.rstrip('/')+('/reconstruction.html' if args.reconstruction else '/')
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as context:
            executable = context.chromium.executable_path
        profile = Path.home()/'.cache/deform-ultrasound-viewer'
        log = (Path('/tmp')/'deform-ultrasound-browser.log').open('a')
        process = subprocess.Popen([executable, '--no-sandbox', '--no-first-run', '--no-default-browser-check',
            '--disable-dev-shm-usage', '--use-gl=angle', '--use-angle=gl', '--disable-gpu-vsync',
            '--disable-frame-rate-limit', '--user-data-dir='+str(profile), '--window-size=1600,1050',
            '--new-window', url], env=dict(os.environ, DISPLAY=os.environ.get('DISPLAY', ':1')),
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(f'Opened {url} (browser PID {process.pid})')
    except ImportError:
        subprocess.Popen(['xdg-open', url], start_new_session=True)
        print(url)


if __name__ == '__main__':
    main()
