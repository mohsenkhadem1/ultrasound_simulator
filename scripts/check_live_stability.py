"""Run repeated gentle scans and report positivity, travel and wall-clock rate."""
import argparse
import json
from pathlib import Path
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8765')
    parser.add_argument('--seconds', type=int, default=100)
    parser.add_argument('--min-time-ratio', type=float, default=.95)
    parser.add_argument('--output', type=Path, default=Path('outputs/live_stability.json'))
    args = parser.parse_args()
    if args.seconds < 3:
        parser.error('--seconds must be at least 3')

    def control(data):
        request = urllib.request.Request(args.url+'/api/control', data=json.dumps(data).encode(),
            headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=5):
            pass

    control(dict(reset=True))
    time.sleep(.5)
    control(dict(auto=True))
    samples = []
    start = time.monotonic()
    try:
        for i in range(args.seconds):
            with urllib.request.urlopen(args.url+'/api/status', timeout=5) as response:
                state = json.load(response)
            state['check_wall_seconds'] = time.monotonic()-start
            samples.append(state)
            assert not state['controls']['paused'] and 'error' not in state, state
            assert state['jacobian_scope'] == 'all FEM tetrahedra', state
            assert state['min_jacobian'] > 0 and state['actual_press_travel_mm'] < 3.2, state
            if i % 10 == 0:
                print(f'{i}s: sim={state["simulation_time_s"]:.2f}s, J={state["min_jacobian"]:.3f}', flush=True)
            time.sleep(1)
    finally:
        report = dict(samples=samples, wall_seconds=time.monotonic()-start)
        if len(samples) >= 2:
            report['simulation_wall_ratio'] = ((samples[-1]['simulation_time_s']-samples[0]['simulation_time_s']) /
                (samples[-1]['check_wall_seconds']-samples[0]['check_wall_seconds']))
            report['min_observed_volume_ratio'] = min(s['min_jacobian'] for s in samples)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    assert report['simulation_wall_ratio'] >= args.min_time_ratio, report['simulation_wall_ratio']
    print(f'PASS: simulation/wall time = {report["simulation_wall_ratio"]:.3f}', flush=True)


if __name__ == '__main__':
    main()
