"""Loopback-only live Gaussian stream and controls, independent of the Kit loop."""
import asyncio
from contextlib import suppress
import json
from pathlib import Path
import threading
import numpy as np


class LiveServer:
    def __init__(self, model, port=8765, autoplay=False):
        self.model, self.port = model, port
        self.controls = dict(x_mm=0., y_mm=0., press_mm=0., yaw_deg=0., auto=autoplay,
                             paused=False, reset=False, stop=False)
        self.lock = threading.Lock()
        self.latest = None
        self.sequence = 0
        self.status = dict(ready=False, connected_clients=0)
        self.ready = threading.Event()
        self.error = None
        self.static = np.column_stack((model['colors'], model['opacity'], model['semantic'],
                                       model['youngs'], model['means'])).astype(np.float32).tobytes()

    def command(self, data):
        if not isinstance(data, dict):
            raise ValueError('controls must be a JSON object')
        numeric = dict(x_mm=(-6, 6), y_mm=(-32, 32), press_mm=(0, 5), yaw_deg=(-15, 15))
        clean = {}
        for key, value in data.items():
            if key in numeric:
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                    raise ValueError(f'{key} must be a finite number')
                clean[key] = float(np.clip(value, *numeric[key]))
            elif key in self.controls:
                if not isinstance(value, bool):
                    raise ValueError(f'{key} must be boolean')
                clean[key] = value
        with self.lock:
            self.controls.update(clean)
        return clean

    def get_controls(self):
        with self.lock:
            controls = self.controls.copy()
            self.controls['reset'] = False
            return controls

    def publish(self, packet, status):
        with self.lock:
            self.latest = packet
            self.sequence += 1
            if 'error' not in status:
                self.status.pop('error', None)
            self.status.update(status, ready=True)

    async def _serve(self):
        from aiohttp import web, WSMsgType
        root = Path(__file__).with_name('viewer')
        app = web.Application(client_max_size=4096)

        async def static_file(request):
            name = request.match_info.get('name', 'index.html')
            if name not in ('index.html', 'app.js', 'style.css'):
                raise web.HTTPNotFound()
            return web.FileResponse(root/name)

        async def model_data(request):
            return web.Response(body=self.static, content_type='application/octet-stream')

        async def status(request):
            with self.lock:
                data = dict(self.status, sequence=self.sequence, controls=self.controls.copy(),
                            gaussian_count=len(self.model['means']), soft_count=int(self.model['soft_count']))
            return web.json_response(data)

        async def control(request):
            try:
                self.command(await request.json())
            except (ValueError, TypeError) as e:
                return web.json_response(dict(error=str(e)), status=400)
            return web.json_response(dict(ok=True))

        async def socket(request):
            ws = web.WebSocketResponse(max_msg_size=4096, heartbeat=30, compress=False)
            await ws.prepare(request)
            with self.lock:
                self.status['connected_clients'] += 1
            await ws.send_json(dict(type='init', protocol=1, count=len(self.model['means']),
                                    soft_count=int(self.model['soft_count'])))

            async def send_latest():
                previous = -1
                while not ws.closed:
                    with self.lock:
                        seq, packet = self.sequence, self.latest
                    if seq != previous and packet is not None:
                        try:
                            await ws.send_bytes(packet)
                        except ConnectionResetError:
                            return
                        previous = seq
                    await asyncio.sleep(.008)
            sender = asyncio.create_task(send_latest())
            try:
                async for message in ws:
                    if message.type == WSMsgType.TEXT:
                        try:
                            self.command(json.loads(message.data))
                        except (ValueError, TypeError) as e:
                            await ws.send_json(dict(type='error', message=str(e)))
            finally:
                sender.cancel()
                with suppress(asyncio.CancelledError, ConnectionResetError):
                    await sender
                with self.lock:
                    self.status['connected_clients'] -= 1
            return ws

        app.router.add_get('/', static_file)
        app.router.add_get('/{name:app.js|style.css}', static_file)
        app.router.add_get('/model.bin', model_data)
        app.router.add_get('/api/status', status)
        app.router.add_post('/api/control', control)
        app.router.add_get('/ws', socket)
        runner = web.AppRunner(app, access_log=None)
        try:
            await runner.setup()
            await web.TCPSite(runner, '127.0.0.1', self.port).start()
            self.loop = asyncio.get_running_loop()
            self.finished = asyncio.Event()
            self.ready.set()
            await self.finished.wait()
        except Exception as e:
            self.error = e
            self.ready.set()
        finally:
            await runner.cleanup()

    def start(self):
        self.thread = threading.Thread(target=lambda: asyncio.run(self._serve()), daemon=True)
        self.thread.start()
        if not self.ready.wait(15):
            raise TimeoutError('live server did not start')
        if self.error:
            raise self.error

    def close(self):
        if hasattr(self, 'loop') and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.finished.set)
        self.thread.join(timeout=5)
