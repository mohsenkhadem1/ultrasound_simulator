"""Interactive Isaac GPU physics -> material Gaussians -> browser loop."""
import json
import time
import numpy as np
from .live_server import LiveServer
from .probe_motion import scan_terrain, scan_pose


def run_live(args, app, world, bodies, tissue_views, probe_view, tool_view, indices, gpu_id):
    import warp as wp
    model = dict(np.load(args.live_gaussians))
    from isaacsim.core.simulation_manager import SimulationManager
    physics_view = SimulationManager.get_physics_simulation_view()
    native_views = [physics_view.create_volume_deformable_body_view(f'/World/Tissue_{i}') for i in range(len(bodies))]

    def read_positions():
        # Read only simulation nodes. The convenience getter also fetches
        # collision/rest arrays and launches indexing copies for all three.
        return [view.get_simulation_nodal_positions().numpy()[0] for view in native_views]
    # A changed prior must be rebound, rather than silently deforming the wrong anatomy.
    rest_nodes = np.concatenate([body.points for body in bodies])
    soft_count = int(model['soft_count'])
    binding_vertices = rest_nodes[model['nodes'][:soft_count]]
    rest_centers = np.einsum('ni,nij->nj', model['weights'][:soft_count], binding_vertices)
    if not np.allclose(rest_centers, model['means'][:soft_count], atol=2e-6, rtol=0):
        raise ValueError('Gaussian prior does not match this anatomy; rerun prepare_gaussians.py')
    terrain = scan_terrain(np.concatenate([b.points for b in bodies]))
    server = LiveServer(model, args.live_port, args.live_autoplay)
    server.start()
    url = f'http://127.0.0.1:{args.live_port}'
    (args.output/'live_url.txt').write_text(url+'\n')
    print('LIVE GAUSSIAN VIEWER: '+url, flush=True)
    device = f'cuda:{gpu_id}'
    from .gaussians_warp import WarpGaussianTransport
    transport = WarpGaussianTransport(model, device, bodies)
    current_pose = probe_view.get_transforms().numpy()[0].astype(float)
    transport.pack([body.points for body in bodies], current_pose, [])  # compile before timing
    yaw = 0.
    simulation_time = 0.
    total_simulation_time = 0.
    sequence = 0
    records = []
    start = previous = time.monotonic()
    next_tick = start
    dt = 1/args.live_physics_hz
    steps = args.live_physics_hz//args.live_fps
    parked = tool_view.get_transforms().numpy()[0].astype(float)
    try:
        while app.is_running():
            loop_start = time.monotonic()
            controls = server.get_controls()
            if controls['stop'] or (args.live_seconds and loop_start-start >= args.live_seconds):
                break
            if controls['reset']:
                for body, view in zip(bodies, tissue_views):
                    view.set_nodal_positions(body.points.astype(np.float32)[None])
                    view.set_nodal_velocities(np.zeros((1, len(body.points), 3), np.float32))
                simulation_time = 0.
                current_pose = np.array([0, 0, float(np.interp(0., *terrain))+.012, 0, 0, 0, 1.])
                yaw = 0.
                server.command(dict(auto=False, x_mm=0., y_mm=0., press_mm=0., yaw_deg=0., paused=False))
                controls = server.get_controls()
            physics_start = time.monotonic()
            for _ in range(steps if not controls['paused'] else 0):
                if controls['auto']:
                    clock = simulation_time % 24
                    wanted, wanted_yaw, stroke, _, _ = scan_pose(clock if clock <= 12 else 24-clock, 12, terrain)
                    stroke *= .6  # gentle automatic contact: at most 3 mm travel
                else:
                    stroke = controls['press_mm']*.001
                    wanted = np.array([controls['x_mm']*.001, controls['y_mm']*.001,
                                       float(np.interp(controls['y_mm']*.001, *terrain))+.004-stroke])
                    wanted_yaw = np.deg2rad(controls['yaw_deg'])
                # Substep rate limits keep dragging/slider jumps from teleporting contact.
                difference = wanted[:2]-current_pose[:2]
                length = np.linalg.norm(difference)
                current_pose[:2] += difference*min(1., .025*dt/max(length, 1e-12))
                # Follow terrain at the ACTUAL rate-limited position. Using the
                # desired position can drive the head below a slope on return.
                wanted_z = float(np.interp(current_pose[1], *terrain))+.004-stroke
                current_pose[2] += np.clip(wanted_z-current_pose[2], -.008*dt, .035*dt)
                yaw += np.clip(wanted_yaw-yaw, -np.deg2rad(45)*dt, np.deg2rad(45)*dt)
                current_pose[3:] = [0, 0, np.sin(yaw/2), np.cos(yaw/2)]
                probe_view.set_transforms(wp.array(current_pose[None], dtype=wp.float32, device=device), indices)
                tool_view.set_transforms(wp.array(parked[None], dtype=wp.float32, device=device), indices)
                # Authoritative tensor writes require no viewport or USD update.
                world.step(render=False)
                simulation_time += dt
                total_simulation_time += dt
            physics_ms = (time.monotonic()-physics_start)*1000
            read_start = time.monotonic()
            positions = read_positions()
            measured_pose = probe_view.get_transforms().numpy()[0]
            read_ms = (time.monotonic()-read_start)*1000
            actual_stroke = float(np.interp(measured_pose[1], *terrain))+.004-float(measured_pose[2])
            now = time.monotonic()
            interval = now-previous
            previous = now
            header = [1, simulation_time, sequence, len(model['means']), *measured_pose[:3], yaw,
                      actual_stroke, 1/max(interval, 1e-6), 0, 0, 0, float(controls['paused'])]
            gaussian_start = time.monotonic()
            packet, info = transport.pack(positions, measured_pose, header)
            gaussian_ms = (time.monotonic()-gaussian_start)*1000
            info.update(simulation_time_s=simulation_time, frame=sequence, physics_ms=physics_ms,
                        nodal_read_ms=read_ms,
                        gaussian_update_ms=gaussian_ms, update_hz=1/max(interval, 1e-6),
                        total_simulated_seconds=total_simulation_time,
                        simulation_wall_ratio=total_simulation_time/max(now-start, 1e-6),
                        probe_position_m=measured_pose[:3].tolist(), actual_press_travel_mm=actual_stroke*1000,
                        mode='Isaac FEM-driven material Gaussians', constitutive_diagnostic='Neo-Hookean stress proxy, not solver stress',
                        learned_reconstruction=False)
            if info['min_jacobian'] <= 0 or not np.isfinite(info['max_displacement_mm']):
                server.command(dict(paused=True, auto=False))
                info['error'] = 'Invalid tissue state: paused. Reset the scene.'
            server.publish(packet, info)
            if sequence % max(1, round(args.live_fps)) == 0:
                records.append(info)
                (args.output/'live_metrics.json').write_text(json.dumps(dict(latest=info, samples=records), indent=2)+'\n')
                print(f'LIVE t={simulation_time:.2f}s, {info["update_hz"]:.1f} updates/s, motion={info["max_displacement_mm"]:.2f}mm, Jmin={info["min_jacobian"]:.3f}', flush=True)
            sequence += 1
            # Absolute deadlines prevent sleep/scheduling overhead from
            # accumulating into a slower simulation clock each frame.
            next_tick += 1/args.live_fps
            now = time.monotonic()
            next_tick = max(next_tick, now-.1)
            time.sleep(max(0, next_tick-now))
    finally:
        np.savez_compressed(args.output/'live_final_state.npz', probe_pose=current_pose,
            **{f'positions_{i}': points for i, points in enumerate(read_positions())})
        server.close()
        (args.output/'live_complete.json').write_text(json.dumps(dict(frames=sequence, simulation_seconds=simulation_time,
            total_simulated_seconds=total_simulation_time,
            wall_seconds=time.monotonic()-start))+'\n')
