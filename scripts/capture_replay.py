"""Capture matched 3D states in Isaac's renderer, without rerunning mechanics."""
import argparse
import asyncio
import json
from pathlib import Path
import time
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('episode', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--frames', type=int, default=0)
    args = parser.parse_args()
    output = args.output or args.episode/'scene_frames'
    output.mkdir(exist_ok=False)
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'active_gpu': 0, 'physics_gpu': 0,
                         'multi_gpu': False, 'width': 960, 'height': 720,
                         'open_usd': str((args.episode/'scene.usda').resolve())})
    exit_code = 0
    try:
        import carb
        import numpy as np
        import omni.usd
        import omni.timeline
        from pxr import Gf, UsdGeom, Vt
        from isaacsim.core.utils.viewports import set_camera_view
        from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
        from deform_sim.phantom import surface_positions
        omni.timeline.get_timeline_interface().stop()
        stage = omni.usd.get_context().get_stage()
        carb.settings.get_settings().set_int('/persistent/app/viewport/displayOptions', 0)
        reference = np.load(args.episode/'reference.npz')
        metadata = json.loads((args.episode/'metadata.json').read_text())
        count = int(reference['body_count']) if 'body_count' in reference else 2
        anatomy = bool(metadata.get('anatomy'))
        visuals = [UsdGeom.Mesh.Get(stage, f'/World/Display_{i}') for i in range(count)]
        # Show the inclusion by removing the front half of the DISPLAY surface.
        # The saved physical states come from the intact, bonded volume model.
        faces = np.asarray(visuals[0].GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        rest_skin = reference['display_points_0'] if 'display_points_0' in reference else reference['points_0']
        keep = rest_skin[faces].mean(axis=1)[:, 1] >= 0
        visuals[0].GetFaceVertexCountsAttr().Set([3]*int(keep.sum()))
        visuals[0].GetFaceVertexIndicesAttr().Set(faces[keep].reshape(-1).tolist())
        visuals[0].GetDisplayColorAttr().Set([Gf.Vec3f(.65, .25, .30)])
        visuals[1].GetDisplayColorAttr().Set([Gf.Vec3f(.55, .18, .60) if anatomy else Gf.Vec3f(.10, .40, .90)])
        tool = UsdGeom.Cube.Get(stage, '/World/CompressionTool')
        tool.GetDisplayColorAttr().Set([Gf.Vec3f(.06, .60, .40)])
        translate = next(op for op in tool.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate)
        probe_path = metadata.get('probe_prim_path', '/World/FixedUltrasoundProbe')
        probe = UsdGeom.Cube.Get(stage, probe_path)
        probe.GetDisplayColorAttr().Set([Gf.Vec3f(.90, .48, .06)])
        if metadata.get('probe_motion') in ('press', 'sweep', 'scan'):
            from deform_sim.scene import add_probe_handle
            add_probe_handle(stage, probe_path)
        probe_translate = next(op for op in probe.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate)
        probe_orient = next((op for op in probe.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeOrient), None)
        plane = None
        if metadata.get('imaging_plane_overlay'):
            from deform_sim.scene import imaging_plane
            plane = imaging_plane(stage)
        if anatomy:
            center = rest_skin.mean(axis=0)
            set_camera_view(eye=center+np.array([.18, -.27, .18]), target=center)
        else:
            set_camera_view(eye=[.062, -.095, .058], target=[0, 0, .022])
        viewport = get_active_viewport()
        viewport.resolution = (960, 720)
        camera = UsdGeom.Camera.Get(stage, viewport.camera_path)
        camera.GetFocalLengthAttr().Set(30.)
        camera.GetClippingRangeAttr().Set(Gf.Vec2f(.001, 10.))
        for _ in range(20):
            app.update()
        states = sorted(p for p in (args.episode/'state').glob('*.npz') if p.stem.isdecimal())
        if args.frames:
            states = states[:args.frames]
        for index, path in enumerate(states):
            state = np.load(path)
            positions = [state[f'positions_{i}'] for i in range(count)]
            for i, visual in enumerate(visuals):
                visual.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(surface_positions(reference, positions, i).astype(np.float32)))
            pose = state['tool_pose'][:3] if 'tool_pose' in state else [0, 0, .044-float(state['command'])]
            translate.Set(Gf.Vec3d(*map(float, pose)))
            if 'probe_body_pose' in state:
                probe_translate.Set(Gf.Vec3d(*map(float, state['probe_body_pose'][:3])))
                if probe_orient:
                    q = state['probe_body_pose'][3:]
                    probe_orient.Set(Gf.Quatf(float(q[3]), Gf.Vec3f(*map(float, q[:3]))))
            if plane:
                local = np.array([[-.02, 0, .002], [.02, 0, .002], [.02, 0, .05], [-.02, 0, .05]])
                corners = local@state['probe_rotation'].T+state['probe_position']
                plane[0].GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(corners.astype(np.float32)))
                plane[1].GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(corners[[0, 1, 2, 3, 0]].astype(np.float32)))
            # Let Hydra consume the new points before scheduling the capture.
            for _ in range(6):
                app.update()
            capture = capture_viewport_to_file(viewport, str((output/f'{path.stem}.png').resolve()))
            future = asyncio.ensure_future(capture.wait_for_result())
            started = time.monotonic()
            while not future.done():
                app.update()
                if time.monotonic()-started > 60:
                    raise TimeoutError('Isaac viewport capture timed out')
            future.result()
            destination = output/f'{path.stem}.png'
            while not destination.exists():
                app.update()
                time.sleep(.002)
                if time.monotonic()-started > 60:
                    raise RuntimeError('renderer did not write captured frame')
            print(f'{index+1}/{len(states)} Isaac frames captured', flush=True)
    except Exception:
        import traceback
        traceback.print_exc()
        exit_code = 1
        raise
    finally:
        app.close(exit_code=exit_code)


if __name__ == '__main__':
    main()
