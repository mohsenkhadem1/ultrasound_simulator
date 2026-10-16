"""Isaac Sim 6.2 / PhysX GPU phantom with acoustic scatterer export.

Run with .venv/bin/python scripts/run_isaac.py --headless --output outputs/isaac
Then render exported snapshots in the separate ultrasound environment.
"""
import argparse
import json
from pathlib import Path
import sys
import os
import subprocess
import time
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('outputs/isaac'))
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--seconds', type=float, default=2.0)
    parser.add_argument('--fps', type=int, default=20)
    parser.add_argument('--mesh-resolution', type=int, default=9)
    parser.add_argument('--scatterers', type=int, default=120000)
    parser.add_argument('--tumor-backscatter', type=float, help='tumour echo-amplitude multiplier relative to brain; e.g. 6 for a conspicuous demonstration')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--anatomy', type=Path, help='prepared segmented anatomy NPZ (scripts/prepare_brain.py)')
    parser.add_argument('--gpu', type=int, help='GPU index; GUI defaults to the GPU driving the display')
    parser.add_argument('--probe-motion', choices=['fixed', 'press', 'sweep', 'scan'], default='fixed', help='scan: cross the tumour, pause and press gently; use --seconds 6 with brain anatomy')
    parser.add_argument('--ultrasound-preview', action='store_true', help='show acoustically rendered B-mode frames in an Isaac window')
    parser.add_argument('--ultrasound-quality', choices=['preview', 'standard'], default='preview')
    parser.add_argument('--exit-after-preview', action='store_true', help='close Isaac after all acoustic preview frames finish')
    parser.add_argument('--capture-preview', action='store_true', help='save the Isaac window directly from the renderer')
    args = parser.parse_args()
    if args.headless and args.ultrasound_preview:
        parser.error('--ultrasound-preview requires a visible Isaac window')
    if args.output.exists():
        parser.error('output exists; choose a new directory')
    if args.seconds <= 0 or args.fps <= 0 or args.scatterers <= 0:
        parser.error('seconds, fps and scatterers must be positive')
    if args.tumor_backscatter is not None and not 0 < args.tumor_backscatter < float('inf'):
        parser.error('--tumor-backscatter must be finite and positive')
    if args.anatomy and args.probe_motion == 'fixed':
        parser.error('segmented brain anatomy requires --probe-motion press or sweep')
    if args.probe_motion == 'scan' and not args.anatomy:
        parser.error('--probe-motion scan requires --anatomy')

    worker = None
    worker_log = None
    exit_code = 0
    gpu_id = args.gpu if args.gpu is not None else 0
    if args.gpu is None and not args.headless:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,pci.bus_id,display_active', '--format=csv,noheader'], capture_output=True, text=True)
        gpu_rows = []
        for line in result.stdout.splitlines():
            index, bus, active = [part.strip() for part in line.split(',')]
            gpu_rows.append((int(index), bus.lower()))
            if active.strip().lower() == 'enabled':
                gpu_id = int(index)
                break
        else:
            # Some drivers report display_active=Disabled even on Xorg's primary
            # GPU. Match its PCI address rather than assuming CUDA GPU 0 presents.
            logs = sorted((Path.home()/'.local/share/xorg').glob('Xorg.*.log'), key=lambda p: p.stat().st_mtime, reverse=True)
            for log in logs[:1]:
                match = re.search(r'NVIDIA\(0\): NVIDIA GPU .*? at PCI:(\d+):(\d+):(\d+)', log.read_text(errors='replace'))
                if match:
                    bus, device, function = map(int, match.groups())
                    suffix = f'{bus:02x}:{device:02x}.{function}'
                    gpu_id = next((index for index, pci in gpu_rows if pci.endswith(suffix)), 0)
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': args.headless, 'width': 960, 'height': 720,
                         'active_gpu': gpu_id, 'physics_gpu': gpu_id, 'multi_gpu': False,
                         'disable_viewport_updates': args.headless,
                         'fast_shutdown': True})
    try:
        import carb
        import numpy as np
        from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt
        from isaacsim.core.api import World
        from isaacsim.core.experimental.prims import DeformablePrim
        from isaacsim.core.utils.viewports import set_camera_view
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
        from deform_sim.phantom import make_phantom, sample_scatterers, advect, load_bodies, surface_positions

        settings = carb.settings.get_settings()
        settings.set_bool('/persistent/physics/enableDeformableBeta', True)
        settings.set_bool('/physics/updateToUsd', True)
        settings.set_bool('/physics/updateVelocitiesToUsd', True)
        world = World(stage_units_in_meters=1.0, physics_dt=1/240, rendering_dt=1/240,
                      backend='warp', device=f'cuda:{gpu_id}')
        stage = world.stage
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        scene = UsdPhysics.Scene.Get(stage, '/physicsScene')
        if not scene:
            raise RuntimeError('World did not create /physicsScene')
        scene.CreateGravityMagnitudeAttr(0.0)
        gpu = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
        gpu.CreateEnableGPUDynamicsAttr(True)
        gpu.CreateBroadphaseTypeAttr('GPU')
        anatomy = dict(np.load(args.anatomy)) if args.anatomy else {}
        if anatomy:
            bodies = load_bodies(anatomy)
        else:
            bodies, _ = make_phantom(args.mesh_resolution)
        count = len(bodies)
        moduli = np.asarray(anatomy.get('youngs_modulus', [15000., 60000.]))
        amplitudes = np.asarray(anatomy.get('scatter_amplitudes', [1.]+[.18]*(count-1)), dtype=float).copy()
        if args.tumor_backscatter is not None:
            amplitudes[1] = args.tumor_backscatter
        probe_height = float(anatomy.get('probe_height', .044))
        parked_height = max(.075, probe_height+.06)
        minimum_z = min(b.points[:, 2].min() for b in bodies)
        scan_motion = args.probe_motion == 'scan'
        if scan_motion:
            from deform_sim.probe_motion import scan_terrain, scan_pose
            terrain = scan_terrain(np.concatenate([b.points for b in bodies]))
            initial_position = scan_pose(0, args.seconds, terrain)[0]
            anatomy['scatter_bounds'] = np.array([[-.035, -.044, .075], [.035, .045, .155]])
            anatomy['acoustic_elevation_limit'] = .008

        def material(path, youngs):
            mat = UsdShade.Material.Define(stage, path)
            prim = mat.GetPrim()
            for api in ('OmniPhysicsBaseMaterialAPI', 'OmniPhysicsDeformableMaterialAPI', 'PhysxDeformableMaterialAPI'):
                if not prim.ApplyAPI(api):
                    raise RuntimeError(f'Isaac build lacks {api}')
            for key, value in {'omniphysics:density': 1000., 'omniphysics:dynamicFriction': 0.3,
                               'omniphysics:youngsModulus': youngs, 'omniphysics:poissonsRatio': 0.45,
                               'physxDeformableMaterial:elasticityDamping': 0.01}.items():
                prim.GetAttribute(key).Set(value)
            return mat

        meshes, visuals, paths = [], [], []
        for i, body in enumerate(bodies):
            path = f'/World/Tissue_{i}'
            paths.append(path)
            mesh = UsdGeom.TetMesh.Define(stage, path)
            mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(body.points.astype(np.float32)))
            mesh.CreateVelocitiesAttr(Vt.Vec3fArray.FromNumpy(np.zeros_like(body.points, dtype=np.float32)))
            mesh.GetTetVertexIndicesAttr().Set(Vt.Vec4iArray.FromNumpy(body.tetrahedra))
            faces = UsdGeom.TetMesh.ComputeSurfaceFaces(mesh)
            mesh.GetSurfaceFaceVertexIndicesAttr().Set(faces)
            prim = mesh.GetPrim()
            for api in ('OmniPhysicsDeformableBodyAPI', 'OmniPhysicsVolumeDeformableSimAPI', 'PhysxBaseDeformableBodyAPI'):
                if not prim.ApplyAPI(api):
                    raise RuntimeError(f'Isaac build lacks {api}')
            prim.GetAttribute('omniphysics:restShapePoints').Set(mesh.GetPointsAttr().Get())
            prim.GetAttribute('omniphysics:restTetVtxIndices').Set(mesh.GetTetVertexIndicesAttr().Get())
            prim.GetAttribute('physxDeformableBody:disableGravity').Set(True)
            UsdPhysics.CollisionAPI.Apply(prim)
            contact = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            contact.CreateContactOffsetAttr(0.0002)
            contact.CreateRestOffsetAttr(0.0)
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material(f'/World/Material_{i}', float(moduli[i])), materialPurpose='physics')
            mesh.CreatePurposeAttr('guide')
            # Explicit surface display geometry; acoustic rendering uses volume state.
            surface = UsdGeom.Mesh.Define(stage, f'/World/Display_{i}')
            display_faces = anatomy.get(f'display_faces_{i}', np.asarray(faces))
            surface.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(anatomy.get(f'display_points_{i}', body.points), dtype=np.float32)))
            surface.CreateFaceVertexCountsAttr([3]*len(display_faces))
            surface.CreateFaceVertexIndicesAttr(display_faces.reshape(-1).tolist())
            color = [(0.72, .42, .43), (.50, .22, .62), (.17, .50, .80)][min(i, 2)] if anatomy else [(0.7, 0.35, 0.35), (0.2, 0.6, 0.9)][i]
            surface.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            surface.CreateDoubleSidedAttr(True)
            surface.CreateSubdivisionSchemeAttr('none')
            if anatomy and f'display_points_{i}' not in anatomy:
                surface.CreateVisibilityAttr('invisible')
            meshes.append(mesh)
            visuals.append(surface)

        # Piecewise material regions share exact rest nodes and are fully bonded.
        # Omni PhysX does not implement per-tetrahedron multi-material subsets.
        for i in range(count):
            UsdPhysics.FilteredPairsAPI.Apply(meshes[i].GetPrim()).CreateFilteredPairsRel().SetTargets([paths[j] for j in range(count) if j != i])
            for j in range(i+1, count):
                _, a, b = np.intersect1d(bodies[i].global_ids, bodies[j].global_ids, return_indices=True)
                if not len(a):
                    continue
                bond = stage.DefinePrim(f'/World/Bond_{i}_{j}', 'OmniPhysicsVtxVtxAttachment')
                for src, region, ids in ((0, i, a), (1, j, b)):
                    bond.GetRelationship(f'omniphysics:src{src}').SetTargets([paths[region]])
                    bond.GetAttribute(f'omniphysics:vtxIndicesSrc{src}').Set(ids.tolist())
        anchor = UsdGeom.Xform.Define(stage, '/World/BaseAnchor')
        UsdPhysics.RigidBodyAPI.Apply(anchor.GetPrim()).CreateKinematicEnabledAttr(True)
        for i, body in enumerate(bodies):
            base_ids = np.flatnonzero(np.isclose(body.points[:, 2], minimum_z))
            if not len(base_ids):
                continue
            pin = stage.DefinePrim(f'/World/BaseBond_{i}', 'OmniPhysicsVtxXformAttachment')
            pin.GetRelationship('omniphysics:src0').SetTargets([paths[i]])
            pin.GetRelationship('omniphysics:src1').SetTargets([anchor.GetPath()])
            pin.GetAttribute('omniphysics:vtxIndicesSrc0').Set(base_ids.tolist())
            pin.GetAttribute('omniphysics:localPositionsSrc1').Set(Vt.Vec3fArray.FromNumpy(body.points[base_ids].astype(np.float32)))

        tool = UsdGeom.Cube.Define(stage, '/World/CompressionTool')
        tool.CreateSizeAttr(1.0)
        tool.AddScaleOp().Set(Gf.Vec3f(0.018, 0.018, 0.006))
        tool_translate = tool.AddTranslateOp()
        # Put translation outside scale in the xform stack.
        tool.SetXformOpOrder([tool_translate, tool.GetOrderedXformOps()[0]])
        tool_translate.Set(Gf.Vec3d(0, 0, 0.044))
        UsdPhysics.CollisionAPI.Apply(tool.GetPrim())
        rigid = UsdPhysics.RigidBodyAPI.Apply(tool.GetPrim())
        rigid.CreateKinematicEnabledAttr(True)
        PhysxSchema.PhysxCollisionAPI.Apply(tool.GetPrim()).CreateContactOffsetAttr(0.0002)
        tool.CreateDisplayColorAttr([Gf.Vec3f(0.3, 0.3, 0.35)])
        pressing_probe = args.probe_motion in ('press', 'sweep', 'scan')
        if pressing_probe:
            tool_translate.Set(Gf.Vec3d(0, 0, parked_height))
            tool.CreateVisibilityAttr('invisible')
        probe_path = '/World/UltrasoundProbe' if pressing_probe else '/World/FixedUltrasoundProbe'
        probe = UsdGeom.Cube.Define(stage, probe_path)
        probe.CreateSizeAttr(1.)
        probe_translate = probe.AddTranslateOp()
        probe_translate.Set(Gf.Vec3d(0, 0, probe_height) if pressing_probe else Gf.Vec3d(0, -.017, .02))
        if scan_motion:
            probe_translate.Set(Gf.Vec3d(*map(float, initial_position)))
        probe_orient = probe.AddOrientOp()
        probe.AddScaleOp().Set(Gf.Vec3f(.04, .008, .006) if pressing_probe else Gf.Vec3f(.04, .004, .008))
        probe.CreateDisplayColorAttr([Gf.Vec3f(0.15, 0.15, 0.18)])
        if pressing_probe:
            UsdPhysics.CollisionAPI.Apply(probe.GetPrim())
            UsdPhysics.RigidBodyAPI.Apply(probe.GetPrim()).CreateKinematicEnabledAttr(True)
            PhysxSchema.PhysxCollisionAPI.Apply(probe.GetPrim()).CreateContactOffsetAttr(.0002)
            from deform_sim.scene import add_probe_handle
            add_probe_handle(stage, probe_path)
        from pxr import UsdLux
        UsdLux.DomeLight.Define(stage, '/World/Light').CreateIntensityAttr(1000)
        camera_target = np.concatenate([b.points for b in bodies]).mean(axis=0)
        camera_eye = camera_target+np.array([.20, -.25, .20]) if anatomy else [.10, -.10, .09]
        set_camera_view(eye=camera_eye, target=camera_target if anatomy else [0, 0, .02])

        args.output.mkdir(parents=True)
        (args.output/'state').mkdir()
        scatter = sample_scatterers(bodies, args.scatterers, args.seed, amplitudes, anatomy.get('scatter_bounds'))
        payload = dict(anatomy)
        payload['scatter_amplitudes'] = amplitudes
        payload.update({f'{key}_{i}': value for i, b in enumerate(bodies) for key, value in [('points', b.points), ('tets', b.tetrahedra), ('ids', b.global_ids)]})
        payload['body_count'] = count
        np.savez_compressed(args.output/'reference.npz', **payload, **scatter)
        preview_image = None
        preview_provider = None
        if args.ultrasound_preview:
            import omni.ui as ui
            window = ui.Window('Acoustic ultrasound — SIMUS', width=480, height=560, raster_policy=ui.RasterPolicy.NEVER)
            preview_provider = ui.ByteImageProvider()
            preview_provider.set_bytes_data([0, 0, 0, 255], [1, 1], strict=True)
            with window.frame:
                with ui.VStack():
                    ui.Label('RF simulation / DAS / B-mode — updates asynchronously', height=32)
                    preview_image = ui.ImageWithProvider(preview_provider, width=440, height=440, fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT)
            root = Path(__file__).resolve().parents[1]
            environment = os.environ.copy()
            environment['PYTHONPATH'] = str(root/'src')
            environment['OPENBLAS_NUM_THREADS'] = '1'
            worker_log = (args.output/'acoustics.log').open('w')
            worker_args = [str(root/'.venv-ultrasound/bin/python'), '-u', '-m', 'deform_sim.ultrasound', str(args.output), '--watch']
            if args.ultrasound_quality == 'preview':
                worker_args.append('--preview')
            worker = subprocess.Popen(worker_args, env=environment, stdout=worker_log, stderr=subprocess.STDOUT)
        metadata = dict(engine='Isaac Sim 6.2.0.0 / PhysX GPU', units='m, s, Pa, kg/m^3',
                        youngs_modulus=moduli.tolist(), poissons_ratio=0.45, body_count=count,
                        scatter_amplitudes=amplitudes.tolist(),
                        anatomy=str(args.anatomy) if args.anatomy else None,
                        duration_seconds=args.seconds, imaging_plane_overlay=scan_motion,
                        inclusion='conforming tetrahedral regions with bonded shared vertices',
                        probe_motion=args.probe_motion, probe_prim_path=probe_path,
                        probe=('moving top head; depth -z, lateral +x, elevation -y; aperture 3 mm behind contact face' if pressing_probe else 'fixed at world (0,-0.015,0.02); depth +y, lateral +x, elevation -z'),
                        dt=1/240, observation_fps=args.fps, seed=args.seed,
                        limitations='elasticity damping is not calibrated tissue viscoelasticity')
        (args.output/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
        if args.anatomy and args.anatomy.with_name('provenance.json').exists():
            (args.output/'anatomy_source.json').write_text(args.anatomy.with_name('provenance.json').read_text())
        stage.GetRootLayer().Export(str(args.output/'scene.usda'))
        world.reset()
        from isaacsim.core.simulation_manager import SimulationManager
        import warp as wp
        physics_view = SimulationManager.get_physics_simulation_view()
        tool_view = physics_view.create_rigid_body_view('/World/CompressionTool')
        probe_view = physics_view.create_rigid_body_view(probe_path) if pressing_probe else None
        tool_indices = wp.array([0], dtype=wp.int32, device=f'cuda:{gpu_id}')
        # Read authoritative GPU simulation nodes, avoiding stale USD/Fabric data.
        tissue_views = [DeformablePrim(path, deformable_type='volume') for path in paths]
        frame = 0
        max_displacement = 0.
        marker_mask = (scatter['body'] == 1) | (np.abs(advect(bodies, [b.points for b in bodies], scatter)[:, 2]-0.02) > 0.007)
        marker_ids = np.flatnonzero(marker_mask)[::max(1, args.scatterers//32)][:32]
        steps_per_frame = max(1, round(240/args.fps))
        for step in range(round(args.seconds*240)+1):
            t = step/240
            depth = 0.004*min(max(t/0.5, 0), 1) if t < 1.2 else 0.004*max(0, 1-(t-1.2)/0.4)
            contact_height, phase = 0., ''
            if scan_motion:
                scan_position, scan_yaw, depth, contact_height, phase = scan_pose(t, args.seconds, terrain)
            tool_z = parked_height if pressing_probe else .044-depth
            tool_translate.Set(Gf.Vec3d(0, 0, tool_z))
            # GPU PhysX supports rigid transforms, but not kinematic-target tensor
            # writes. Apply the small commanded pose increment before each solve.
            target = wp.array([[0., 0., tool_z, 0., 0., 0., 1.]], dtype=wp.float32, device=f'cuda:{gpu_id}')
            tool_view.set_transforms(target, tool_indices)
            if pressing_probe:
                sweep_phase = np.clip((t-.45)/.75, 0, 1)
                probe_x = .006*np.sin(2*np.pi*sweep_phase) if args.probe_motion == 'sweep' else 0.
                probe_y = .003*(1-np.cos(2*np.pi*sweep_phase)) if args.probe_motion == 'sweep' else 0.
                yaw = np.deg2rad(15)*np.sin(2*np.pi*sweep_phase) if args.probe_motion == 'sweep' else 0.
                probe_z = probe_height-depth
                if scan_motion:
                    probe_x, probe_y, probe_z = map(float, scan_position)
                    yaw = scan_yaw
                probe_translate.Set(Gf.Vec3d(probe_x, probe_y, probe_z))
                probe_orient.Set(Gf.Quatf(float(np.cos(yaw/2)), Gf.Vec3f(0, 0, float(np.sin(yaw/2)))))
                probe_target = wp.array([[probe_x, probe_y, probe_z, 0., 0., np.sin(yaw/2), np.cos(yaw/2)]], dtype=wp.float32, device=f'cuda:{gpu_id}')
                probe_view.set_transforms(probe_target, tool_indices)
            if step:
                # One Kit update per physics tick pulls USD kinematic targets.
                # Both clocks use 1/240 s; viewport rendering is disabled headlessly.
                world.step(render=True)
            if step % steps_per_frame:
                continue
            positions = [view.get_nodal_positions()[0].numpy()[0].astype(np.float64) for view in tissue_views]
            velocities = [view.get_nodal_velocities().numpy()[0].astype(np.float64) for view in tissue_views]
            if not all(p.shape == b.points.shape and np.all(np.isfinite(p)) for p, b in zip(positions, bodies)):
                raise RuntimeError('invalid PhysX vertex state')
            for i in range(count):
                visuals[i].GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(surface_positions(payload, positions, i).astype(np.float32)))
                max_displacement = max(max_displacement, float(np.max(np.linalg.norm(positions[i]-bodies[i].points, axis=1))))
            temporary = args.output/'state'/f'{frame:05d}.partial.npz'
            probe_pose = probe_view.get_transforms().numpy()[0] if pressing_probe else np.array([0, -.017, .02, 0, 0, 0, 1.])
            acoustic_position = probe_pose[:3] if pressing_probe else np.array([0, -.015, .02])
            if pressing_probe:
                # Use the measured rigid quaternion, including the sweep heading.
                qx, qy, qz, qw = probe_pose[3:]
                rigid_rotation = np.array([[1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
                                           [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
                                           [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)]])
                acoustic_rotation = rigid_rotation@np.diag([1., -1., -1.])
            else:
                acoustic_rotation = np.array([[1., 0, 0], [0, 0, 1.], [0, -1., 0]])
            np.savez_compressed(temporary, time=t, command=depth,
                                phase=phase, probe_contact_height_m=contact_height,
                                **{f'positions_{i}': p for i, p in enumerate(positions)},
                                **{f'velocities_{i}': v for i, v in enumerate(velocities)},
                                tool_pose=tool_view.get_transforms().numpy()[0],
                                probe_body_pose=probe_pose, probe_position=acoustic_position, probe_rotation=acoustic_rotation,
                                marker_ids=marker_ids, markers=advect(bodies, positions, scatter)[marker_ids])
            temporary.rename(args.output/'state'/f'{frame:05d}.npz')
            print(f'frame {frame}: t={t:.3f}s, max displacement={max_displacement*1000:.3f} mm', flush=True)
            frame += 1
        if args.seconds >= 0.6 and max_displacement < 0.0001:
            raise RuntimeError('No deformation recorded: check GPU PhysX and USD writeback')
        (args.output/'complete.json').write_text(json.dumps(dict(frames=frame, max_displacement_m=max_displacement))+'\n')
        print(f'Isaac episode complete: {args.output}', flush=True)
        if preview_image is not None:
            world.pause()
            last_image = None
            last_image_mtime = None
            while app.is_running():
                images = sorted((args.output/'ultrasound').glob('*.png'))
                if images and (images[-1] != last_image or images[-1].stat().st_mtime_ns != last_image_mtime):
                    last_image = images[-1]
                    last_image_mtime = last_image.stat().st_mtime_ns
                    from PIL import Image
                    rgba = np.asarray(Image.open(last_image).convert('RGBA'), dtype=np.uint8)
                    preview_provider.set_bytes_data(rgba.reshape(-1).tolist(), [rgba.shape[1], rgba.shape[0]], strict=True)
                    window.frame.invalidate_raster()
                    # Replay matching mechanics so a delayed acoustic frame is
                    # displayed beside its own tissue state, not the final state.
                    state = np.load(args.output/'state'/f'{last_image.stem}.npz')
                    replay_positions = [state[f'positions_{i}'] for i in range(count)]
                    for i in range(count):
                        visuals[i].GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(surface_positions(payload, replay_positions, i).astype(np.float32)))
                    tool_translate.Set(Gf.Vec3d(*map(float, state['tool_pose'][:3])))
                    if 'probe_body_pose' in state:
                        probe_translate.Set(Gf.Vec3d(*map(float, state['probe_body_pose'][:3])))
                        q = state['probe_body_pose'][3:]
                        probe_orient.Set(Gf.Quatf(float(q[3]), Gf.Vec3f(*map(float, q[:3]))))
                    if args.capture_preview:
                        from omni.kit.renderer.capture import acquire_renderer_capture_interface
                        # Let UI resolve the image before capturing the swapchain.
                        for _ in range(30):
                            app.update()
                        acquire_renderer_capture_interface().capture_next_frame_swapchain(str((args.output/'isaac_window.png').resolve()))
                if worker.poll() not in (None, 0):
                    raise RuntimeError(f'Acoustic worker failed; see {args.output}/acoustics.log')
                app.update()
                if args.exit_after_preview and worker.poll() == 0:
                    break
                time.sleep(.01)
    except Exception:
        exit_code = 1
        import traceback
        traceback.print_exc()
        raise
    finally:
        if worker is not None and worker.poll() is None:
            worker.terminate()
            worker.wait()
        if worker_log is not None:
            worker_log.close()
        app.close(exit_code=exit_code)


if __name__ == '__main__':
    main()
