import unittest
import numpy as np
from scipy.ndimage import gaussian_filter, shift
from deform_sim.reconstruction import Observation, TrackedVolume, estimate_inplane_motion, advance_material_tracks
from deform_sim.fast_ultrasound import FastUltrasound
from deform_sim.inverse_elasticity import ElasticInverse
from deform_sim.phantom import make_phantom


class ReconstructionTests(unittest.TestCase):
    def test_calibrated_plane_and_unknown_volume_support(self):
        rows, cols = np.indices((32, 32))
        image = (rows+cols)/62
        rotation = np.diag([1., -1., -1.])
        observation = Observation(image, np.array([.01, .02, .03]), rotation, 0, -.01, .01, .002, .022)
        points = observation.world_points()
        values, normal, inside = observation.sample_world(points)
        np.testing.assert_allclose(values[1:-1, 1:-1], image[1:-1, 1:-1], atol=1e-6)
        self.assertTrue(inside[1:-1, 1:-1].all())
        np.testing.assert_allclose(normal, 0)
        volume = TrackedVolume.from_observations([observation], spacing=.001)
        volume.add(observation)
        reconstructed, confidence = volume.sample(points[4:-4, 4:-4])
        self.assertGreater(confidence.min(), .05)
        np.testing.assert_allclose(reconstructed, image[4:-4, 4:-4], atol=.015)
        _, unknown = volume.sample(np.array([[1., 1., 1.]]))
        self.assertEqual(unknown[0], 0)

    def test_image_motion_and_probe_motion_are_separated(self):
        rng = np.random.default_rng(12)
        texture = gaussian_filter(rng.normal(size=(64, 64)), 1)
        image = .3+.6*(texture-texture.min())/(texture.max()-texture.min())
        rotation = np.diag([1., -1., -1.])
        anchor = Observation(image, np.array([0., 0., .03]), rotation, 0, -.0063, .0063, .002, .0146)
        moved = Observation(shift(image, (2, 1), mode='nearest'), anchor.position, rotation, 1,
            anchor.lateral_min, anchor.lateral_max, anchor.depth_min, anchor.depth_max)
        result = estimate_inplane_motion(anchor, moved)
        valid = result['confidence']>.7
        measured = np.median(result['displacement'][valid], axis=0)
        np.testing.assert_allclose(measured, [.0002, 0, -.0004], atol=.00008)
        probe_moved = Observation(shift(image, (-2, 0), mode='nearest'), anchor.position+[0, 0, -.0004], rotation, 2,
            anchor.lateral_min, anchor.lateral_max, anchor.depth_min, anchor.depth_max)
        compensated = estimate_inplane_motion(anchor, probe_moved)
        valid = compensated['confidence']>.7
        self.assertLess(np.median(np.linalg.norm(compensated['displacement'][valid], axis=1)), .00004)
        out_of_plane = Observation(image, anchor.position+[0, .002, 0], rotation, 3)
        self.assertIsNone(estimate_inplane_motion(anchor, out_of_plane))
        rr, cc = np.mgrid[16:48:8, 16:48:8]
        world = anchor.world_points(rr.ravel(), cc.ravel())
        tracked = advance_material_tracks(anchor, moved, world, np.zeros_like(world), np.zeros_like(world))
        self.assertIsNotNone(tracked)
        displacement, confidence = tracked
        np.testing.assert_allclose(np.median(displacement[confidence>.7], axis=0), [.0002, 0, -.0004], atol=.00008)

    def test_elastic_prior_has_rigid_null_modes_and_fits_image_constraints(self):
        bodies, _ = make_phantom(5)
        prior = dict(body_count=2, youngs_modulus=np.array([15000., 60000.]))
        for i, b in enumerate(bodies):
            prior.update({f'points_{i}': b.points, f'tets_{i}': b.tetrahedra, f'ids_{i}': b.global_ids})
        inverse = ElasticInverse(prior)
        rigid = np.cross(np.array([.02, -.01, .03]), inverse.points)+[.001, .002, -.001]
        self.assertLess(np.linalg.norm(inverse.stiffness@rigid.ravel()), 1e-8)
        self.assertLess(np.max(np.abs((inverse.stiffness-inverse.stiffness.T).data), initial=0), 1e-10)
        x, z = np.meshgrid(np.linspace(-.012, .012, 5), np.linspace(.01, .03, 5))
        world = np.column_stack((x.ravel(), np.zeros(x.size), z.ravel()))
        inverse.prepare(world, np.array([[1, 0, 0], [0, 0, 1]]), np.ones(len(world)))
        positions, report = inverse.solve(np.column_stack((.01125*world[:, 0], -.025*world[:, 2])))
        self.assertLess(report['image_constraint_rms_mm'], .1)
        self.assertGreater(report['min_volume_jacobian'], .8)
        self.assertGreater(report['max_inferred_motion_mm'], .5)
        _, less_certain = inverse.solve(np.column_stack((.01125*world[:, 0], -.025*world[:, 2])), confidence=np.ones(len(world))*.5)
        self.assertGreater(less_certain['max_inferred_motion_mm'], .5)
        self.assertLess(less_certain['image_constraint_rms_mm'], .1)
        _, unobserved = inverse.solve(np.ones((len(world), 2)), confidence=np.zeros(len(world)))
        self.assertEqual(unobserved['measurement_count'], 0)
        self.assertLess(unobserved['max_inferred_motion_mm'], .001)
        _, stationary = inverse.solve(np.zeros((len(world), 2)))
        self.assertLess(stationary['max_inferred_motion_mm'], .001)

    def test_psf_point_target_localizes_and_coherent_signals_cancel(self):
        renderer = FastUltrasound()
        xyz = np.array([[0., 0., .02]])
        image = renderer.render(xyz, np.ones(1))
        row, col = np.unravel_index(image.argmax(), image.shape)
        self.assertLess(abs(row/(image.shape[0]-1)*.048+.002-.02), .0005)
        self.assertLess(abs(col/(image.shape[1]-1)*.04-.02), .0005)
        cancelled = renderer.render(np.repeat(xyz, 2, axis=0), np.array([1., -1.]))
        self.assertEqual(float(cancelled.max()), 0.)


if __name__ == '__main__':
    unittest.main()
