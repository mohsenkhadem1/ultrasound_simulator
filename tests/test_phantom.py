import unittest
import numpy as np
from deform_sim.phantom import make_phantom, sample_scatterers, advect, probe_coordinates, acoustic_scatterers, BodyMesh, load_bodies, surface_positions


class PhantomTests(unittest.TestCase):
    def test_scatterers_cover_more_than_two_regions_and_keep_their_material(self):
        base = self.bodies[1]
        bodies = [BodyMesh(base.points+[i*.02, 0, 0], base.tetrahedra, base.global_ids+i*1000) for i in range(4)]
        scatter = sample_scatterers(bodies, 2000, 12, amplitudes=[1., 2., 3., 4.])
        np.testing.assert_array_equal(np.unique(scatter['body']), [0, 1, 2, 3])
        points = advect(bodies, [b.points for b in bodies], scatter)
        moved = advect(bodies, [b.points+[i*.001, 0, 0] for i, b in enumerate(bodies)], scatter)
        np.testing.assert_allclose(moved-points, np.column_stack([scatter['body']*.001, np.zeros((2000, 2))]), atol=1e-16)

    def test_detailed_skin_follows_translation_across_material_boundaries(self):
        reference = {'body_count': 2}
        for i, body in enumerate(self.bodies):
            reference.update({f'points_{i}': body.points, f'tets_{i}': body.tetrahedra, f'ids_{i}': body.global_ids})
        reference.update(display_points_0=np.array([[.001, 0, .02]]), display_nodes_0=np.array([[0, len(self.bodies[0].points)]]), display_weights_0=np.array([[.3, .7]]))
        self.assertEqual(len(load_bodies(reference)), 2)
        shift = np.array([.001, -.002, .003])
        skin = surface_positions(reference, [b.points+shift for b in self.bodies], 0)
        np.testing.assert_allclose(skin, reference['display_points_0']+shift)

    def test_top_probe_coordinates_follow_probe_motion(self):
        rotation = np.diag([1., -1., -1.])
        points = np.array([[.005, .002, .025]])
        before = probe_coordinates(points, [0, 0, .044], rotation)
        after = probe_coordinates(points, [0, 0, .040], rotation)
        np.testing.assert_allclose(before, [[.005, -.002, .019]])
        np.testing.assert_allclose(after, [[.005, -.002, .015]])

    def test_scatterers_behind_moving_aperture_are_excluded(self):
        xyz, coefficients = acoustic_scatterers(np.array([[0, 0, .02], [0, 0, .05]]),
            np.array([2., 7.]), dict(probe_position=np.array([0, 0, .044]), probe_rotation=np.diag([1., -1., -1.])))
        np.testing.assert_allclose(xyz, [[0, 0, .024]])
        np.testing.assert_array_equal(coefficients, [2.])

    def setUp(self):
        self.bodies, self.interface = make_phantom(7)

    def test_conforming_bond(self):
        a, b = self.interface
        self.assertGreater(len(a), 0)
        np.testing.assert_array_equal(self.bodies[0].points[a], self.bodies[1].points[b])

    def test_partition_has_positive_tets_and_exact_block_volume(self):
        total = 0
        for body in self.bodies:
            x = body.points[body.tetrahedra]
            determinants = np.linalg.det((x[:, 1:]-x[:, :1]).transpose(0, 2, 1))
            self.assertTrue(np.all(determinants > 0))
            total += determinants.sum()/6
        self.assertAlmostEqual(total, 0.04*0.02*0.04, places=12)

    def test_material_scatterers_follow_affine_motion_exactly(self):
        scatter = sample_scatterers(self.bodies, 200, 1)
        reference = advect(self.bodies, [b.points for b in self.bodies], scatter)
        transform = np.array([[1.1, .1, 0], [0, 0.9, .2], [0, .1, .95]])
        shift = np.array([.001, -.002, .003])
        moved = [b.points @ transform.T + shift for b in self.bodies]
        np.testing.assert_allclose(advect(self.bodies, moved, scatter), reference @ transform.T+shift)


if __name__ == '__main__':
    unittest.main()
