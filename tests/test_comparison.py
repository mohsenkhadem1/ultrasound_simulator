"""Geometric contour and scoring checks independent of the viewer."""
import unittest
import numpy as np
from deform_sim.comparison import slice_surface, contour_mask, contour_metrics
from deform_sim.comparison_mesh import cluster_skin


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.points = np.array([[x, y, z] for x in [-.005, .005]
                                for y in [-.003, .003] for z in [.02, .03]])
        quads = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1],
                 [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]
        self.faces = np.array([t for a, b, c, d in quads for t in [[a, b, c], [a, c, d]]])

    def test_calibrated_plane_intersection_and_known_translation(self):
        segments = slice_surface(self.points, self.faces, np.zeros(3), np.eye(3))
        self.assertEqual(segments.shape, (8, 2, 2))
        np.testing.assert_allclose(segments.min(axis=(0, 1)), [-.005, .02])
        np.testing.assert_allclose(segments.max(axis=(0, 1)), [.005, .03])
        moved = slice_surface(self.points, self.faces, np.array([.002, 0, .004]), np.eye(3))
        np.testing.assert_allclose(moved, segments-[.002, .004])
        self.assertEqual(len(slice_surface(self.points, self.faces, np.array([0, .01, 0]), np.eye(3))), 0)

    def test_overlap_and_distance_penalize_wrong_contour(self):
        reference = slice_surface(self.points, self.faces, np.zeros(3), np.eye(3))
        exact = contour_metrics(reference, reference)
        self.assertEqual(exact['dice'], 1.)
        self.assertEqual(exact['mean_contour_distance_mm'], 0.)
        shifted = contour_metrics(reference+[.002, 0], reference)
        self.assertAlmostEqual(shifted['dice'], .8, delta=.02)
        self.assertGreater(shifted['mean_contour_distance_mm'], .5)
        self.assertGreater(shifted['contour_hd95_mm'], 1.9)

    def test_unknown_tumour_is_not_perfect_overlap(self):
        empty = np.empty((0, 2, 2))
        self.assertFalse(contour_mask(empty).any())
        metrics = contour_metrics(empty, empty)
        self.assertIsNone(metrics['dice'])
        self.assertIsNone(metrics['mean_contour_distance_mm'])

    def test_triangle_skin_binding_preserves_measured_fem_translation(self):
        nodes = np.tile(np.arange(len(self.points))[:, None], (1, 4))
        weights = np.tile([1., 0, 0, 0], (len(self.points), 1))
        rest, faces, binding = cluster_skin(self.points, self.faces, nodes, weights,
            len(self.points), spacing=.001)
        displacement = np.tile([.002, -.001, .003], (len(self.points), 1))
        np.testing.assert_allclose(binding@displacement, np.tile(displacement[0], (len(rest), 1)))
        self.assertEqual(faces.shape, self.faces.shape)
        self.assertTrue(np.isfinite(rest).all())


if __name__ == '__main__':
    unittest.main()
