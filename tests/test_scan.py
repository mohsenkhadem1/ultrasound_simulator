import unittest
import numpy as np
from deform_sim.probe_motion import scan_pose, contact_height_function
from deform_sim.phantom import make_phantom, BodyMesh
from deform_sim.slice_geometry import plane_segments


class ScanTests(unittest.TestCase):
    def test_contact_height_excludes_high_tissue_outside_actual_head(self):
        bodies, _ = make_phantom(5)
        high = [BodyMesh(b.points+[0,.02,.02], b.tetrahedra, b.global_ids+10000) for b in bodies]
        height = contact_height_function(bodies+high)
        self.assertAlmostEqual(height(0.,0.), .04)
        self.assertAlmostEqual(height(0.,.02), .06)

    def test_scan_crosses_target_adds_pressure_then_lifts_without_pose_jumps(self):
        terrain = (np.array([-.04, .04]), np.array([.14, .14]))
        start = scan_pose(0, 6, terrain)
        center = scan_pose(3, 6, terrain)
        finish = scan_pose(6, 6, terrain)
        self.assertLess(start[0][1], -.03)
        self.assertGreater(finish[0][1], .03)
        self.assertAlmostEqual(center[0][1], 0)
        self.assertAlmostEqual(center[2], .005)
        self.assertAlmostEqual(finish[2], 0)
        for fraction in [.08, .36, .44, .52, .60, .88]:
            a = scan_pose(fraction*6-1e-6, 6, terrain)[0]
            b = scan_pose(fraction*6+1e-6, 6, terrain)[0]
            self.assertLess(np.linalg.norm(a-b), 1e-7)

    def test_tumour_reference_intersection_matches_known_triangular_surface(self):
        p = np.array([[-.01, -.01, .02], [.01, -.01, .02], [0, .01, .04]])
        segments = plane_segments(p, np.array([[0, 1, 2]]))
        np.testing.assert_allclose(segments[0, :, 1], .03)
        np.testing.assert_allclose(np.sort(segments[0, :, 0]), [-.005, .005])
        self.assertEqual(len(plane_segments(p+[0, .03, 0], np.array([[0, 1, 2]]))), 0)


if __name__ == '__main__':
    unittest.main()
