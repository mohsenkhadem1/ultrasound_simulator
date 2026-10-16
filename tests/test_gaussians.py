import unittest
import importlib.util
import numpy as np
from deform_sim.gaussians import advect_gaussians, neo_hookean_diagnostics, pack_gaussian_frame
from deform_sim.live_server import LiveServer


class GaussianTests(unittest.TestCase):
    def setUp(self):
        self.vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        self.model = dict(soft_count=1, means=np.array([[.2, .3, .1], [0, 0, .02]], np.float32),
            covariance=np.array([np.diag([.01, .02, .003])]*2, np.float32),
            nodes=np.array([[0, 1, 2, 3]]), weights=np.array([[.4, .2, .3, .1]]),
            inverse_rest_edges=np.eye(3)[None], youngs=np.array([3000., 0]),
            colors=np.ones((2, 3)), opacity=np.ones(2), semantic=np.array([0, 3]))

    def test_affine_material_transport_and_rigid_probe(self):
        f = np.array([[.9, .2, 0], [0, 1.1, .1], [0, 0, .8]])
        offset = np.array([.1, -.2, .3])
        means, cov, disp, stress, j = advect_gaussians(self.model, [self.vertices@f.T+offset], [1, 2, 3, 0, 0, 0, 1])
        np.testing.assert_allclose(means[0], f@self.model['means'][0]+offset, atol=1e-7)
        np.testing.assert_allclose(cov[0], f@self.model['covariance'][0]@f.T, atol=1e-8)
        np.testing.assert_allclose(means[1], [1, 2, 3.02], atol=1e-6)
        self.assertTrue(np.all(np.linalg.eigvalsh(cov)>0))
        self.assertAlmostEqual(j, np.linalg.det(f))
        self.assertGreater(stress[0], 0)
        self.assertEqual(disp[1], 0)

    def test_rigid_rotation_has_no_constitutive_stress(self):
        angle = .7
        r = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        j, stress = neo_hookean_diagnostics(r[None], np.array([3000.]))
        np.testing.assert_allclose(j, 1)
        np.testing.assert_allclose(stress, 0, atol=1e-10)

    def test_binary_protocol_and_invalid_control(self):
        packet, info = pack_gaussian_frame(self.model, [self.vertices], [0, 0, 0, 0, 0, 0, 1], [1, 0, 42, 2])
        values = np.frombuffer(packet, np.float32)
        self.assertEqual(len(values), 16+2*11)
        self.assertEqual(values[2], 42)
        self.assertAlmostEqual(info['min_jacobian'], 1)
        server = LiveServer(self.model)
        self.assertEqual(server.command(dict(press_mm=500))['press_mm'], 5)
        for data in [[], dict(press_mm=float('nan')), dict(paused=1)]:
            with self.assertRaises(ValueError):
                server.command(data)

    @unittest.skipUnless(importlib.util.find_spec('warp'), 'optional Isaac/Warp environment required')
    def test_warp_transport_matches_cpu_affine_and_rigid_rotation(self):
        import warp as wp
        from deform_sim.gaussians_warp import WarpGaussianTransport
        wp.init()
        device = 'cuda:0' if wp.is_cuda_available() else 'cpu'
        transport = WarpGaussianTransport(self.model, device)
        f = np.array([[.9, .1, 0], [0, 1.1, .05], [0, 0, .95]], np.float32)
        pose = [.1, .2, .3, 0, 0, np.sin(.2), np.cos(.2)]
        positions = [self.vertices@f.T]
        expected, _ = pack_gaussian_frame(self.model, positions, pose, [1, .2, 3, 2])
        actual, _ = transport.pack(positions, pose, [1, .2, 3, 2])
        np.testing.assert_allclose(np.frombuffer(actual, np.float32),
            np.frombuffer(expected, np.float32), rtol=1e-4, atol=.002)


if __name__ == '__main__':
    unittest.main()
