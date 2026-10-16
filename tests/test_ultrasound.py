import unittest
import numpy as np
from deform_sim.ultrasound import Acquisition, Ultrasound, log_compress, iter_snapshots


class AcousticTests(unittest.TestCase):
    def test_point_target_localizes_at_correct_depth(self):
        renderer = Ultrasound(Acquisition(elements=16, angles_deg=(0.,), width_pixels=65, depth_pixels=97))
        iq, rf = renderer.render(np.array([[0., 0., .015]]), np.array([1.]))
        peak = np.unravel_index(np.argmax(np.abs(iq)), iq.shape)
        self.assertLess(abs(renderer.z[peak]-.015), .001)
        self.assertLess(abs(renderer.x[peak]), .001)
        self.assertEqual(rf[0].shape[1], 16)
        self.assertTrue(np.isfinite(rf[0]).all())

    def test_fixed_display_gain_preserves_frame_amplitude_changes(self):
        full = log_compress(np.array([1.]), 1., 60.)
        half = log_compress(np.array([.5]), 1., 60.)
        self.assertGreater(full[0], half[0])

    def test_watch_drains_frames_created_during_render(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'state').mkdir()
            (root/'state/00000.npz').touch()
            frames = iter_snapshots(root, watch=True)
            self.assertEqual(next(frames).name, '00000.npz')
            # Producer finishes while consumer is rendering the first frame.
            (root/'state/00001.npz').touch()
            (root/'state/00002.partial.npz').touch()
            (root/'complete.json').touch()
            self.assertEqual([p.name for p in frames], ['00001.npz'])


if __name__ == '__main__':
    unittest.main()
