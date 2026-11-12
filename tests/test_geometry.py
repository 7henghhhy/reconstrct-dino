"""Regression checks for SfM pose conventions, gauge fixing and failed PnP."""

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from ba import run_bundle_adjustment
from initialize import build_pose_matrix, project_points, triangulate_points, triangulation_mask
from main import reprojection_stats
from reconstruct import Reconstruction, solve_pnp_pose


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(4)
        self.K = np.array([[800., 0., 320.], [0., 800., 240.], [0., 0., 1.]])
        self.points = self.rng.uniform([-1., -0.6, 4.], [1., 0.6, 7.], (50, 3))
        self.poses = np.array([np.eye(4), build_pose_matrix(np.eye(3), [-1., 0., 0.]),
                               build_pose_matrix(np.eye(3), [-0.5, -0.2, 0.])])

    def test_triangulation_and_camera_convention(self):
        xy1 = project_points(self.points, self.K, self.poses[0])
        xy2 = project_points(self.points, self.K, self.poses[1])
        recovered = triangulate_points(xy1, xy2, self.K, *self.poses[:2])
        np.testing.assert_allclose(recovered, self.points, atol=1e-9)
        self.assertTrue(triangulation_mask(recovered, xy1, xy2, self.K, *self.poses[:2]).all())
        self.assertFalse(triangulation_mask(-recovered, xy1, xy2, self.K, *self.poses[:2]).any())

    def test_pnp_refinement_and_failure(self):
        cv2.setRNGSeed(0)
        observed = project_points(self.points, self.K, self.poses[2])
        observed[:8] = self.rng.uniform([0, 0], [640, 480], (8, 2))
        recovered, inliers = solve_pnp_pose(self.points, observed, self.K)
        np.testing.assert_allclose(recovered, self.poses[2], atol=1e-6)
        self.assertGreaterEqual(len(inliers), 40)
        with self.assertRaises(ValueError):
            solve_pnp_pose(self.points[:5], observed[:5], self.K)

    def test_bundle_adjustment_preserves_gauge(self):
        camera_ids = np.repeat(np.arange(3), len(self.points))
        point_ids = np.tile(np.arange(len(self.points)), 3)
        observed = np.concatenate([project_points(self.points, self.K, T) for T in self.poses])
        noisy = self.points + self.rng.normal(0, 0.02, self.points.shape)
        poses = self.poses.copy()
        poses[2, 0, 3] += 0.03
        before = reprojection_stats(poses, noisy, camera_ids, point_ids, observed, self.K)
        optimized, points, info = run_bundle_adjustment(poses, noisy, camera_ids, point_ids, observed, self.K, 80)
        after = reprojection_stats(optimized, points, camera_ids, point_ids, observed, self.K)
        self.assertLess(after['mean_px'], before['mean_px'] * 0.01)
        np.testing.assert_array_equal(optimized[0], np.eye(4))
        self.assertAlmostEqual(optimized[1, 0, 3], -1.0, places=12)
        self.assertTrue(info['converged'])

    def test_failed_candidate_is_retried_without_registering_it(self):
        state = Reconstruction([[], [], []], {}, self.K, poses={0: np.eye(4)})
        state.feature_to_point = [{}, {}, {}]
        values = (self.points, np.zeros((50, 2)), list(zip(range(50), range(50))))
        with patch.object(state, 'correspondences', return_value=values), \
             patch('reconstruct.solve_pnp_pose', side_effect=[ValueError('bad candidate'), (self.poses[2], [])]), \
             patch.object(state, 'triangulate_new_points'):
            self.assertEqual(state.register_next(), 2)
        self.assertNotIn(1, state.poses)


if __name__ == '__main__':
    unittest.main()
