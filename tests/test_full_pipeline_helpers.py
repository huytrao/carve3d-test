from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from full_pipeline.run import _prepare_rgb, find_view_paths, parse_args


class FindViewPathsTest(unittest.TestCase):
    def test_returns_canonical_angle_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for filename in ("view_270.jpg", "view_000.png", "view_180.webp", "view_090.jpeg"):
                (root / filename).touch()
            self.assertEqual(
                [path.name for path in find_view_paths(root)],
                ["view_000.png", "view_090.jpeg", "view_180.webp", "view_270.jpg"],
            )

    def test_requires_all_four_named_views(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "view_000.png").touch()
            with self.assertRaises(FileNotFoundError):
                find_view_paths(root)

    def test_unmatted_images_are_normalized_to_lgm_resolution(self):
        source = np.zeros((120, 340, 3), dtype=np.uint8)
        prepared = _prepare_rgb(source, remove_background=False, recenter_foreground=False)
        self.assertEqual(prepared.shape, (256, 256, 3))
        self.assertEqual(prepared.dtype, np.float32)

    def test_accepts_negative_and_positive_elevation_search_candidates(self):
        with patch.object(
            sys,
            "argv",
            ["run.py", "--input-dir", "/tmp/views", "--elevation-candidates", "-10", "-5", "0", "5", "10"],
        ):
            args = parse_args()
        self.assertEqual(args.elevation_candidates, [-10.0, -5.0, 0.0, 5.0, 10.0])

    def test_exposes_safe_direct_appearance_search_argument(self):
        with patch.object(
            sys,
            "argv",
            [
                "run.py",
                "--input-dir",
                "/tmp/views",
                "--appearance-search",
            ],
        ):
            args = parse_args()
        self.assertTrue(args.appearance_search)

    def test_accepts_legacy_refinement_flags_for_old_kaggle_cells(self):
        with patch.object(
            sys,
            "argv",
            ["run.py", "--input-dir", "/tmp/views", "--refine-steps", "120"],
        ):
            args = parse_args()
        self.assertEqual(args.refine_steps, 120)


if __name__ == "__main__":
    unittest.main()
