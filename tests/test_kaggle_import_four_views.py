import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from kaggle_import_four_views import resolve_input_dir, select_lgm_device, validate_four_views


class KaggleImportFourViewsTest(unittest.TestCase):
    def _write_views(self, directory: Path) -> None:
        for angle in (0, 90, 180, 270):
            (directory / f"view_{angle:03d}.png").write_bytes(b"not-decoded-in-validation")

    def test_validate_four_views_returns_canonical_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self._write_views(directory)
            result = validate_four_views(directory)
        self.assertEqual([path.name for path in result], ["view_000.png", "view_090.png", "view_180.png", "view_270.png"])

    def test_validate_four_views_names_missing_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "view_180.png").write_bytes(b"test")
            (directory / "view_270.png").write_bytes(b"test")
            with self.assertRaisesRegex(FileNotFoundError, "view_000.png.*view_090.png"):
                validate_four_views(directory)

    def test_resolve_input_dir_requires_existing_folder(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.assertEqual(resolve_input_dir(temporary), Path(temporary))
        with self.assertRaisesRegex(FileNotFoundError, "Could not find"):
            resolve_input_dir("/definitely/not/a/kaggle/input")

    def test_selects_second_t4_when_not_explicitly_set(self):
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        fake_torch.cuda.device_count.return_value = 2
        fake_torch.cuda.get_device_name.side_effect = ("Tesla T4", "Tesla T4")
        with patch.dict(sys.modules, {"torch": fake_torch}):
            self.assertEqual(select_lgm_device(None), 1)


if __name__ == "__main__":
    unittest.main()
