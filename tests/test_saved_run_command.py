import os
import unittest
from unittest.mock import patch

from run_full_pipeline import DEFAULT_PROMPT, build_pipeline_arguments


class SavedRunCommandTest(unittest.TestCase):
    def test_uses_reproducible_default_prompt(self):
        with patch.dict(os.environ, {}, clear=True):
            arguments = build_pipeline_arguments()
        self.assertEqual(arguments[arguments.index("--prompt") + 1], DEFAULT_PROMPT)
        self.assertEqual(arguments[arguments.index("--seed") + 1], "42")

    def test_allows_kaggle_environment_overrides(self):
        with patch.dict(
            os.environ,
            {
                "CARVE3D_PROMPT": "a ceramic teapot",
                "CARVE3D_OUTPUT_DIR": "/tmp/out",
                "CARVE3D_EXTRA_ARGS": "--render-size 256 --no-orbit",
            },
            clear=True,
        ):
            arguments = build_pipeline_arguments()
        self.assertEqual(arguments[arguments.index("--prompt") + 1], "a ceramic teapot")
        self.assertEqual(arguments[arguments.index("--output-dir") + 1], "/tmp/out")
        self.assertEqual(arguments[-3:], ["--render-size", "256", "--no-orbit"])


if __name__ == "__main__":
    unittest.main()
