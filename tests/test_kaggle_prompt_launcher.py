import sys
import unittest
from unittest.mock import patch

import kaggle_full_pipeline_prompt
from kaggle_full_pipeline_prompt import _has_option, _option_value, _require_prompt


class KagglePromptLauncherTest(unittest.TestCase):
    def test_recognizes_space_and_equals_option_forms(self):
        arguments = ["--prompt=a wooden chair", "--lgm-root", "/kaggle/working/LGM"]
        self.assertTrue(_has_option(arguments, "--prompt"))
        self.assertEqual(_option_value(arguments, "--prompt"), "a wooden chair")
        self.assertEqual(_option_value(arguments, "--lgm-root"), "/kaggle/working/LGM")

    def test_requires_prompt_and_rejects_real_image_flags(self):
        with self.assertRaises(SystemExit):
            _require_prompt([])
        with self.assertRaises(SystemExit):
            _require_prompt(["--prompt", "chair", "--input-dir", "images"])
        _require_prompt(["--prompt", "chair"])

    def test_notebook_can_pass_prompt_directly_without_cell_sys_argv(self):
        original_argv = sys.argv
        try:
            with patch.object(kaggle_full_pipeline_prompt, "_verify_dual_t4"):
                with patch("full_pipeline.run.main") as common_runner:
                    kaggle_full_pipeline_prompt.main(
                        ["--prompt", "chair", "--lgm-root", "/tmp/LGM", "--no-bootstrap"]
                    )
            common_runner.assert_called_once()
            self.assertIn("--prompt", sys.argv)
            self.assertEqual(sys.argv[sys.argv.index("--prompt") + 1], "chair")
        finally:
            sys.argv = original_argv


if __name__ == "__main__":
    unittest.main()
