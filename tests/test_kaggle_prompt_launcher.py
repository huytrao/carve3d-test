import unittest

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


if __name__ == "__main__":
    unittest.main()
