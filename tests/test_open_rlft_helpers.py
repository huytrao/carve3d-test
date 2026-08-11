import sys
import unittest
from unittest.mock import patch

from open_mvdream_rlft import MVDREAM_TO_LGM, default_prompts, parse_args


class OpenRlftHelpersTest(unittest.TestCase):
    def test_mvdream_output_is_reordered_to_lgm_camera_order(self):
        self.assertEqual(MVDREAM_TO_LGM, (1, 2, 3, 0))

    def test_default_training_set_has_multiple_prompts_for_advantages(self):
        prompts = default_prompts()
        self.assertGreaterEqual(len(prompts), 2)
        self.assertTrue(all(isinstance(prompt, str) and prompt for prompt in prompts))

    def test_rlft_parser_exposes_separate_devices_and_required_lgm_root(self):
        with patch.object(sys, "argv", ["open_mvdream_rlft.py", "--lgm-root", "/tmp/LGM"]):
            args = parse_args()
        self.assertEqual(str(args.lgm_root), "/tmp/LGM")
        self.assertEqual((args.diffusion_device, args.lgm_device), (0, 1))
        self.assertEqual(args.samples_per_epoch, 2)


if __name__ == "__main__":
    unittest.main()
