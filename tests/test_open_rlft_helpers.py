import sys
import unittest
from unittest.mock import patch

from open_mvdream_rlft import (
    MVDREAM_TO_LGM,
    default_prompts,
    grouped_normalized_advantages,
    parse_args,
)


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
        self.assertEqual(args.samples_per_epoch, 4)
        self.assertEqual(args.epochs, 16)
        self.assertEqual(args.num_steps, 30)
        self.assertEqual(args.lora_rank, 4)
        self.assertEqual(args.learning_rate, 1e-5)
        self.assertEqual(args.prompts_per_update, 1)
        self.assertEqual(args.kl_coeff, 0.2)
        self.assertEqual(args.timestep_loss_reduction, "mean")
        self.assertEqual(args.advantage_clip, 5.0)
        self.assertEqual(args.min_validation_improvement, 1e-4)
        self.assertEqual(args.final_candidates, 1)
        self.assertFalse(args.transactional_validation)

    def test_advantages_are_normalized_independently_per_prompt(self):
        advantages = grouped_normalized_advantages(
            [-1.0, -3.0, 10.0, 14.0],
            ["chair", "chair", "teapot", "teapot"],
        )
        self.assertEqual(advantages, [1.0, -1.0, -1.0, 1.0])

    def test_constant_prompt_group_has_zero_advantage(self):
        advantages = grouped_normalized_advantages([2.0, 2.0], ["chair", "chair"])
        self.assertEqual(advantages, [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
