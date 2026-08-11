import sys
import unittest
from unittest.mock import patch

from open_mvdream_rlft import (
    MVDREAM_TO_LGM,
    PerPromptRunningNormalizer,
    grouped_normalized_advantages,
    parse_args,
    resolve_prompt_configuration,
)


class OpenRlftHelpersTest(unittest.TestCase):
    def test_mvdream_output_is_reordered_to_lgm_camera_order(self):
        self.assertEqual(MVDREAM_TO_LGM, (1, 2, 3, 0))

    def test_target_prompt_is_used_consistently_without_demo_fallbacks(self):
        with patch.object(
            sys,
            "argv",
            ["open_mvdream_rlft.py", "--lgm-root", "/tmp/LGM", "--target-prompt", "a steel staircase"],
        ):
            args = parse_args()
        prompts = resolve_prompt_configuration(args)
        self.assertEqual(prompts.training, ["a steel staircase"])
        self.assertEqual(prompts.validation, ["a steel staircase"])
        self.assertEqual(prompts.final, "a steel staircase")

    def test_explicit_training_prompts_are_reused_when_validation_is_omitted(self):
        with patch.object(
            sys,
            "argv",
            [
                "open_mvdream_rlft.py",
                "--lgm-root",
                "/tmp/LGM",
                "--prompt",
                "steel stairs front",
                "--prompt",
                "steel stairs side",
                "--prompts-per-update",
                "2",
            ],
        ):
            prompts = resolve_prompt_configuration(parse_args())
        self.assertEqual(prompts.training, ["steel stairs front", "steel stairs side"])
        self.assertEqual(prompts.validation, prompts.training)
        self.assertEqual(prompts.final, "steel stairs front")

    def test_missing_prompt_fails_instead_of_using_unrelated_demo_objects(self):
        with patch.object(sys, "argv", ["open_mvdream_rlft.py", "--lgm-root", "/tmp/LGM"]):
            args = parse_args()
        with self.assertRaisesRegex(ValueError, "explicit object description"):
            resolve_prompt_configuration(args)

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
        self.assertEqual(args.curate_prompt_count, 0)
        self.assertEqual(args.kl_coeff, 0.2)
        self.assertEqual(args.timestep_loss_reduction, "mean")
        self.assertEqual(args.advantage_clip, 5.0)
        self.assertEqual(args.stat_buffer_epochs, 3)
        self.assertEqual(args.min_validation_improvement, 1e-4)
        self.assertEqual(args.final_candidates, 1)
        self.assertIsNone(args.kl_early_stop_threshold)
        self.assertFalse(args.overlap_reward)
        self.assertIsNone(args.target_prompt)
        self.assertIsNone(args.final_prompt)
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

    def test_running_normalizer_retains_three_epoch_prompt_history(self):
        tracker = PerPromptRunningNormalizer(buffer_size=6, min_count=2)
        first = tracker.update([-1.0, -3.0], ["chair", "chair"])
        second = tracker.update([-2.0, -4.0], ["chair", "chair"])
        self.assertAlmostEqual(first[0], 1.0, places=5)
        self.assertAlmostEqual(first[1], -1.0, places=5)
        self.assertAlmostEqual(second[0], 0.447213, places=5)
        self.assertAlmostEqual(second[1], -1.341639, places=5)
        self.assertEqual(tracker.summary()["chair"]["count"], 4)


if __name__ == "__main__":
    unittest.main()
