import unittest

from test_prompt_only import ANGLES


class PromptOnlyTest(unittest.TestCase):
    def test_prompt_views_use_four_canonical_angles(self):
        self.assertEqual(ANGLES, (0, 90, 180, 270))


if __name__ == "__main__":
    unittest.main()
