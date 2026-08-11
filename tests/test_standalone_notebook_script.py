from pathlib import Path
import unittest


class StandaloneNotebookScriptTest(unittest.TestCase):
    def test_standalone_script_contains_no_dunder_file_dependency(self):
        script = (Path(__file__).parents[1] / "kaggle_full_pipeline_prompt_notebook.py").read_text()
        self.assertNotIn("__file__", script)
        self.assertIn("kaggle_full_pipeline_prompt", script)
        self.assertIn("implement_full_pipeline", script)


if __name__ == "__main__":
    unittest.main()
