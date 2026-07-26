import json
import tempfile
import unittest
from pathlib import Path

from synwts.option_submission import export_vqa_option_candidates_from_inference_dataset


class OptionSubmissionTest(unittest.TestCase):
    def test_export_prompt_variant_preserves_question_options_and_videos(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "vqa.json"
            output = root / "candidates.json"
            dataset.write_text(
                json.dumps(
                    [
                        {
                            "instruction": (
                                "You are a traffic safety visual question answering model. "
                                "Answer the multiple-choice question from the visual evidence and return only the option letter.\n\n"
                                "Scenario type: event\n"
                                "Scope: environment\n\n"
                                "<video>\n\n"
                                "Question: What is the weather?\n"
                                "Options:\n"
                                "a. clear\n"
                                "b. cloudy\n\n"
                                "Return only the correct option letter."
                            ),
                            "input": "",
                            "output": "",
                            "videos": ["clip.mp4"],
                            "metadata": {"task": "vqa", "vqa_id": "q1"},
                        }
                    ]
                ),
                encoding="utf-8",
            )

            rows = export_vqa_option_candidates_from_inference_dataset(
                dataset=dataset,
                output=output,
                prompt_variant="evidence",
            )

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["metadata"]["prompt_variant"], "evidence")
            self.assertEqual(rows[0]["options"], {"a": "clear", "b": "cloudy"})
            self.assertIn("<video>", rows[0]["instruction"])
            self.assertIn("Question: What is the weather?", rows[0]["instruction"])
            self.assertIn("Compare every option", rows[0]["instruction"])


if __name__ == "__main__":
    unittest.main()
