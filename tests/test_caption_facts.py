from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from synwts.caption_facts import augment_caption_dataset_with_facts


class CaptionFactsTest(unittest.TestCase):
    def test_injects_predicted_wts_phase_facts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "caption.json"
            vqa_json = root / "vqa.json"
            vqa_submission = root / "vqa_submission.json"
            output = root / "caption_fact.json"

            dataset.write_text(
                json.dumps(
                    [
                        {
                            "instruction": "Prompt\n<video>\n\nReturn JSON with exactly these keys:",
                            "input": "",
                            "output": "{}",
                            "videos": ["/tmp/video.mp4"],
                            "metadata": {
                                "task": "caption",
                                "scenario_id": "20230707_11_SY3_T1",
                                "scenario_type": "event",
                                "phase": "4",
                                "view": "overhead_view",
                            },
                        }
                    ]
                ),
                encoding="utf-8",
            )
            vqa_json.write_text(
                json.dumps(
                    [
                        {
                            "videos": ["20230707_11_SY3_T1_192.168.0.11_1.mp4"],
                            "event_phase": [
                                {
                                    "labels": ["4"],
                                    "conversations": [
                                        {
                                            "id": "q1",
                                            "question": "What is pedestrian's action?",
                                            "a": "walking",
                                            "b": "standing",
                                        }
                                    ],
                                }
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            vqa_submission.write_text(json.dumps([{"id": "q1", "correct": "b"}]), encoding="utf-8")

            rows = augment_caption_dataset_with_facts(
                dataset,
                output,
                wts_vqa_json=vqa_json,
                vqa_submission=vqa_submission,
                dataset_info_output=root / "dataset_info.json",
                dataset_name="caption_fact",
            )

            instruction = rows[0]["instruction"]
            self.assertIn("VQA fact context", instruction)
            self.assertIn("pedestrian.action: standing", instruction)
            self.assertLess(
                instruction.index("VQA fact context"),
                instruction.index("Return JSON with exactly these keys:"),
            )
            self.assertTrue(rows[0]["metadata"]["fact_conditioned"])


if __name__ == "__main__":
    unittest.main()
