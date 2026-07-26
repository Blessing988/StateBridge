import json
import tempfile
import unittest
from pathlib import Path

from synwts.caption_realizer import (
    assemble_caption_realizer_submission,
    export_caption_realizer_test,
)


class CaptionRealizerTest(unittest.TestCase):
    def test_export_test_rows_from_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cap = {
                "s1": [
                    {
                        "labels": ["4"],
                        "caption_pedestrian": "The pedestrian, a male in his 20s, watched the vehicle.",
                        "caption_vehicle": "The vehicle is positioned in front of the pedestrian.",
                    }
                ]
            }
            base = root / "base.json"
            other = root / "other.json"
            base.write_text(json.dumps(cap), encoding="utf-8")
            other.write_text(json.dumps(cap), encoding="utf-8")
            out = root / "realizer.json"

            rows = export_caption_realizer_test(
                captions={"base": base, "other": other},
                fallback_name="base",
                output=out,
            )

            self.assertEqual(len(rows), 1)
            self.assertIn("[base]", rows[0]["instruction"])
            self.assertIn("[other]", rows[0]["instruction"])
            self.assertEqual(rows[0]["metadata"]["scenario_id"], "s1")

    def test_assemble_uses_fallback_for_bad_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fallback = {
                "s1": [
                    {
                        "labels": ["4"],
                        "caption_pedestrian": "The pedestrian was near the vehicle and watched it carefully for a long time.",
                        "caption_vehicle": "The vehicle was near the pedestrian and kept the pedestrian in its field of view.",
                    }
                ]
            }
            dataset = [
                {
                    "metadata": {
                        "scenario_id": "s1",
                        "phase": "4",
                        "fallback_idx": 0,
                        "fallback_caption": {
                            "caption_pedestrian": fallback["s1"][0]["caption_pedestrian"],
                            "caption_vehicle": fallback["s1"][0]["caption_vehicle"],
                        },
                    }
                }
            ]
            fallback_path = root / "fallback.json"
            dataset_path = root / "dataset.json"
            pred_path = root / "pred.jsonl"
            out = root / "out.json"
            fallback_path.write_text(json.dumps(fallback), encoding="utf-8")
            dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
            pred_path.write_text(json.dumps({"predict": "bad"}) + "\n", encoding="utf-8")

            submission, report = assemble_caption_realizer_submission(
                inference_dataset=dataset_path,
                predictions=pred_path,
                fallback_caption=fallback_path,
                output=out,
            )

            self.assertEqual(report["parse_failed"], 1)
            self.assertEqual(submission["s1"][0]["caption_pedestrian"], fallback["s1"][0]["caption_pedestrian"])


if __name__ == "__main__":
    unittest.main()
