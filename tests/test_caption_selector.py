import json
from pathlib import Path
import tempfile
import unittest

from synwts.caption_selector import (
    assemble_caption_selector_submission,
    export_caption_selector_test,
    repair_caption_consistency,
)


class CaptionSelectorTest(unittest.TestCase):
    def test_export_selector_test_creates_one_row_per_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cap = {
                "s1": [
                    {
                        "labels": ["4"],
                        "caption_pedestrian": "The pedestrian, a male in his 20s, watched the vehicle near the road.",
                        "caption_vehicle": "The vehicle is positioned near the pedestrian with a clear field of view.",
                    }
                ]
            }
            base = root / "base.json"
            other = root / "other.json"
            out = root / "selector.json"
            base.write_text(json.dumps(cap), encoding="utf-8")
            other.write_text(json.dumps(cap), encoding="utf-8")

            rows = export_caption_selector_test(
                captions={"base": base, "other": other},
                fallback_name="base",
                output=out,
            )

            self.assertEqual(len(rows), 2)
            self.assertIn("Answer only good or bad", rows[0]["instruction"])

    def test_assemble_selector_uses_good_nonfallback_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = {
                "s1": [
                    {
                        "labels": ["4"],
                        "caption_pedestrian": "The pedestrian was near the vehicle and watched it carefully.",
                        "caption_vehicle": "The vehicle was near the pedestrian and stopped.",
                    }
                ]
            }
            other = {
                "s1": [
                    {
                        "labels": ["4"],
                        "caption_pedestrian": "The pedestrian was near the vehicle and watched it carefully on the dry road.",
                        "caption_vehicle": "The vehicle was near the pedestrian and stopped with a clear field of view.",
                    }
                ]
            }
            dataset = [
                {"metadata": {"scenario_id": "s1", "phase": "4", "fallback_idx": 0, "source": "base"}},
                {"metadata": {"scenario_id": "s1", "phase": "4", "fallback_idx": 0, "source": "other"}},
            ]
            paths = {}
            for name, obj in {"base": base, "other": other, "dataset": dataset}.items():
                path = root / f"{name}.json"
                path.write_text(json.dumps(obj), encoding="utf-8")
                paths[name] = path
            pred = root / "pred.jsonl"
            pred.write_text(json.dumps({"predict": "bad"}) + "\n" + json.dumps({"predict": "good"}) + "\n", encoding="utf-8")
            out = root / "out.json"

            submission, report = assemble_caption_selector_submission(
                inference_dataset=paths["dataset"],
                predictions=pred,
                captions={"base": paths["base"], "other": paths["other"]},
                fallback_name="base",
                output=out,
                max_changed_rows=10,
            )

            self.assertEqual(report["changed_rows"], 1)
            self.assertIn("dry road", submission["s1"][0]["caption_pedestrian"])

    def test_consistency_repair_falls_back_on_age_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = {
                "20230707_11_SY3_T1": [
                    {
                        "labels": ["4"],
                        "caption_pedestrian": "The pedestrian, a male in his 70s, watched the vehicle.",
                        "caption_vehicle": "The vehicle was close to the pedestrian.",
                    }
                ]
            }
            fallback = {
                "20230707_11_SY3_T1": [
                    {
                        "labels": ["4"],
                        "caption_pedestrian": "The pedestrian, a male in his 20s, watched the vehicle.",
                        "caption_vehicle": "The vehicle was close to the pedestrian.",
                    }
                ]
            }
            vqa_json = [
                {
                    "videos": ["20230707_11_SY3_T1_192.168.0.11_1.mp4"],
                    "conversations": [
                        {"id": "age", "question": "What is the age group of the pedestrian?", "a": "20s", "b": "70s"}
                    ],
                }
            ]
            vqa_sub = [{"id": "age", "correct": "a"}]
            paths = {}
            for name, obj in {"candidate": candidate, "fallback": fallback, "vqa": vqa_json, "sub": vqa_sub}.items():
                path = root / f"{name}.json"
                path.write_text(json.dumps(obj), encoding="utf-8")
                paths[name] = path
            out = root / "out.json"
            repaired, report = repair_caption_consistency(
                caption=paths["candidate"],
                fallback_caption=paths["fallback"],
                output=out,
                wts_vqa_json=paths["vqa"],
                vqa_submission=paths["sub"],
            )

            self.assertEqual(report["changed_rows"], 1)
            self.assertIn("20s", repaired["20230707_11_SY3_T1"][0]["caption_pedestrian"])


if __name__ == "__main__":
    unittest.main()
