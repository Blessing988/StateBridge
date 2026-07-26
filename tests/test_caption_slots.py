from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from synwts.caption_slots import rerank_caption_slot_variants, rewrite_caption_submission_slots
from synwts.validators import validate_caption_submission


class CaptionSlotsTest(unittest.TestCase):
    def test_rewrites_caption_into_slot_order_with_numeric_phase_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            caption = root / "caption.json"
            vqa_json = root / "vqa.json"
            vqa_submission = root / "vqa_submission.json"
            output = root / "rewritten.json"
            report = root / "report.json"

            caption.write_text(
                json.dumps(
                    {
                        "20230707_11_SY3_T1": [
                            {
                                "labels": ["4"],
                                "caption_pedestrian": (
                                    "The weather was cloudy and the road was dry. "
                                    "The pedestrian's line of sight was fixed on the vehicle. "
                                    "The pedestrian was directly in front of the vehicle."
                                ),
                                "caption_vehicle": (
                                    "The vehicle was going straight ahead. "
                                    "The vehicle was in front of the pedestrian."
                                ),
                            }
                        ]
                    }
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
                                    "labels": ["avoidance"],
                                    "conversations": [
                                        {
                                            "id": "dist",
                                            "question": "What is relative distance of pedestrian from vehicle?",
                                            "a": "Far",
                                            "b": "Close",
                                        },
                                        {
                                            "id": "action",
                                            "question": "What is pedestrian's action?",
                                            "a": "Going straight ahead",
                                            "b": "Collision",
                                        },
                                    ],
                                }
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            vqa_submission.write_text(
                json.dumps(
                    [
                        {"id": "dist", "correct": "b"},
                        {"id": "action", "correct": "a"},
                    ]
                ),
                encoding="utf-8",
            )

            rewritten, stats = rewrite_caption_submission_slots(
                caption=caption,
                output=output,
                wts_vqa_json=vqa_json,
                vqa_submission=vqa_submission,
                report_output=report,
                mode="balanced",
                max_added_sentences=1,
            )

            ped = rewritten["20230707_11_SY3_T1"][0]["caption_pedestrian"]
            self.assertTrue(ped.startswith("The pedestrian was directly"))
            self.assertIn("going straight ahead", ped.lower())
            self.assertEqual(stats["total_rows"], 1)
            self.assertEqual(validate_caption_submission(output)["ok"], True)
            self.assertTrue(report.exists())

    def test_reorder_mode_does_not_add_new_facts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            caption = root / "caption.json"
            vqa_json = root / "vqa.json"
            vqa_submission = root / "vqa_submission.json"
            output = root / "rewritten.json"
            caption.write_text(
                json.dumps(
                    {
                        "s1": [
                            {
                                "labels": ["4"],
                                "caption_pedestrian": "The road was dry. The pedestrian was near the vehicle.",
                                "caption_vehicle": "The vehicle stopped.",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            vqa_json.write_text(json.dumps([]), encoding="utf-8")
            vqa_submission.write_text(json.dumps([]), encoding="utf-8")

            rewritten, stats = rewrite_caption_submission_slots(
                caption=caption,
                output=output,
                wts_vqa_json=vqa_json,
                vqa_submission=vqa_submission,
                mode="reorder",
            )

            ped = rewritten["s1"][0]["caption_pedestrian"]
            self.assertEqual(ped, "The pedestrian was near the vehicle. The road was dry.")
            self.assertEqual(stats["added_sentences"], 0)

    def test_reranker_keeps_base_when_candidate_has_low_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.json"
            bad = root / "bad.json"
            vqa_json = root / "vqa.json"
            vqa_submission = root / "vqa_submission.json"
            output = root / "reranked.json"
            base.write_text(
                json.dumps(
                    {
                        "s1": [
                            {
                                "labels": ["4"],
                                "caption_pedestrian": "The pedestrian was near the vehicle and watched it.",
                                "caption_vehicle": "The vehicle stopped near the pedestrian.",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            bad.write_text(
                json.dumps(
                    {
                        "s1": [
                            {
                                "labels": ["4"],
                                "caption_pedestrian": "Weather road asphalt traffic brightness sidewalk.",
                                "caption_vehicle": "Cloudy residential lane surface dry level.",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            vqa_json.write_text(json.dumps([]), encoding="utf-8")
            vqa_submission.write_text(json.dumps([]), encoding="utf-8")

            reranked, report = rerank_caption_slot_variants(
                base_caption=base,
                candidates={"bad": bad},
                output=output,
                wts_vqa_json=vqa_json,
                vqa_submission=vqa_submission,
                min_switch_margin=0.0,
                min_source_overlap=0.72,
            )

            row = reranked["s1"][0]
            self.assertEqual(row["caption_pedestrian"], "The pedestrian was near the vehicle and watched it.")
            self.assertEqual(report["changed_rows"], 0)


if __name__ == "__main__":
    unittest.main()
