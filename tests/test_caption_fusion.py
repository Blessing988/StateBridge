from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from synwts.caption_fusion import fuse_caption_submissions


class CaptionFusionTest(unittest.TestCase):
    def test_selects_fact_consistent_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad = root / "bad.json"
            good = root / "good.json"
            vqa_json = root / "vqa.json"
            vqa_sub = root / "vqa_sub.json"
            output = root / "fused.json"
            report = root / "report.json"

            _write_caption(
                bad,
                "The pedestrian is a male in his 20s. caption_pedestrian artifact.",
                "The vehicle moves at 20 km/h.",
            )
            _write_caption(
                good,
                "The pedestrian is a male in his 30s and watches the vehicle closely.",
                "The vehicle is stopped at 0 km/h near the pedestrian.",
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
                                            "id": "age",
                                            "question": "What is the age group of the pedestrian?",
                                            "a": "20s",
                                            "b": "30s",
                                        },
                                        {
                                            "id": "speed",
                                            "question": "What is vehicle's speed?",
                                            "a": "0 km/h",
                                            "b": "20 km/h",
                                        },
                                    ],
                                }
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            vqa_sub.write_text(
                json.dumps(
                    [
                        {"id": "age", "correct": "b"},
                        {"id": "speed", "correct": "a"},
                    ]
                ),
                encoding="utf-8",
            )

            fused, fusion_report = fuse_caption_submissions(
                submissions={"bad": bad, "good": good},
                fallback_name="bad",
                output=output,
                wts_vqa_json=vqa_json,
                vqa_submission=vqa_sub,
                report_output=report,
            )

            row = fused["20230707_11_SY3_T1"][0]
            self.assertIn("30s", row["caption_pedestrian"])
            self.assertIn("0 km/h", row["caption_vehicle"])
            self.assertEqual(fusion_report["source_counts"], {"good": 1})
            self.assertTrue(output.exists())
            self.assertTrue(report.exists())


def _write_caption(path: Path, ped: str, veh: str) -> None:
    path.write_text(
        json.dumps(
            {
                "20230707_11_SY3_T1": [
                    {
                        "labels": ["4"],
                        "caption_pedestrian": ped,
                        "caption_vehicle": veh,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
