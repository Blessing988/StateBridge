from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from synwts.caption_retrieval import (
    augment_caption_dataset_with_retrieval,
    retrieve_caption_examples,
)


class CaptionRetrievalTest(unittest.TestCase):
    def test_retrieves_matching_phase_and_view(self) -> None:
        row = _caption_row(phase="4", view="overhead_view")
        bank = [
            _style("wrong_phase", phase="2", view="overhead_view"),
            _style("right", phase="4", view="overhead_view"),
            _style("wrong_view", phase="4", view="vehicle_view"),
        ]

        examples = retrieve_caption_examples(row, bank, k=2)

        self.assertEqual(examples[0]["style_id"], "right")
        self.assertEqual([example["style_id"] for example in examples], ["right", "wrong_view"])

    def test_augment_injects_examples_before_return_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path = root / "dataset.json"
            bank_path = root / "bank.jsonl"
            output_path = root / "augmented.json"
            info_path = root / "dataset_info.json"
            dataset_path.write_text(json.dumps([_caption_row()]), encoding="utf-8")
            bank_path.write_text(json.dumps(_style("example")) + "\n", encoding="utf-8")

            rows = augment_caption_dataset_with_retrieval(
                dataset_path,
                bank_path,
                output_path,
                k=1,
                dataset_info_output=info_path,
                dataset_name="caption_retrieval",
            )

            instruction = rows[0]["instruction"]
            self.assertIn("Retrieved synthetic annotation-style examples", instruction)
            self.assertLess(
                instruction.index("Retrieved synthetic annotation-style examples"),
                instruction.index("Return JSON with exactly these keys:"),
            )
            self.assertEqual(rows[0]["metadata"]["retrieval_style_ids"], ["example"])
            self.assertTrue(output_path.exists())
            self.assertTrue(info_path.exists())


def _caption_row(*, phase: str = "4", view: str = "overhead_view") -> dict:
    return {
        "instruction": (
            "Scenario type: event\n"
            f"View: {view}\n"
            f"Phase label: {phase} (avoidance)\n\n"
            "<video>\n\n"
            "BBox context: coordinates are [x1, y1, x2, y2] normalized to a 0-1000 full-frame grid.\n"
            "- pedestrian overhead_view ped_bbox: first=[100, 100, 150, 220] "
            "mid=[105, 105, 155, 225] last=[110, 110, 160, 230] mean=[105, 105, 155, 225] frames=10\n"
            "- vehicle overhead_view veh_bbox: first=[500, 500, 700, 700] "
            "mid=[500, 500, 700, 700] last=[500, 500, 700, 700] mean=[500, 500, 700, 700] frames=10\n\n"
            "Return JSON with exactly these keys:"
        ),
        "input": "",
        "output": "{}",
        "videos": ["/tmp/video.mp4"],
        "metadata": {
            "task": "caption",
            "scenario_id": "query_scenario",
            "scenario_type": "event",
            "view": view,
            "phase": phase,
        },
    }


def _style(style_id: str, *, phase: str = "4", view: str = "overhead_view") -> dict:
    return {
        "style_id": style_id,
        "split": "train",
        "scenario_id": f"scenario_{style_id}",
        "scenario_type": "event",
        "view": view,
        "phase": phase,
        "phase_name": "avoidance",
        "caption_pedestrian": "The pedestrian slows and watches the vehicle while crossing.",
        "caption_vehicle": "The vehicle brakes and yields near the pedestrian.",
        "bbox_features": {
            "pedestrian": {"cx": 130.0, "cy": 165.0, "area": 6000.0},
            "vehicle": {"cx": 600.0, "cy": 600.0, "area": 40000.0},
        },
    }


if __name__ == "__main__":
    unittest.main()
