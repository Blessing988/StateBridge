import json

from synwts.exporters import export_llamafactory
from synwts.preferences import build_vqa_option_candidates
from synwts.schema import ScenarioRecord


def _write_vqa(path):
    path.write_text(
        json.dumps(
            {
                "event_phase": [
                    {
                        "labels": ["4"],
                        "conversations": [
                            {
                                "question": "What does the pedestrian do?",
                                "a": "wait",
                                "b": "cross",
                                "correct": "b",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def _write_manifest(path):
    rows = [
        {
            "scenario_id": "scenario-1",
            "view": "overhead_view",
            "phase": "4",
            "clip_path": f"phase-{index}.mp4",
        }
        for index in range(4)
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_media_policy_first_applies_after_phase_clip_lookup(tmp_path):
    vqa_path = tmp_path / "vqa.json"
    manifest_path = tmp_path / "clips.jsonl"
    output_path = tmp_path / "output.json"
    _write_vqa(vqa_path)
    _write_manifest(manifest_path)
    record = ScenarioRecord(
        split="train",
        scenario_id="scenario-1",
        scenario_type="event",
        videos={"overhead_view": ["full-0.mp4", "full-1.mp4"]},
        vqa_files={"overhead_view": str(vqa_path)},
    )

    rows = export_llamafactory(
        [record],
        output_path,
        tasks={"vqa"},
        media_policy="first",
        phase_clip_manifest=manifest_path,
    )

    assert rows[0]["videos"] == ["phase-0.mp4"]
    assert rows[0]["instruction"].count("<video>") == 1


def test_preference_candidates_apply_media_policy_after_phase_clip_lookup(tmp_path):
    vqa_path = tmp_path / "vqa.json"
    manifest_path = tmp_path / "clips.jsonl"
    _write_vqa(vqa_path)
    _write_manifest(manifest_path)
    record = ScenarioRecord(
        split="train",
        scenario_id="scenario-1",
        scenario_type="event",
        videos={"overhead_view": ["full-0.mp4", "full-1.mp4"]},
        vqa_files={"overhead_view": str(vqa_path)},
    )

    rows = build_vqa_option_candidates(
        [record],
        media_policy="first",
        phase_clip_manifest=manifest_path,
    )

    assert rows[0]["videos"] == ["phase-0.mp4"]
    assert rows[0]["instruction"].count("<video>") == 1
