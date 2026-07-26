import json
import tempfile
import unittest
from pathlib import Path

from synwts.vqa_consistency import repair_vqa_scenario_consistency


class VqaConsistencyTest(unittest.TestCase):
    def test_repairs_repeated_stable_scenario_fact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidates = root / "candidates.json"
            submission = root / "submission.json"
            output = root / "repaired.json"
            report = root / "report.json"

            _write_json(
                candidates,
                [
                    _candidate("q1", "s1", "What is the age group of the pedestrian?"),
                    _candidate("q2", "s1", "What is the age group of the pedestrian?"),
                    _candidate("q3", "s1", "What is the age group of the pedestrian?"),
                    _candidate("q4", "s1", "What is the age group of the pedestrian?"),
                ],
            )
            _write_json(
                submission,
                [
                    {"id": "q1", "correct": "b"},
                    {"id": "q2", "correct": "b"},
                    {"id": "q3", "correct": "b"},
                    {"id": "q4", "correct": "d"},
                ],
            )

            rows, details = repair_vqa_scenario_consistency(
                candidates_path=candidates,
                submission=submission,
                output=output,
                report_output=report,
            )

            self.assertEqual(rows[-1], {"id": "q4", "correct": "b"})
            self.assertEqual(details["changed"], 1)
            self.assertEqual(details["applied_groups"], 1)
            self.assertEqual(_read_json(output)[-1]["correct"], "b")
            self.assertEqual(_read_json(report)["changed"], 1)

    def test_dynamic_phase_questions_are_not_repaired(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidates = root / "candidates.json"
            submission = root / "submission.json"
            output = root / "repaired.json"

            question = "What is the position of the pedestrian relative to the vehicle?"
            _write_json(
                candidates,
                [
                    _candidate("q1", "s1", question, scope="phase"),
                    _candidate("q2", "s1", question, scope="phase"),
                    _candidate("q3", "s1", question, scope="phase"),
                ],
            )
            _write_json(
                submission,
                [
                    {"id": "q1", "correct": "a"},
                    {"id": "q2", "correct": "a"},
                    {"id": "q3", "correct": "c"},
                ],
            )

            rows, details = repair_vqa_scenario_consistency(
                candidates_path=candidates,
                submission=submission,
                output=output,
            )

            self.assertEqual(rows[-1], {"id": "q3", "correct": "c"})
            self.assertEqual(details["changed"], 0)

    def test_threshold_blocks_weak_majority(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidates = root / "candidates.json"
            submission = root / "submission.json"
            output = root / "repaired.json"

            _write_json(
                candidates,
                [
                    _candidate("q1", "s1", "What is the gender of the pedestrian?"),
                    _candidate("q2", "s1", "What is the gender of the pedestrian?"),
                    _candidate("q3", "s1", "What is the gender of the pedestrian?"),
                ],
            )
            _write_json(
                submission,
                [
                    {"id": "q1", "correct": "a"},
                    {"id": "q2", "correct": "a"},
                    {"id": "q3", "correct": "b"},
                ],
            )

            rows, details = repair_vqa_scenario_consistency(
                candidates_path=candidates,
                submission=submission,
                output=output,
                min_top_share=0.75,
            )

            self.assertEqual(rows[-1], {"id": "q3", "correct": "b"})
            self.assertEqual(details["changed"], 0)


def _candidate(qid: str, scenario_id: str, question: str, *, scope: str = "environment") -> dict:
    return {
        "options": {"a": "20s", "b": "30s", "c": "40s", "d": "70s"},
        "metadata": {
            "vqa_id": qid,
            "scenario_id": scenario_id,
            "question": question,
            "scope": scope,
        },
    }


def _write_json(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle)


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    unittest.main()
