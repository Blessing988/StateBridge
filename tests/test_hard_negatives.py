from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from synwts.hard_negatives import (
    build_hard_negative_preference_dataset,
    select_hard_negative,
)
from synwts.option_scoring import _interleave_prompt_and_videos


class HardNegativeSelectionTest(unittest.TestCase):
    def test_selects_highest_scoring_incorrect_option(self) -> None:
        result = select_hard_negative(
            options={"a": "one", "b": "two", "c": "three", "d": "four"},
            correct="b",
            scores={"a": -3.0, "b": -0.8, "c": -1.0, "d": -2.0},
        )
        self.assertEqual(result["rejected"], "c")
        self.assertEqual(result["predicted"], "b")
        self.assertAlmostEqual(result["gold_margin"], 0.2)

    def test_ties_are_deterministic(self) -> None:
        result = select_hard_negative(
            options={"a": "one", "b": "two", "c": "three"},
            correct="c",
            scores={"a": -1.0, "b": -1.0, "c": -2.0},
        )
        self.assertEqual(result["rejected"], "a")
        self.assertEqual(result["predicted"], "a")
        self.assertTrue(result["is_model_error"])

    def test_requires_complete_finite_scores(self) -> None:
        with self.assertRaisesRegex(ValueError, "Missing scores"):
            select_hard_negative(
                options={"a": "one", "b": "two"},
                correct="a",
                scores={"a": -1.0},
            )


class HardNegativeDatasetTest(unittest.TestCase):
    def test_builds_filtered_balanced_dataset(self) -> None:
        candidates = [
            _candidate("q1", correct="a", question_type="type_a"),
            _candidate("q2", correct="b", question_type="type_a"),
            _candidate("q3", correct="c", question_type="type_b"),
            _candidate("q4", correct="d", question_type="type_b"),
        ]
        scores = [
            _score("q1", a=-0.2, b=-0.4, c=-2.0, d=-3.0),
            _score("q2", a=-0.1, b=-1.0, c=-2.0, d=-3.0),
            _score("q3", a=-4.0, b=-3.0, c=-0.1, d=-3.0),
            _score("q4", a=-3.0, b=-2.0, c=-1.5, d=-0.1),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_path = root / "candidates.json"
            score_path = root / "scores.jsonl"
            output_path = root / "preference.json"
            report_path = root / "report.json"
            info_path = root / "dataset_info.json"
            candidate_path.write_text(json.dumps(candidates), encoding="utf-8")
            score_path.write_text(
                "".join(json.dumps(row) + "\n" for row in scores),
                encoding="utf-8",
            )

            rows, report = build_hard_negative_preference_dataset(
                candidates_path=candidate_path,
                scores_path=score_path,
                output=output_path,
                report_output=report_path,
                dataset_info_output=info_path,
                selection="errors_and_margin",
                max_gold_margin=0.5,
                balance_fields=("question_type",),
                max_rows=2,
            )

            self.assertEqual(len(rows), 2)
            self.assertEqual({row["metadata"]["vqa_id"] for row in rows}, {"q1", "q2"})
            q1 = next(row for row in rows if row["metadata"]["vqa_id"] == "q1")
            self.assertEqual(q1["rejected"], "b")
            self.assertEqual(report["model_errors"], 1)
            self.assertEqual(report["selected"], 2)
            self.assertTrue(output_path.exists())
            self.assertTrue(report_path.exists())
            self.assertTrue(info_path.exists())

    def test_missing_scores_are_strict_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_path = root / "candidates.json"
            score_path = root / "scores.jsonl"
            candidate_path.write_text(json.dumps([_candidate("missing")]), encoding="utf-8")
            score_path.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Incomplete or invalid"):
                build_hard_negative_preference_dataset(
                    candidates_path=candidate_path,
                    scores_path=score_path,
                    output=root / "out.json",
                )

    def test_can_balance_by_correct_and_rejected_letters(self) -> None:
        candidates = [
            _candidate("q1", correct="b"),
            _candidate("q2", correct="b"),
            _candidate("q3", correct="b"),
            _candidate("q4", correct="c"),
        ]
        scores = [
            _score("q1", a=-0.2, b=-0.1, c=-3.0, d=-4.0),
            _score("q2", a=-0.3, b=-0.1, c=-3.0, d=-4.0),
            _score("q3", a=-0.4, b=-0.1, c=-3.0, d=-4.0),
            _score("q4", a=-3.0, b=-0.2, c=-0.1, d=-4.0),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_path = root / "candidates.json"
            score_path = root / "scores.jsonl"
            candidate_path.write_text(json.dumps(candidates), encoding="utf-8")
            score_path.write_text(
                "".join(json.dumps(row) + "\n" for row in scores),
                encoding="utf-8",
            )

            rows, report = build_hard_negative_preference_dataset(
                candidates_path=candidate_path,
                scores_path=score_path,
                output=root / "preference.json",
                selection="margin",
                max_gold_margin=1.0,
                balance_fields=("correct", "rejected"),
                max_per_group=1,
            )

            self.assertEqual(len(rows), 2)
            self.assertEqual(
                {(row["chosen"], row["rejected"]) for row in rows},
                {("b", "a"), ("c", "b")},
            )
            self.assertEqual(report["selected_distribution"]["rejected"], {"a": 1, "b": 1})

    def test_can_remap_option_letters_to_remove_response_prior(self) -> None:
        candidates = [
            _candidate("q1", correct="b"),
            _candidate("q2", correct="b"),
            _candidate("q3", correct="b"),
            _candidate("q4", correct="b"),
        ]
        scores = [
            _score("q1", a=-0.2, b=-0.1, c=-3.0, d=-4.0),
            _score("q2", a=-0.2, b=-0.1, c=-3.0, d=-4.0),
            _score("q3", a=-0.2, b=-0.1, c=-3.0, d=-4.0),
            _score("q4", a=-0.2, b=-0.1, c=-3.0, d=-4.0),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_path = root / "candidates.json"
            score_path = root / "scores.jsonl"
            candidate_path.write_text(json.dumps(candidates), encoding="utf-8")
            score_path.write_text(
                "".join(json.dumps(row) + "\n" for row in scores),
                encoding="utf-8",
            )

            rows, report = build_hard_negative_preference_dataset(
                candidates_path=candidate_path,
                scores_path=score_path,
                output=root / "preference.json",
                selection="margin",
                max_gold_margin=1.0,
                balance_fields=(),
                remap_option_letters=True,
            )

            self.assertEqual(len(rows), 4)
            self.assertEqual(
                [(row["chosen"], row["rejected"]) for row in rows],
                [("a", "b"), ("a", "c"), ("a", "d"), ("b", "a")],
            )
            self.assertIn("a. two", rows[0]["instruction"])
            self.assertIn("b. one", rows[0]["instruction"])
            self.assertEqual(rows[0]["metadata"]["original_correct"], "b")
            self.assertEqual(rows[0]["metadata"]["original_rejected"], "a")
            self.assertTrue(report["remap_option_letters"])


class PromptInterleaveTest(unittest.TestCase):
    def test_preserves_video_placeholder_order(self) -> None:
        content = _interleave_prompt_and_videos(
            instruction="before\n<video>\nmiddle\n<video>\nafter",
            videos=["/tmp/one.mp4", "/tmp/two.mp4"],
            video_max_pixels=65536,
            fps=2.0,
        )
        self.assertEqual([item["type"] for item in content], ["text", "video", "text", "video", "text"])
        self.assertEqual(content[1]["max_pixels"], 65536)
        self.assertEqual(content[1]["fps"], 2.0)

    def test_rejects_media_count_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "placeholders"):
            _interleave_prompt_and_videos(
                instruction="<video><video>",
                videos=["/tmp/one.mp4"],
                video_max_pixels=None,
                fps=None,
            )


def _candidate(
    qid: str,
    *,
    correct: str = "a",
    question_type: str = "type_a",
) -> dict:
    return {
        "instruction": (
            "Question\n<video>\n\n"
            "Options:\n"
            "a. one\n"
            "b. two\n"
            "c. three\n"
            "d. four\n\n"
            "Return only the correct option letter."
        ),
        "input": "",
        "options": {"a": "one", "b": "two", "c": "three", "d": "four"},
        "correct": correct,
        "videos": ["/tmp/video.mp4"],
        "metadata": {
            "vqa_id": qid,
            "question_type": question_type,
            "scope": "overhead_view",
            "phase": "2",
            "scenario_type": "event",
        },
    }


def _score(qid: str, **scores: float) -> dict:
    return {"vqa_id": qid, "scores": scores, "model_name_or_path": "model"}


if __name__ == "__main__":
    unittest.main()
