from __future__ import annotations

import unittest

from synwts.submission import _parse_caption_prediction


class CaptionParsingTest(unittest.TestCase):
    def test_strips_json_key_fragments_from_partial_caption_prediction(self) -> None:
        parsed = _parse_caption_prediction(
            '"caption_pedestrian": "The pedestrian is crossing.",\n'
            '"caption_vehicle": "The vehicle slows down."',
            "4",
        )

        self.assertEqual(parsed["caption_pedestrian"], "The pedestrian is crossing.")
        self.assertEqual(parsed["caption_vehicle"], "The vehicle slows down.")


if __name__ == "__main__":
    unittest.main()
