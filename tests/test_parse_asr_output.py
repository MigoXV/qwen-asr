from __future__ import annotations

import unittest

from qwen_asr.inferencers.text.asr_output import parse_asr_output


class ParseAsrOutputTest(unittest.TestCase):
    def test_forced_language_strips_echoed_prompt_prefix(self):
        language, text = parse_asr_output(
            "language Chinese<asr_text>你好", user_language="Chinese"
        )
        self.assertEqual(language, "Chinese")
        self.assertEqual(text, "你好")

    def test_forced_language_keeps_plain_text_output(self):
        language, text = parse_asr_output("hello world", user_language="English")
        self.assertEqual(language, "English")
        self.assertEqual(text, "hello world")


if __name__ == "__main__":
    unittest.main()
