import itertools
import random
import unittest

from qwen_asr.inferencers.text.asr_output import detect_and_fix_repetitions
from tests.repetition_reference import detect_and_fix_repetitions as reference


class RepetitionCleanupTest(unittest.TestCase):
    def test_exhaustive_small_strings_and_threshold_boundaries(self):
        for threshold in (1,2,3,20):
            for length in range(9):
                for chars in itertools.product('a\n',repeat=length):
                    text=''.join(chars)
                    self.assertEqual(detect_and_fix_repetitions(text,threshold),reference(text,threshold))
        for count in (19,20,21,39,40,41):
            for prefix in ('','abc','language Chinese<asr_text>'):
                for suffix in ('','xyz','z'*40):
                    for pattern in ('a','ab','你好','a\nb','x'*19+'y'):
                        text=prefix+pattern*count+suffix
                        self.assertEqual(detect_and_fix_repetitions(text),reference(text))

    def test_random_unicode_repetition_and_streaming_prefixes(self):
        rng=random.Random(731)
        for _ in range(250):
            pattern=''.join(rng.choices('a你。\n xy',k=rng.randint(1,20)))
            text=''.join(rng.choices('abc你好\n',k=rng.randint(0,30)))+pattern*rng.randint(1,25)+'尾巴'
            for end in range(1,len(text)+1,7):
                self.assertEqual(detect_and_fix_repetitions(text[:end]),reference(text[:end]))
