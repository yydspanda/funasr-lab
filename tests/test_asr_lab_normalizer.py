import unittest

from eval.normalizers import NORMALIZER_VERSION
from eval.normalizers import normalize_content


class ContentNormalizerTest(unittest.TestCase):
    def test_normalizes_width_case_whitespace_and_punctuation(self):
        self.assertEqual(normalize_content(" 你好，World！１２３ "), "你好world123")

    def test_preserves_lexical_variants_and_symbols(self):
        self.assertEqual(normalize_content("臺灣+AI🙂"), "臺灣+ai🙂")

    def test_is_idempotent_and_versioned(self):
        text = normalize_content("语音，识别。")
        self.assertEqual(normalize_content(text), text)
        self.assertEqual(NORMALIZER_VERSION, "zh-content-v0.1")


if __name__ == "__main__":
    unittest.main()
