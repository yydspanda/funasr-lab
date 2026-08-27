import unittest

from eval.normalizers import NORMALIZER_VERSION
from eval.offline_baseline import EditCounts as BaselineEditCounts
from eval.offline_baseline import cer_components as baseline_cer_components
from eval.scoring import ALIGNMENT_VERSION
from eval.scoring import MER_TOKENIZER_VERSION
from eval.scoring import EditCounts
from eval.scoring import cer_components
from eval.scoring import cer_score
from eval.scoring import cer_units
from eval.scoring import mer_components
from eval.scoring import mer_score
from eval.scoring import mixed_units
from eval.scoring import sequence_edit_counts


class ScoringTest(unittest.TestCase):
    def test_offline_baseline_reexports_the_shared_scoring_api(self):
        self.assertIs(BaselineEditCounts, EditCounts)
        self.assertIs(baseline_cer_components, cer_components)

    def test_sequence_components_cover_substitution_deletion_and_insertion(self):
        self.assertEqual(
            sequence_edit_counts(("甲", "乙", "丙"), ("甲", "丁", "丙")),
            EditCounts(substitutions=1),
        )
        self.assertEqual(
            sequence_edit_counts(("甲", "乙", "丙"), ("甲", "丙")),
            EditCounts(deletions=1),
        )
        self.assertEqual(
            sequence_edit_counts(("甲", "丙"), ("甲", "乙", "丙")),
            EditCounts(insertions=1),
        )

    def test_tied_alignments_use_stable_diagonal_first_accounting(self):
        self.assertEqual(
            ALIGNMENT_VERSION,
            "levenshtein-diagonal-deletion-insertion-v1",
        )
        expected = EditCounts(substitutions=2)

        for _ in range(20):
            self.assertEqual(sequence_edit_counts("ab", "ba"), expected)
        self.assertEqual(
            sequence_edit_counts("abab", "baaba"),
            EditCounts(deletions=1, insertions=2),
        )

    def test_cer_uses_the_frozen_content_normalizer(self):
        result = cer_score(" 你 好，ＷＯＲＬＤ！", "你号 world。")

        self.assertEqual(NORMALIZER_VERSION, "zh-content-v0.1")
        self.assertEqual(cer_units(" 你 好，ＷＯＲＬＤ！"), tuple("你好world"))
        self.assertEqual(result.counts, EditCounts(substitutions=1))
        self.assertEqual(result.reference_units, 7)
        self.assertEqual(result.error_rate, 1 / 7)

    def test_cer_components_remain_compatible_for_normalized_content(self):
        reference = "测试"
        hypothesis = "测验"

        self.assertEqual(
            cer_components(reference, hypothesis), EditCounts(substitutions=1)
        )
        self.assertEqual(
            baseline_cer_components(reference, hypothesis),
            cer_components(reference, hypothesis),
        )

    def test_mixed_units_freeze_code_switch_boundaries(self):
        self.assertEqual(MER_TOKENIZER_VERSION, "zh-en-mixed-v0.1")
        self.assertEqual(
            mixed_units("今天 OpenAI2026 发布"),
            ("今", "天", "openai2026", "发", "布"),
        )
        self.assertEqual(
            mer_components("你好 OpenAI 2026", "你好 openai 2027"),
            EditCounts(substitutions=1),
        )

    def test_mer_reports_hand_calculated_mixed_substitution_and_insertion(self):
        result = mer_score("中 hello 文", "中 hallo new 文")

        self.assertEqual(result.counts, EditCounts(substitutions=1, insertions=1))
        self.assertEqual(result.reference_units, 3)
        self.assertEqual(result.error_rate, 2 / 3)

    def test_hand_calculated_cer_and_mer_fixtures_freeze_components(self):
        self.assertEqual(
            cer_score("今天下雨", "今天大雨").counts,
            EditCounts(substitutions=1),
        )
        self.assertEqual(
            cer_score("语音识别", "语音别").counts,
            EditCounts(deletions=1),
        )
        self.assertEqual(
            cer_score("你好", "你们好").counts,
            EditCounts(insertions=1),
        )
        self.assertEqual(
            cer_score("ＡＩ，测试！", "ai测试").counts,
            EditCounts(),
        )

        mixed_cases = (
            ("你好 world", "你号 word", EditCounts(substitutions=2), 3),
            ("今天 weather 好", "今天好", EditCounts(deletions=1), 4),
            ("请打开 app", "请打开 new app", EditCounts(insertions=1), 4),
            ("你好，OpenAI！", "你 好 openai", EditCounts(), 3),
            (
                "hello world",
                "helloworld",
                EditCounts(substitutions=1, deletions=1),
                2,
            ),
        )
        for reference, hypothesis, counts, denominator in mixed_cases:
            with self.subTest(reference=reference, hypothesis=hypothesis):
                result = mer_score(reference, hypothesis)
                self.assertEqual(result.counts, counts)
                self.assertEqual(result.reference_units, denominator)

    def test_mer_nfkc_casefold_punctuation_and_other_symbol_contract(self):
        self.assertEqual(
            mixed_units("你好，ＦｕｎＡＳＲ！Straße １２３🙂+α"),
            ("你", "好", "funasr", "strasse", "123", "🙂", "+", "α"),
        )
        self.assertEqual(mixed_units("foo_bar C++"), ("foo", "bar", "c", "+", "+"))
        self.assertEqual(mer_score("Straße", "STRASSE").error_rate, 0.0)

    def test_empty_references_keep_components_without_inventing_a_rate(self):
        empty = cer_score("", "")
        cer_insertions = cer_score("， ", "啊")
        mer_insertions = mer_score("！", "hello world")

        self.assertEqual(empty.counts, EditCounts())
        self.assertEqual(empty.reference_units, 0)
        self.assertIsNone(empty.error_rate)
        self.assertEqual(cer_insertions.counts, EditCounts(insertions=1))
        self.assertEqual(cer_insertions.reference_units, 0)
        self.assertIsNone(cer_insertions.error_rate)
        self.assertEqual(mer_insertions.counts, EditCounts(insertions=2))
        self.assertEqual(mer_insertions.reference_units, 0)
        self.assertIsNone(mer_insertions.error_rate)


if __name__ == "__main__":
    unittest.main()
