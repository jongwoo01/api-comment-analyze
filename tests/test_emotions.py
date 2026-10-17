"""Small regression set, not a model accuracy benchmark. Uses the pinned model."""
import unittest
from fastapi import HTTPException
from pydantic import ValidationError
from app.main import AnalyzeBatchRequest, LABEL_MAP, _predict_probabilities, _validate_texts

CASES = [
    ("이 영상 너무 좋아요", "joy"),
    ("정말 화나고 짜증나요", "anger"),
    ("설명이 정말 유익하네요 감사합니다", "joy"),
    ("너무 재밌어서 계속 웃었어요", "joy"),
    ("너무 슬퍼서 눈물이 나요", "sadness"),
    ("앞으로 어떻게 될지 걱정돼요", "fear"),
    ("갑자기 이런 말을 하니 당황스럽네요", "embarrassment"),
    ("믿었던 사람에게 배신당해서 마음이 아파요", "hurt"),
]

class EmotionTests(unittest.TestCase):
    def test_korean_labels_and_batch_order(self):
        # Cross the eight-item microbatch boundary to check ordering too.
        cases = CASES + CASES[:2]
        results = _predict_probabilities([text for text, _ in cases])
        self.assertEqual(len(results), len(cases))
        for result, (_, expected) in zip(results, cases):
            self.assertEqual(result['dominant'], expected)
            self.assertEqual(set(result['emotions']), set(LABEL_MAP.values()))
            self.assertAlmostEqual(sum(result['emotions'].values()), 1, places=5)

    def test_invalid_input(self):
        with self.assertRaises(HTTPException):
            _validate_texts(['valid', ' '])
        for texts in ([], ['x'] * 129):
            with self.assertRaises(ValidationError):
                AnalyzeBatchRequest(texts=texts)

if __name__ == '__main__':
    unittest.main()
