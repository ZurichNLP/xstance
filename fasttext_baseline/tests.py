import os
from unittest import TestCase

import jsonlines
import numpy as np

from fasttext_baseline import run as run_baseline


class FastTextBaselineTestCase(TestCase):

    def setUp(self) -> None:
        self.data_dir = "../data"
        self.model_path = "test_model.bin"
        input_path = os.path.join(self.data_dir, "valid.jsonl")
        self.valid_path = os.path.join("test_output", "valid.txt")
        run_baseline._jsonl_to_fasttext_format(input_path, self.valid_path)

    def test_jsonl_to_fasttext_format(self):
        with open(self.valid_path) as f:
            first_line = next(f)
        label, *tokens = first_line.split()
        self.assertEqual("__label__FAVOR", label)
        self.assertEqual(
            """\
Sollen Ausländer / -innen , die seit mindestens zehn Jahren \
in der Schweiz leben , das Stimm- und Wahlrecht auf Gemeindeebene \
erhalten ? Ich bin finde das geht zu wenig weit . Alle \
Menschen die hier leben sollen das Recht auf Mitsprache haben .""",
            first_line.replace(label, "").strip()
        )
        num_lines = 0
        with open(self.valid_path) as f:
            for line in f:
                if line.strip():
                    num_lines += 1
        self.assertEqual(3926, num_lines)

    def test_predictions_to_jsonl(self):
        predictions = (((u'__label__AGAINST',), np.array([0.15613931]),),)
        predictions_path = os.path.join("test_output", "pred.jsonl")
        run_baseline._predictions_to_jsonl(predictions, predictions_path)
        with jsonlines.open(predictions_path) as f:
            first_line = next(iter(f))
        self.assertDictEqual({"label": "AGAINST"}, first_line)


    def test_train(self):
        run_baseline.train(self.model_path, self.valid_path)

    def test_predict(self):
        predictions = run_baseline.predict(self.model_path, self.valid_path)
        self.assertEqual(3926, len(predictions))
