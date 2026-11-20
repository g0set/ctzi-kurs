import contextlib
from io import StringIO
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from pii_recognizer.cli import main, print_evaluation_report


class ConsoleReportTests(unittest.TestCase):
    @staticmethod
    def example():
        return {
            'n_texts': 2,
            'n_entities': 3,
            'entity_exact_span': {
                'per_attribute': {
                    'NAME': {'precision': 1, 'recall': 0.5, 'f1': 2 / 3, 'tp': 1, 'fp': 0, 'fn': 1, 'support': 2},
                    'SURNAME': {'precision': 1, 'recall': 1, 'f1': 1, 'tp': 1, 'fp': 0, 'fn': 0, 'support': 1},
                },
                'weighted_f1': 0.777777,
                'macro_f1': 0.833333,
                'support': 3,
            },
            'token_level': {
                'per_attribute': {'NAME': {'precision': 1, 'recall': 1, 'f1': 1, 'tp': 2, 'fp': 0, 'fn': 0, 'support': 2}},
                'weighted_f1': 1,
                'macro_f1': 1,
                'support': 2,
            },
        }

    def test_table_shows_both_sections_and_scores(self):
        stream = StringIO()
        with contextlib.redirect_stdout(stream):
            print_evaluation_report(self.example())
        output = stream.getvalue()
        for snippet in ['Фамилия', 'Имя', 'Precision', 'Recall', '66.67%', '77.78%', '83.33%', 'ТОКЕНОВ']:
            self.assertIn(snippet, output)

    def test_show_report_does_not_invoke_evaluation(self):
        with TemporaryDirectory() as folder:
            source = Path(folder) / 'result.json'
            source.write_text(json.dumps(self.example()), encoding='utf-8')
            stream = StringIO()
            with patch('pii_recognizer.cli.evaluate_records', side_effect=AssertionError('Модель запущена!')):
                with contextlib.redirect_stdout(stream):
                    main(['--show-report', str(source)])
            self.assertIn('ОТЧЁТ', stream.getvalue())

    def test_evaluate_saves_json_and_prints_report(self):
        with TemporaryDirectory() as folder:
            output = Path(folder) / 'output.json'
            stream = StringIO()
            with patch('pii_recognizer.cli.load_jsonl', return_value=[]), \
                 patch('pii_recognizer.cli.evaluate_records', return_value=self.example()):
                with contextlib.redirect_stdout(stream):
                    main(['--evaluate', 'dataset.jsonl', '--out', str(output)])
            self.assertTrue(output.is_file())
            self.assertIn('Результат сохранён', stream.getvalue())
            self.assertIn('Weighted F1', stream.getvalue())


if __name__ == '__main__':
    unittest.main()
