import collections
import contextlib
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from pii_recognizer.detector import LABELS
from pii_recognizer.metrics import load_jsonl, evaluate_records
from pii_recognizer.cli import print_evaluation_report
from pii_recognizer.combined_eval import build_testset
from pii_recognizer.generate_data import dataset

TEST=Path('data/variant18_extra_test.jsonl')

class RestoredVariant18Tests(unittest.TestCase):
    def test_all_ten_types_and_minimum_support(self):
        recs=load_jsonl(str(TEST))
        self.assertEqual(240,len(recs))
        counts=collections.Counter(e['label'] for r in recs for e in r['entities'])
        self.assertEqual(set(LABELS),set(counts))
        self.assertTrue(all(counts[label]>=50 for label in LABELS))
        self.assertTrue(all(counts[label]==120 for label in LABELS))

    def test_reproducible_dataset(self):
        rows=dataset(seed=20261119,n_positive=200,n_negative=40,prefix='TEST')
        current=load_jsonl(str(TEST))
        self.assertEqual([(r['id'],r['text'],r['entities']) for r in rows],
                         [(r['id'],r['text'],r['entities']) for r in current])

    def test_ten_rows_in_printed_metrics(self):
        rows=load_jsonl(str(TEST))[:1]
        from pii_recognizer.detector import Entity
        def gold_stub(text, *args, **kwargs):
            return [Entity(e['label'],e['start'],e['end'],text[e['start']:e['end']],'test') for e in rows[0]['entities']]
        with patch('pii_recognizer.metrics.detect',side_effect=gold_stub):
            result=evaluate_records(rows)
        out=StringIO()
        with contextlib.redirect_stdout(out):print_evaluation_report(result)
        table=out.getvalue()
        for label in ['ИНН','Паспорт','ОГРНИП','IP-адрес','Эл. почта','КЛАДР','Имя','Фамилия','Банк. карта','Св-во о рождении']:
            self.assertIn(label,table)
        self.assertEqual(10,len(result['token_level']['per_attribute']))

    def test_combined_corpus(self):
        extra=load_jsonl(str(TEST))
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'factru.jsonl',Path(tmp)/'joined.jsonl'
            a.write_text('{"id":"FACTRU-MOCK","text":"Иван Петров","entities":[{"label":"NAME","start":0,"end":4},{"label":"SURNAME","start":5,"end":11}]}\n',encoding='utf-8')
            with contextlib.redirect_stdout(StringIO()):
                joined=build_testset(a,TEST,b)
            self.assertEqual(241,len(joined))
            self.assertTrue(b.exists())

if __name__=='__main__':unittest.main()
