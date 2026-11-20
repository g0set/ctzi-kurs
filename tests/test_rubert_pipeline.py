import unittest
from pii_recognizer.train_rubert import split_dev_records
from pii_recognizer.rubert_ner import decode_word_predictions
from pii_recognizer.detector import detect

class RuBERTTests(unittest.TestCase):
    def test_train_validation_disjoint_documents(self):
        rows=[{'id':f'FACTRU-{i}','source':'FactRuEval-2016','text':f'Текст {i}', 'entities':[]} for i in range(60)]
        train,val=split_dev_records(rows,seed=42)
        self.assertEqual(60,len(train)+len(val))
        self.assertFalse({x['id'] for x in train}&{x['id'] for x in val})
        a,b=split_dev_records(rows[::-1],seed=42)
        self.assertEqual([x['id'] for x in train],[x['id'] for x in a])
    def test_no_duplicate_ids(self):
        rows=[{'id':'FACTRU-1','source':'FactRuEval-2016','text':'x','entities':[]} for _ in range(20)]
        with self.assertRaises(ValueError):split_dev_records(rows)
    def test_bio_span_reconstruction(self):
        text='Филипп Стручков встретил Анну.'
        words=[(0,6,'B-NAME'),(7,15,'B-SURNAME'),(16,24,'O'),(25,29,'B-NAME')]
        output=decode_word_predictions(text,words)
        self.assertEqual([(x.label,x.text) for x in output],[('NAME','Филипп'),('SURNAME','Стручков'),('NAME','Анну')])
    def test_rubert_without_weights_explicit_error(self):
        with self.assertRaises(ValueError):detect('Иван Петров',backend='rubert')

if __name__=='__main__':unittest.main()
