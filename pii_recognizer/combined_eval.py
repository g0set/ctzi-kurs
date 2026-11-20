

from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
from .detector import LABELS
from .factru import write_jsonl
from .metrics import load_jsonl


def validate(records: list[dict], origin: str):
    ids=set()
    for record in records:
        doc_id=record['id']
        if doc_id in ids:
            raise ValueError(f'Повторяющийся id в {origin}: {doc_id}')
        ids.add(doc_id)
        text=record['text']
        for entity in record['entities']:
            label=entity['label']
            if label not in LABELS:
                raise ValueError(f'Неизвестная метка {label!r} в {doc_id}')
            a,b=entity['start'],entity['end']
            if not 0 <= a < b <= len(text):
                raise ValueError(f'Некорректные границы {a}:{b} в {doc_id}')
    return ids


def build_testset(factru_path: Path, extra_path: Path, output_path: Path,
                  min_texts: int=100, min_entities: int=50):
    factru=load_jsonl(str(factru_path))
    extra=load_jsonl(str(extra_path))
    if not factru:
        raise ValueError('В файле FactRuEval нет документов')
    if not extra:
        raise ValueError('В дополнительном наборе нет документов')
    ids_a=validate(factru, 'FactRuEval')
    ids_b=validate(extra, 'дополнительном наборе')
    if ids_a & ids_b:
        raise ValueError('Совпадают id документов между наборами')
    for doc in extra:
        counts={e['label'] for e in doc['entities']}

        doc.setdefault('source','Дополнительный синтетический набор')
    combined=factru+extra
    counts=Counter(e['label'] for doc in combined for e in doc['entities'])
    if len(combined)<min_texts:
        raise ValueError(f'Недостаточно текстов: {len(combined)} < {min_texts}')
    missing={label:counts[label] for label in LABELS if counts[label]<min_entities}
    if missing:
        raise ValueError(f'Недостаточно сущностей для типов: {missing}')
    write_jsonl(combined, output_path)
    print('Объединённый тестовый набор создан:',output_path)
    print('Текстов:',len(combined),'из них FactRuEval:',len(factru),
          'дополнение:',len(extra))
    print('Размеченных сущностей:',dict(counts))
    print('ВАЖНО: FactRuEval не имеет полной разметки восьми форматных типов.')
    print('Для корректной финальной оценки 10 типов рекомендуется отдельно')
    print('проверять полностью размеченное дополнение и FactRuEval по NAME/SURNAME.')
    return combined


def main():
    parser=argparse.ArgumentParser(description='Собрать тестовый корпус 10 ПДн')
    parser.add_argument('--factru',type=Path,default=Path('data/factru_test.jsonl'))
    parser.add_argument('--extra',type=Path,default=Path('data/variant18_extra_test.jsonl'))
    parser.add_argument('--out',type=Path,default=Path('data/variant18_full_test.jsonl'))
    args=parser.parse_args()
    build_testset(args.factru,args.extra,args.out)

if __name__=='__main__':
    main()
