
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .detector import recognize
from .metrics import evaluate_records, load_jsonl


_LABEL_NAMES = {
    'NAME': 'Имя',
    'SURNAME': 'Фамилия',
    'INN': 'ИНН',
    'PASSPORT': 'Паспорт',
    'CARD': 'Банк. карта',
    'OGRNIP': 'ОГРНИП',
    'IP': 'IP-адрес',
    'EMAIL': 'Эл. почта',
    'BIRTH_CERT': 'Св-во о рождении',
    'KLADR': 'КЛАДР',
}


def _make_table(headers: list[str], rows: list[list[str]]) -> str:

    values = [headers, *rows]
    widths = [max(len(row[i]) for row in values) for i in range(len(headers))]
    border = '+' + '+'.join('-' * (width + 2) for width in widths) + '+'

    def line(cells: list[str]) -> str:
        return '| ' + ' | '.join(cell.ljust(width) for cell, width in zip(cells, widths)) + ' |'

    return '\n'.join([border, line(headers), border, *(line(row) for row in rows), border])


def print_evaluation_report(result: dict) -> None:

    if not isinstance(result, dict) or not all(
        key in result for key in ('n_texts', 'n_entities', 'token_level', 'entity_exact_span')
    ):
        raise ValueError('JSON не содержит метрик оценки (token_level и entity_exact_span).')

    print('\n' + '=' * 67)
    print('ОТЧЁТ О КАЧЕСТВЕ РАСПОЗНАВАНИЯ — ЦТЗИ / RuBERT')
    print('=' * 67)
    print(f"Документов: {result['n_texts']}    Размеченных сущностей: {result['n_entities']}")

    sections = [
        ('entity_exact_span', '1. ТОЧНОЕ СОВПАДЕНИЕ СУЩНОСТЕЙ (основная оценка)'),
        ('token_level', '2. ОЦЕНКА НА УРОВНЕ ТОКЕНОВ'),
    ]
    headers = ['Атрибут', 'Precision', 'Recall', 'F1', 'TP', 'FP', 'FN', 'Всего']
    for key, title in sections:
        section = result[key]
        rows = []
        for label, scores in section['per_attribute'].items():
            name = _LABEL_NAMES.get(label, label)
            rows.append([
                name,
                f"{scores['precision'] * 100:.2f}%",
                f"{scores['recall'] * 100:.2f}%",
                f"{scores['f1'] * 100:.2f}%",
                str(scores['tp']),
                str(scores['fp']),
                str(scores['fn']),
                str(scores['support']),
            ])

        print('\n' + title)
        print(_make_table(headers, rows))
        print(f"Weighted F1: {section['weighted_f1'] * 100:.2f}%    "
              f"Macro F1: {section['macro_f1'] * 100:.2f}%")

    print('\nTP — верно найдено; FP — лишние срабатывания; FN — пропуски.')
    print('Precision — точность; Recall — полнота; F1 — баланс точности и полноты.')
    print('Exact span: совпадают тип, начало и конец сущности.\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description='ЦТЗИ КМ-2: RuBERT + правила, вариант 18')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--text', help='Одна исходная строка')
    source.add_argument('--file', help='Путь к UTF-8 файлу с исходным текстом')
    source.add_argument('--evaluate', help='Размеченный тестовый набор JSONL')
    source.add_argument('--show-report', metavar='JSON',
                        help='Показать таблицу из готового JSON без повторной оценки модели')
    parser.add_argument('--out', help='Файл JSON с результатом')
    parser.add_argument('--backend', choices=['rubert'], default='rubert',
                        help='RuBERT для NAME/SURNAME')
    parser.add_argument('--model-path', help='Путь к сохранённой модели NER')
    parser.add_argument('--names-only', action='store_true',
                        help='Отдельная оценка FactRuEval по NAME/SURNAME')
    args = parser.parse_args(argv)

    if args.show_report:
        if args.out:
            parser.error('--out нельзя использовать вместе с --show-report')
        result = json.loads(Path(args.show_report).read_text(encoding='utf-8'))
        print_evaluation_report(result)
        return

    if args.evaluate:
        result = evaluate_records(
            load_jsonl(args.evaluate), backend=args.backend,
            model_path=args.model_path,
            labels=('NAME', 'SURNAME') if args.names_only else None,
        )
    else:
        raw = args.text if args.text is not None else Path(args.file).read_text(encoding='utf-8')
        result = recognize(raw, backend=args.backend, model_path=args.model_path)

    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + '\n', encoding='utf-8')
        print('Результат сохранён:', args.out)

    if args.evaluate:
        print_evaluation_report(result)
    elif not args.out:
        print(serialized)


if __name__ == '__main__':
    main()
