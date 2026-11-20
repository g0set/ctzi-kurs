

from __future__ import annotations
from collections import Counter
import json
from .detector import LABELS, Entity, bio_tags, detect


def class_of(tag: str) -> str:
    return 'O' if tag == 'O' else tag.split('-', 1)[1]


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp/(tp+fp) if tp+fp else 0.0
    r = tp/(tp+fn) if tp+fn else 0.0
    f = 2*p*r/(p+r) if p+r else 0.0
    return {'precision': round(p, 6), 'recall': round(r, 6), 'f1': round(f, 6),
            'tp': tp, 'fp': fp, 'fn': fn, 'support': tp+fn}


def summarize(rows: list[dict]):
    supports = sum(row['support'] for row in rows)
    weighted = sum(row['support']*row['f1'] for row in rows)/supports if supports else 0.
    macro = sum(row['f1'] for row in rows)/len(rows) if rows else 0.
    return {'weighted_f1': round(weighted, 6), 'macro_f1': round(macro, 6), 'support': supports}


def evaluate_records(records: list[dict], backend: str = "rubert", model_path=None, labels=None) -> dict:
    labels = tuple(labels) if labels is not None else LABELS
    tok_counts = {label: Counter() for label in labels}
    ent_counts = {label: Counter() for label in labels}
    examples = []
    for rec in records:
        text = rec['text']
        gold_ents = [Entity(e['label'], e['start'], e['end'], text[e['start']:e['end']], 'gold') for e in rec['entities']]
        pred_ents = detect(text, backend=backend, model_path=model_path)
        if labels != LABELS:
            gold_ents = [e for e in gold_ents if e.label in labels]
            pred_ents = [e for e in pred_ents if e.label in labels]
        gold_toks = bio_tags(text, gold_ents)
        pred_toks = bio_tags(text, pred_ents)
        assert len(gold_toks) == len(pred_toks)
        for gold, pred in zip(gold_toks, pred_toks):
            assert (gold['start'],gold['end']) == (pred['start'],pred['end'])
            g, p = class_of(gold['tag']), class_of(pred['tag'])
            for label in labels:
                if g == label and p == label:
                    tok_counts[label]['tp']+=1
                elif p == label and g != label:
                    tok_counts[label]['fp']+=1
                elif g == label and p != label:
                    tok_counts[label]['fn']+=1
        gs = {(e.label,e.start,e.end) for e in gold_ents}
        ps = {(e.label,e.start,e.end) for e in pred_ents}
        for label in labels:
            ent_counts[label]['tp'] += len([e for e in gs & ps if e[0] == label])
            ent_counts[label]['fp'] += len([e for e in ps-gs if e[0] == label])
            ent_counts[label]['fn'] += len([e for e in gs-ps if e[0] == label])
        if gs != ps and len(examples) < 12:
            examples.append({'id': rec['id'], 'gold': sorted(map(list, gs)), 'pred': sorted(map(list, ps))})
    trows = {label: prf(**{k: tok_counts[label][k] for k in ('tp','fp','fn')}) for label in labels}
    erows = {label: prf(**{k: ent_counts[label][k] for k in ('tp','fp','fn')}) for label in labels}
    return {'n_texts': len(records), 'n_entities': sum(len(r['entities']) for r in records),
            'token_level': {'per_attribute': trows, **summarize(list(trows.values()))},
            'entity_exact_span': {'per_attribute': erows, **summarize(list(erows.values()))},
            'mismatch_examples': examples}


def load_jsonl(path: str) -> list[dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]
