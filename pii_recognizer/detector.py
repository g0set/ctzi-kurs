

from __future__ import annotations
from dataclasses import dataclass, asdict
import re
import ipaddress
from .validators import inn_valid, ogrnip_valid, luhn_valid, ip_valid

LABELS = ('SURNAME', 'NAME', 'INN', 'PASSPORT', 'CARD', 'OGRNIP', 'IP', 'EMAIL', 'BIRTH_CERT', 'KLADR')

@dataclass(frozen=True)
class Entity:
    label: str
    start: int
    end: int
    text: str
    source: str
    def to_dict(self): return asdict(self)

EMAIL_RE = re.compile(r'(?<![\w.@+\-])(?:[A-Za-z0-9_%+\-]+(?:\.[A-Za-z0-9_%+\-]+)*)@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)+\b')
IPV4_RE = re.compile(r'(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w]|\.\d)')
IPV6_RE = re.compile(r'(?<![\w:])(?:[0-9A-Fa-f]{1,4}:){2,7}(?:[0-9A-Fa-f]{1,4})?(?![\w:])')
PASSPORT_RE = re.compile(r'(?<!\d)\d{2}\s?\d{2}[\s-]?\d{6}(?!\d)')
CARD_RE = re.compile(r'(?<!\d)(?:\d{4}[ -]){3}\d{4}(?!\d)|(?<!\d)\d{13,19}(?!\d)')
OGRNIP_RE = re.compile(r'(?<!\d)\d{15}(?!\d)')
INN_RE = re.compile(r'(?<!\d)(?:\d{12}|\d{10})(?!\d)')
KLADR_RE = re.compile(r'(?<!\d)(?:\d{17}|\d{15}|\d{13}|\d{11})(?!\d)')
BIRTH_CERT_RE = re.compile(r'(?<!\w)(?:I|II|III|IV|V|VI|VII|VIII|IX|X)[-– ]?[А-ЯЁ]{2}[ \t]*№?[ \t]*\d{6}(?!\d)', re.I)
TOKEN_RE = re.compile(r'[\w]+|[^\w\s]', re.UNICODE)


def has_context(text: str, start: int, end: int, words: tuple[str, ...], radius=55) -> bool:
    vicinity = text[max(0, start-radius):min(len(text), end+radius)].casefold()
    return any(w.casefold() in vicinity for w in words)


def has_preceding_context(text: str, start: int, words: tuple[str, ...], radius=35) -> bool:
    left = text[max(0, start-radius):start].casefold()
    return any(w.casefold() in left for w in words)


def add_from_pattern(acc, text, pattern, label, predicate=lambda x: True, contextual=None, radius=50):
    for m in pattern.finditer(text):
        val = m.group()
        if not predicate(val):
            continue
        if contextual and not has_context(text, m.start(), m.end(), contextual, radius):
            continue
        acc.append(Entity(label, m.start(), m.end(), val, 'rules'))


def detect(text: str, backend: str = "rubert", model_path: str | None = None) -> list[Entity]:
    if not isinstance(text, str):
        raise TypeError('text должен быть строкой')
    acc = []
    add_from_pattern(acc, text, EMAIL_RE, 'EMAIL')
    add_from_pattern(acc, text, IPV4_RE, 'IP', ip_valid)
    add_from_pattern(acc, text, IPV6_RE, 'IP', ip_valid,
                     contextual=('ip', 'ipv6', 'сервер', 'сети', 'адрес', 'узел'), radius=32)
    add_from_pattern(acc, text, BIRTH_CERT_RE, 'BIRTH_CERT',
                     contextual=('рождении', 'свидетельств', 'актовой записи'), radius=65)
    add_from_pattern(acc, text, PASSPORT_RE, 'PASSPORT',
                     contextual=('паспорт', 'серия', 'удостоверени'), radius=38)
    add_from_pattern(acc, text, OGRNIP_RE, 'OGRNIP', ogrnip_valid, contextual=('огрнип',), radius=35)


    for m in INN_RE.finditer(text):
        if inn_valid(m.group()) and has_preceding_context(text, m.start(), ('инн',), 32):
            acc.append(Entity('INN', m.start(), m.end(), m.group(), 'rules'))
    add_from_pattern(acc, text, CARD_RE, 'CARD',
                     lambda v: luhn_valid(re.sub(r'\D', '', v)) and re.sub(r'\D', '', v)[0] in '23456',
                     contextual=('карт', 'card', 'платёж'), radius=38)
    add_from_pattern(acc, text, KLADR_RE, 'KLADR',
                     lambda x: x[:2] != '00', contextual=('кладр',), radius=38)

    if backend != 'rubert':
        raise ValueError('Поддерживается только backend: rubert')
    from .rubert_ner import detect_rubert
    acc.extend(detect_rubert(text, model_path=model_path))

    priority = {'EMAIL': 0, 'BIRTH_CERT': 1, 'IP': 2, 'OGRNIP': 3, 'KLADR': 4,
                'CARD': 5, 'INN': 6, 'PASSPORT': 7, 'SURNAME': 8, 'NAME': 9}
    accepted = []
    for e in sorted(set(acc), key=lambda e: (priority[e.label], -(e.end-e.start), e.start)):
        if any(e.start < old.end and old.start < e.end for old in accepted):
            continue
        accepted.append(e)
    return sorted(accepted, key=lambda e: (e.start, e.end))


def tokenize(text: str) -> list[dict]:
    return [{'text': m.group(), 'start': m.start(), 'end': m.end()} for m in TOKEN_RE.finditer(text)]


def bio_tags(text: str, entities: list[Entity]) -> list[dict]:
    tok = tokenize(text)
    tagged = []
    for t in tok:
        tag = 'O'
        for e in entities:
            if t['start'] >= e.start and t['end'] <= e.end:
                tag = ('B-' if t['start'] == e.start else 'I-') + e.label
                break
        tagged.append({**t, 'tag': tag})
    return tagged


def recognize(text: str, backend: str = "rubert", model_path: str | None = None) -> dict:
    entities = detect(text, backend=backend, model_path=model_path)
    return {'text': text, 'entities': [e.to_dict() for e in entities],
            'bio': bio_tags(text, entities)}
