

from __future__ import annotations
import random, json, re
from collections import Counter
from pathlib import Path
from .lexicon import MALE_NAMES, FEMALE_NAMES, SURNAME_BASES
from .validators import inn_valid, ogrnip_valid, luhn_valid
from .detector import LABELS


def generate_inn(rng):
    prefix = rng.choice(['77','78','50','66','16','23','54']) + f'{rng.randint(1,99):02d}'
    digits = [int(c) for c in (prefix + ''.join(str(rng.randrange(10)) for _ in range(6)))]
    a = [7,2,4,10,3,5,9,4,6,8]
    b = [3,7,2,4,10,3,5,9,4,6,8]
    n11 = sum(x*y for x,y in zip(digits,a)) % 11 % 10
    digits.append(n11)
    n12 = sum(x*y for x,y in zip(digits,b)) % 11 % 10
    digits.append(n12)
    v=''.join(map(str,digits))
    assert inn_valid(v)
    return v


def generate_ogrnip(rng):
    body = rng.choice(['3','4']) + f'{rng.randint(10,29):02d}' + f'{rng.randint(1,99):02d}' + ''.join(str(rng.randrange(10)) for _ in range(9))
    assert len(body)==14
    val=body + str(int(body)%13%10)
    assert ogrnip_valid(val)
    return val


def generate_card(rng):
    body = rng.choice(['2200', '4276', '5481', '6011']) + ''.join(str(rng.randrange(10)) for _ in range(11))
    for n in '0123456789':
        if luhn_valid(body+n):
            value=body+n
            break
    else:
        raise AssertionError('Luhn check digit not found')
    if rng.random() < .55:
        return ' '.join(value[i:i+4] for i in range(0,16,4))
    return value


def fake_person(rng, label):
    if label=='NAME':
        group=rng.choice([MALE_NAMES,FEMALE_NAMES])
        return rng.choice(rng.choice(list(group.values()))[:3])
    b=rng.choice(SURNAME_BASES)
    return rng.choice([b,b+'а',b+'у',b+'ой'])


def value_for(rng, label):
    if label in ('NAME','SURNAME'): return fake_person(rng,label)
    if label=='INN': return generate_inn(rng)
    if label=='OGRNIP': return generate_ogrnip(rng)
    if label=='PASSPORT':
        digits = f'{rng.randint(10,79):02d}{rng.randint(0,29):02d}{rng.randint(0,999999):06d}'
        return digits[:4] + (' ' if rng.random()<.7 else '') + digits[4:]
    if label=='CARD':return generate_card(rng)
    if label=='IP':
        if rng.random()<.22:
            return f'2001:0db8:{rng.randint(1,65535):04x}:0000:0000:0000:0000:{rng.randint(1,65535):04x}'
        return rng.choice(['192.0.2.', '198.51.100.', '203.0.113.'])+str(rng.randint(1,254))
    if label=='EMAIL':
        left=rng.choice(['user','client','support','contact','staff','operator'])+str(rng.randint(100,999))
        return left+'@'+rng.choice(['example.test','demo.example','mail.example'])
    if label=='BIRTH_CERT':
        roman=rng.choice(['I','II','III','IV','V'])
        letters=rng.choice(['АБ','ВГ','ДЕ','ЖЗ','КЛ','МН','ПР'])
        return f'{roman}-{letters}'+rng.choice([' ', ' № '])+f'{rng.randint(0,999999):06d}'
    if label=='KLADR':
        length=rng.choice([11,13,15,17])
        return rng.choice(['77','78','50','66','16','23','54']) + ''.join(str(rng.randrange(10)) for _ in range(length-2))
    raise ValueError(label)


TEMPLATES = {
    'SURNAME': ['Фамилия заявителя: {v}', 'В карточке клиента записана фамилия {v}', 'Сведения о сотруднике {v}'],
    'NAME': ['Имя заявителя: {v}', 'К заявке прикреплено имя {v}', 'Контактное лицо по имени {v}'],
    'INN': ['ИНН физического лица: {v}', 'Для сверки налогового учёта указан ИНН {v}', 'Налоговый номер (ИНН) {v}'],
    'PASSPORT': ['Паспорт гражданина: {v}', 'Для паспорта указаны серия и номер {v}', 'В анкете номер паспорта {v}'],
    'CARD': ['Номер банковской карты: {v}', 'Карта плательщика {v}', 'Реквизит карты {v}'],
    'OGRNIP': ['ОГРНИП предпринимателя: {v}', 'В выписке указан ОГРНИП {v}', 'Регистрационный ОГРНИП {v}'],
    'IP': ['IP-адрес сервера: {v}', 'В сетевом журнале обнаружен IP {v}', 'Удалённый узел использует адрес {v}'],
    'EMAIL': ['Электронная почта: {v}', 'Для связи используется e-mail {v}', 'Адрес почты в системе {v}'],
    'BIRTH_CERT': ['Свидетельство о рождении: {v}', 'В анкете номер свидетельства о рождении {v}', 'Серия и номер свидетельства о рождении {v}'],
    'KLADR': ['Код КЛАДР: {v}', 'Адресу присвоен КЛАДР {v}', 'В поле классификатора КЛАДР указан код {v}'],
}
NEGATIVES = [
    'Версия приложения 999.999.1.1 не является корректным IP-адресом.',
    'Тестовый идентификатор 000000000000 не прошёл формальный контроль.',
    'Произвольная последовательность 111111111111111 не является регистрационным номером.',
    'Встречается строка 1234-5678-9012-3456 с неверной контрольной цифрой.',
    'Документ обработан без указания фамилии и имени владельца.',
    'Нужно сравнить две схемы входных файлов и сохранить результат.',
    'Администратор обновил таблицу маршрутизации сети.',
    'Сотрудник обсуждает форму заявления и поля реестра.',
    'Клиент не предоставил номер документа или контакты.',
    'Система проверяет шаблон при помощи регулярного выражения.',
]


def dataset(seed: int, n_positive: int, n_negative: int, prefix: str) -> list[dict]:
    rng=random.Random(seed)
    out=[]
    for i in range(n_positive):

        labels=[LABELS[((i*7)%len(LABELS)+j)%len(LABELS)] for j in range(6)]
        rng.shuffle(labels)
        parts=[]; gold=[]; offset=0
        for j,label in enumerate(labels):
            val=value_for(rng,label)
            s=rng.choice(TEMPLATES[label]).format(v=val)
            pos=s.index(val)
            parts.append(s)
            gold.append({'label':label,'start':offset+pos,'end':offset+pos+len(val)})
            offset += len(s)+2
        text='. '.join(parts)+'.'
        assert all(text[e['start']:e['end']] for e in gold)
        out.append({'id':f'{prefix}-{i:04d}','text':text,'entities':sorted(gold,key=lambda a:a['start'])})
    for i in range(n_negative):
        text=rng.choice(NEGATIVES)
        out.append({'id':f'{prefix}-NEG-{i:03d}','text':text,'entities':[]})
    rng.shuffle(out)
    return out


def write_jsonl(records, path):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,'w',encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec,ensure_ascii=False)+'\n')


def main():


    root=Path(__file__).resolve().parents[1]
    test=dataset(seed=20261119,n_positive=200,n_negative=40,prefix='TEST')
    for record in test:
        record['source']='Дополнительные синтетические тексты, вариант 18'
        record['dataset_split']='test'
    path=root/'data'/'variant18_extra_test.jsonl'
    write_jsonl(test,path)
    counts=Counter(e['label'] for r in test for e in r['entities'])
    print('Создан тестовый набор:',path)
    print('Количество текстов:',len(test),'Количество сущностей:',sum(counts.values()))
    print('Распределение по классам:',dict(counts))


if __name__=='__main__':main()
