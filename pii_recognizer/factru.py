

from __future__ import annotations
import argparse
import io
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

ARCHIVE_URL='https://github.com/dialogue-evaluation/factRuEval-2016/archive/refs/heads/master.zip'


def write_jsonl(records, output: Path):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + '\n')


def download(output: Path):
    output.mkdir(parents=True,exist_ok=True)
    if list(output.glob('**/devset/*.spans')):
        print('FactRuEval уже найден в',output);return
    print('Получение официального FactRuEval-2016:',ARCHIVE_URL)
    with urlopen(ARCHIVE_URL,timeout=90) as resp:
        payload=resp.read()
    with ZipFile(io.BytesIO(payload)) as z:

        for entry in z.infolist():
            parts=Path(entry.filename).parts
            if '..' in parts or Path(entry.filename).is_absolute():
                raise ValueError('Опасный путь в ZIP-архиве')
        z.extractall(output)
    print('Архив распакован. Размер:',len(payload),'байт')


def get_root(path: Path) -> Path:
    if (path/'devset').is_dir(): return path
    hits=list(path.glob('*/devset'))
    if len(hits)==1:return hits[0].parent
    raise FileNotFoundError('Не найдена папка devset/ FactRuEval; запустите --download')


def parse_doc(path: Path) -> dict:
    text=path.read_text(encoding='utf-8-sig')
    span_file=path.with_suffix('.spans')
    objects_file=path.with_suffix('.objects')
    spans={}
    for line in span_file.read_text(encoding='utf-8-sig').splitlines():
        fields=line.split('#',1)[0].split()
        if len(fields)<4:continue
        sid,typ,start,length=fields[:4]
        spans[sid]=(typ,int(start),int(start)+int(length))
    person_spans=set()
    for line in objects_file.read_text(encoding='utf-8-sig').splitlines():
        fields=line.split('#',1)[0].split()
        if len(fields)>2 and fields[1].lower()=='person':
            person_spans.update(fields[2:])
    entities=[]; seen=set()
    for sid in sorted(person_spans):
        if sid not in spans:continue
        typ,start,end=spans[sid]
        label={'name':'NAME','surname':'SURNAME'}.get(typ)
        if not label or not 0<=start<end<=len(text):continue
        segment=text[start:end]
        if not segment.strip():continue
        key=(label,start,end)
        if key in seen:continue
        seen.add(key)
        entities.append({'label':label,'start':start,'end':end})
    entities.sort(key=lambda e:(e['start'],e['end']))

    return {'id':'FACTRU-'+path.stem,'text':text,'entities':entities,'source':'FactRuEval-2016'}


def export_factru(root: Path,output: Path,subset='testset'):
    raw=get_root(root)
    src=raw/subset
    files=sorted(src.glob('*.txt'))
    if not files:raise FileNotFoundError(f'{src} не содержит *.txt')
    records=[parse_doc(p) for p in files if p.with_suffix('.spans').exists() and p.with_suffix('.objects').exists()]
    if not records:raise ValueError('Не удалось прочитать разметку Person')
    for record in records:
        record['dataset_split']=subset
    write_jsonl(records,output)
    from collections import Counter
    counts=Counter(e['label'] for rec in records for e in rec['entities'])
    print('FactRuEval',subset,':',len(records),'документов;',dict(counts),'->',output)
    return records


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--download',action='store_true',help='Скачать и распаковать с GitHub')
    ap.add_argument('--root',type=Path,default=Path('external/factrueval'))
    ap.add_argument('--subset',choices=['devset','testset'],default='testset')
    ap.add_argument('--out',type=Path,default=Path('data/factru_test.jsonl'))
    args=ap.parse_args()
    if args.download: download(args.root)
    export_factru(args.root,args.out,args.subset)

if __name__=='__main__':main()
