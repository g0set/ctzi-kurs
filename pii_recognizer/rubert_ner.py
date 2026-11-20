

from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from .detector import Entity

@lru_cache(maxsize=2)
def _load_rubert(model_path: str):
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForTokenClassification
    except ImportError as exc:
        raise RuntimeError('Установите зависимости: pip install -r rubert-requirements.txt') from exc
    folder=Path(model_path)
    if not (folder/'config.json').exists() or not any((folder/f).exists() for f in ('model.safetensors','pytorch_model.bin')):
        raise FileNotFoundError('Не найдены веса обученной RuBERT: '+str(folder))
    tokenizer=AutoTokenizer.from_pretrained(str(folder),use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError('Требуется быстрый токенизатор с offset_mapping')
    model=AutoModelForTokenClassification.from_pretrained(str(folder))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    return tokenizer,model,device


def decode_word_predictions(text: str, words: list[tuple[int,int,str]]):

    entities=[]
    active=None
    def flush():
        nonlocal active
        if active is not None:
            label,start,end=active
            if text[start:end].strip():
                entities.append(Entity(label,start,end,text[start:end],'rubert'))
            active=None
    for start,end,tag in sorted(words, key=lambda x:(x[0],x[1])):
        if not 0<=start<end<=len(text):continue
        if tag=='O' or '-' not in tag:
            flush();continue
        prefix,label=tag.split('-',1)
        if label not in {'NAME','SURNAME'}:
            flush();continue
        if prefix=='I' and active is not None and active[0]==label and start>=active[2] and not text[active[2]:start].strip():
            active[2]=end
        else:
            flush()
            active=[label,start,end]
    flush()
    return entities


def detect_rubert(text: str, model_path: str | Path | None, max_length=384, stride=64, batch_size=8):
    if not model_path:
        raise ValueError('Для RuBERT нужен --model-path с обученной моделью')
    if not text.strip():return []
    import torch
    tok,model,device=_load_rubert(str(model_path))
    enc=tok(text,truncation=True,max_length=max_length,stride=stride,
            return_overflowing_tokens=True,return_offsets_mapping=True)
    if 'offset_mapping' not in enc:raise ValueError('Не найден offset_mapping')


    best={}
    for base in range(0,len(enc['input_ids']),batch_size):
        stop=min(base+batch_size,len(enc['input_ids']))
        batches=tok.pad({'input_ids':enc['input_ids'][base:stop],
                         'attention_mask':enc['attention_mask'][base:stop]},
                        padding=True,return_tensors='pt')
        batches={k:v.to(device) for k,v in batches.items()}
        with torch.inference_mode():
            preds=model(**batches).logits.argmax(dim=-1).cpu().tolist()
        for row,j in enumerate(range(base,stop)):
            ids=enc.word_ids(batch_index=j)
            offsets=enc['offset_mapping'][j]
            seq_len=len(ids)
            k=0
            while k<seq_len:
                word_id=ids[k]
                if word_id is None:
                    k+=1;continue
                last=k+1
                while last<seq_len and ids[last]==word_id:last+=1
                start=int(offsets[k][0]);end=int(offsets[last-1][1])
                if 0<=start<end<=len(text):
                    pred_id=int(preds[row][k])
                    tag=str(model.config.id2label[pred_id])
                    centrality=min(k,seq_len-1-(last-1))
                    key=(start,end)
                    if key not in best or centrality>best[key][0]:
                        best[key]=(centrality,tag)
                k=last
    words=[(s,e,v[1]) for (s,e),v in best.items()]
    return decode_word_predictions(text,words)
