

from __future__ import annotations
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random

LABELS=['O','B-SURNAME','I-SURNAME','B-NAME','I-NAME']
LABEL2ID={v:k for k,v in enumerate(LABELS)}
ID2LABEL={k:v for k,v in enumerate(LABELS)}


def split_dev_records(records: list[dict], seed=42, val_fraction=0.15):

    if len(records)<10:raise ValueError('Мало размеченных документов для обучения')
    ids=[rec['id'] for rec in records]
    if len(ids)!=len(set(ids)):raise ValueError('Найдены дублирующиеся идентификаторы документов')
    if any(rec.get('source') != 'FactRuEval-2016' for rec in records):
        raise ValueError('Ожидался FactRuEval-2016: поле source у части записей отличается')

    ordered=sorted(records,key=lambda rec:rec['id'])
    rng=random.Random(seed)
    rng.shuffle(ordered)
    n_val=max(1,min(len(ordered)-1,math.ceil(len(ordered)*val_fraction)))
    val,train=ordered[:n_val],ordered[n_val:]
    assert not {r['id'] for r in train}&{r['id'] for r in val}
    return train,val


def prepare_windows(records,tokenizer,max_length=384,stride=64):

    if max_length>512 or max_length<32 or stride>=max_length-2:
        raise ValueError('Неверные max_length/stride')
    out=[]
    for rec in records:
        text=rec['text']
        ents=sorted([e for e in rec['entities'] if e['label'] in ('NAME','SURNAME')],
                    key=lambda e:(e['start'],e['end']))
        for e in ents:
            assert text[e['start']:e['end']].strip(),(rec['id'],e)
        encoded=tokenizer(text,truncation=True,max_length=max_length,stride=stride,
                          return_overflowing_tokens=True,return_offsets_mapping=True)
        for j in range(len(encoded['input_ids'])):
            offsets=encoded['offset_mapping'][j]
            word_ids=encoded.word_ids(batch_index=j)
            y=[]
            prev_word=None
            prev_ent=None
            for (start,end),wid in zip(offsets,word_ids):
                if wid is None or (start==end):
                    y.append(-100);continue
                if wid==prev_word:
                    y.append(-100);continue
                prev_word=wid
                found=None
                for e in ents:
                    if start<e['end'] and end>e['start']:
                        found=e;break
                if found is None:
                    y.append(LABEL2ID['O']);prev_ent=None
                else:
                    key=(found['label'],found['start'],found['end'])
                    prefix='B' if key!=prev_ent else 'I'
                    y.append(LABEL2ID[f'{prefix}-{found["label"]}'])
                    prev_ent=key
            out.append({'input_ids':encoded['input_ids'][j],
                        'attention_mask':encoded['attention_mask'][j],
                        'labels':y})
    return out


def compute_metrics(p,seqeval_metrics):
    import numpy as np
    logits,labels=p
    pred=np.argmax(logits,axis=2)
    true_labels=[];pred_labels=[]
    for ys,ps in zip(labels,pred):
        truth=[];guesses=[]
        for y,pred_id in zip(ys,ps):
            if int(y)==-100:continue
            truth.append(ID2LABEL[int(y)])
            guesses.append(ID2LABEL[int(pred_id)])
        true_labels.append(truth)
        pred_labels.append(guesses)
    from seqeval.scheme import IOB2
    return {'precision':seqeval_metrics.precision_score(true_labels,pred_labels,mode='strict',scheme=IOB2),
            'recall':seqeval_metrics.recall_score(true_labels,pred_labels,mode='strict',scheme=IOB2),
            'f1':seqeval_metrics.f1_score(true_labels,pred_labels,mode='strict',scheme=IOB2)}


def train(argv=None):
    parser=argparse.ArgumentParser(description='Обучение RuBERT на FactRuEval devset (без доступа к testset)')
    parser.add_argument('--train-data',type=Path,default=Path('data/factru_dev.jsonl'))
    parser.add_argument('--output',type=Path,default=Path('models/rubert_names'))
    parser.add_argument('--checkpoint-dir',type=Path,default=Path('training/rubert'))
    parser.add_argument('--model-name',default='DeepPavlov/rubert-base-cased')
    parser.add_argument('--epochs',type=float,default=8.0)
    parser.add_argument('--learning-rate',type=float,default=2e-5)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--gradient-accumulation-steps',type=int,default=1,
                        help='Накопление градиентов: эффективный batch = batch-size * число шагов')
    parser.add_argument('--max-length',type=int,default=384)
    parser.add_argument('--stride',type=int,default=64)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--validation-fraction',type=float,default=0.15)
    parser.add_argument('--resume-from',default=None,help='Путь к checkpoint-* для восстановления')
    parser.add_argument('--allow-cpu',action='store_true',help='Разрешить обучение без CUDA (значительно медленнее)')
    args=parser.parse_args(argv)
    if not args.train_data.is_file():raise FileNotFoundError(f'Нет {args.train_data}; сначала запустите pii_recognizer.factru с --subset devset')
    if not 0<args.validation_fraction<0.5:raise ValueError('validation_fraction должна быть между 0 и 0.5')
    if args.batch_size<1 or args.gradient_accumulation_steps<1:
        raise ValueError('batch-size и gradient-accumulation-steps должны быть >= 1')
    from .metrics import load_jsonl
    records=load_jsonl(args.train_data)
    if not all(str(r['id']).startswith('FACTRU-') and r.get('dataset_split')=='devset' for r in records):
        raise ValueError('Обучение разрешено только по данным с dataset_split=devset. Повторно экспортируйте devset текущей версией pii_recognizer.factru')
    train_rec,val_rec=split_dev_records(records,args.seed,args.validation_fraction)
    print('Документов: train =',len(train_rec),'validation =',len(val_rec),flush=True)
    print('Сущности train =',dict(Counter(e['label'] for r in train_rec for e in r['entities'])),flush=True)
    try:
        import torch
        import transformers
        from transformers import (AutoTokenizer,AutoModelForTokenClassification,
                                  DataCollatorForTokenClassification,TrainingArguments,Trainer,set_seed)
        from datasets import Dataset
        import seqeval.metrics
    except ImportError as exc:
        raise RuntimeError('Установите зависимости rubert-requirements.txt') from exc
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError('CUDA не обнаружена. Установите CUDA-сборку PyTorch или добавьте --allow-cpu (медленно)')
    print('Устройство:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',flush=True)
    set_seed(args.seed)
    tok=AutoTokenizer.from_pretrained(args.model_name,use_fast=True)
    if not tok.is_fast:raise ValueError('Требуется fast-токенизатор')
    train_win=prepare_windows(train_rec,tok,args.max_length,args.stride)
    val_win=prepare_windows(val_rec,tok,args.max_length,args.stride)
    print('Окон: train =',len(train_win),'validation =',len(val_win),flush=True)
    model=AutoModelForTokenClassification.from_pretrained(args.model_name,num_labels=5,
              id2label=ID2LABEL,label2id=LABEL2ID,ignore_mismatched_sizes=True)
    kwargs=dict(output_dir=str(args.checkpoint_dir),eval_strategy='epoch',save_strategy='epoch',
        learning_rate=args.learning_rate,per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_eval_batch_size=args.batch_size,num_train_epochs=args.epochs,
        weight_decay=0.01,logging_steps=20,load_best_model_at_end=True,
        metric_for_best_model='f1',greater_is_better=True,save_total_limit=2,
        report_to='none',seed=args.seed,data_seed=args.seed,
        fp16=bool(torch.cuda.is_available()),save_safetensors=True)
    trainer=Trainer(model=model,args=TrainingArguments(**kwargs),
        train_dataset=Dataset.from_list(train_win),eval_dataset=Dataset.from_list(val_win),
        processing_class=tok,data_collator=DataCollatorForTokenClassification(tok),
        compute_metrics=lambda p:compute_metrics(p,seqeval.metrics))
    trainer.train(resume_from_checkpoint=args.resume_from)
    val_metrics=trainer.evaluate()
    args.output.mkdir(parents=True,exist_ok=True)
    trainer.save_model(str(args.output))
    tok.save_pretrained(str(args.output))
    provenance={'model_name':args.model_name,'training_source':'FactRuEval-2016/devset',
       'testset_used_for_training':False,'train_documents':len(train_rec),
       'validation_documents':len(val_rec),'train_ids':[r['id'] for r in train_rec],
       'validation_ids':[r['id'] for r in val_rec],
       'label_names':LABELS,'seed':args.seed,'max_length':args.max_length,
       'stride':args.stride,'epochs':args.epochs,'learning_rate':args.learning_rate,
       'batch_size':args.batch_size,'gradient_accumulation_steps':args.gradient_accumulation_steps,
       'validation_metrics':val_metrics,'transformers_version':transformers.__version__,
       'torch_version':torch.__version__}
    (args.output/'training_manifest.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf-8')
    weights=[args.output/'model.safetensors',args.output/'pytorch_model.bin']
    if not any(f.is_file() and f.stat().st_size>1000000 for f in weights):
        raise RuntimeError('ОБУЧЕНИЕ ЗАВЕРШИЛОСЬ, НО ВЕСА НЕ СОХРАНЕНЫ. Проверьте папку '+str(args.output))
    print('Модель и веса сохранены:',args.output,flush=True)
    print('Проверка файлов:',[(f.name,f.stat().st_size) for f in weights if f.exists()],flush=True)
    print('VALIDATION METRICS:',json.dumps(val_metrics,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':train()
