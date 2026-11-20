# Распознавание персональных данных — ЦТЗИ, вариант №18

Программа на Python для распознавания **10 типов персональных данных** в русскоязычных текстах. Используется гибридный подход: **RuBERT** определяет имена и фамилии по контексту, а **регулярные выражения (Regex) и валидаторы** находят данные заданного формата.

Поддерживаемые типы: `SURNAME` (фамилия), `NAME` (имя), `INN` (ИНН), `PASSPORT` (паспорт), `CARD` (банковская карта), `OGRNIP` (ОГРНИП), `IP` (IP-адрес), `EMAIL` (электронная почта), `BIRTH_CERT` (свидетельство о рождении), `KLADR` (КЛАДР).

Программа выводит найденные сущности и их позиции в тексте. При оценке качества показывает таблицы **Precision, Recall, F1, TP, FP, FN** и сохраняет результаты в JSON. Проект реализует **распознавание**, без обезличивания.

## Требования и установка

- Windows, PowerShell, Python **3.11**;
- подключение к интернету для первоначальной установки зависимостей и скачивания обученной модели;
- для распознавания на CPU не требуется видеокарта NVIDIA.

Откройте PowerShell **в корневой папке проекта** и выполните:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Если виртуальное окружение `.venv` уже существует и зависимости установлены, повторять установку не нужно.

## Загрузка готовой обученной модели

**Повторное обучение RuBERT для проверки проекта не требуется.** Веса модели публикуются отдельно от исходного кода в разделе **Releases** этого GitHub-репозитория, поскольку `model.safetensors` занимает около 709 МБ и превышает лимит обычного Git-репозитория.

1. Откройте вкладку **Releases** на странице репозитория и выберите релиз с обученной RuBERT (например, `v1.0`).
2. В разделе **Assets** скачайте архив `rubert_names_weights.zip`.
3. В корневой папке проекта выполните следующую команду PowerShell — она создаст каталог `models/rubert_names` и распакует в него модель:

```powershell
New-Item -ItemType Directory -Force -Path .\models\rubert_names | Out-Null
Expand-Archive -Path .\rubert_names_weights.zip -DestinationPath .\models\rubert_names -Force
```

Архив нужно предварительно положить в **корень проекта**. После распаковки должны существовать `models/rubert_names/model.safetensors`, `models/rubert_names/config.json` и файлы токенизатора. Архив содержит **содержимое** папки `rubert_names`, а не дополнительную вложенную папку с тем же названием.

Проверка наличия весов:

```powershell
Test-Path .\models\rubert_names\model.safetensors
```

Ожидаемый ответ — `True`. Если релиз ещё не опубликован, за готовой моделью необходимо обратиться к автору проекта: исходный код без весов не может выполнять распознавание имён и фамилий.

## Запуск распознавания

**По тексту из командной строки:**

```powershell
.\.venv\Scripts\python.exe -m pii_recognizer.cli --backend rubert --model-path models/rubert_names --text 'Иван Петров написал на user@example.com с адреса 192.168.1.10'
```

**По тексту из файла UTF-8:** создайте `example.txt` в корневой папке проекта и выполните:

```powershell
.\.venv\Scripts\python.exe -m pii_recognizer.cli --backend rubert --model-path models/rubert_names --file example.txt
```

Без `--out` найденные сущности выводятся в консоль как JSON. Для записи в файл добавьте `--out results/example.json`.

## Оценка качества

**Все 10 типов ПДн** — дополнительный размеченный набор `data/variant18_extra_test.jsonl`: 240 синтетических текстов, по 120 сущностей каждого типа.

```powershell
.\.venv\Scripts\python.exe -m pii_recognizer.cli --backend rubert --model-path models/rubert_names --evaluate data/variant18_extra_test.jsonl --out results/variant18_extra_rubert.json
```

**FactRuEval-2016** — отдельная оценка имён и фамилий на официальном тестовом наборе из 132 документов. Сначала скачайте корпус и подготовьте файл `data/factru_test.jsonl`:

```powershell
.\.venv\Scripts\python.exe -m pii_recognizer.factru --download --root external/factrueval --subset testset --out data/factru_test.jsonl
.\.venv\Scripts\python.exe -m pii_recognizer.cli --backend rubert --model-path models/rubert_names --names-only --evaluate data/factru_test.jsonl --out results/factru_rubert.json
```

После тестирования появится таблица метрик. Чтобы повторно открыть таблицу из сохранённого JSON **без запуска нейросети**, выполните:

```powershell
.\.venv\Scripts\python.exe -m pii_recognizer.cli --show-report results/variant18_extra_rubert.json
```

**Ограничение:** FactRuEval в используемой разметке содержит только классы `NAME` и `SURNAME`. Полная оценка 10 типов проводится на дополнительной синтетической выборке; эти показатели нельзя напрямую переносить на реальные документы.

## Повторное обучение (не обязательно для проверки)

При наличии готового архива из Releases обучение пропускается. Для самостоятельного воспроизведения обучения понадобится исходная RuBERT и обучающая часть FactRuEval-2016:

```powershell
.\.venv\Scripts\python.exe -m pii_recognizer.factru --download --root external/factrueval --subset devset --out data/factru_dev.jsonl
.\.venv\Scripts\python.exe -m pii_recognizer.train_rubert --train-data data/factru_dev.jsonl --output models/rubert_names --checkpoint-dir training/rubert --epochs 8 --batch-size 1 --gradient-accumulation-steps 4 --max-length 128 --stride 32 --allow-cpu
```

## Состав репозитория

- `pii_recognizer/` — распознавание, правила, обучение и оценка;
- `data/variant18_extra_test.jsonl` — размеченный тест для 10 классов;
- `tests/` — автоматические проверки;
- `requirements.txt`, `rubert-requirements.txt` — зависимости;
- `.gitignore` — исключения для временных и больших файлов.

В обычный Git-репозиторий **не включаются** `.venv/`, `models/`, `training/`, `external/`, `results/`, архивы весов и кэш Python. Готовая модель распространяется через **GitHub Releases**. Исходный FactRuEval скачивается отдельно.
