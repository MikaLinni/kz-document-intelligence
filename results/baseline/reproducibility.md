# Воспроизводимость baseline v1

## Окружение

Эксперимент и демо использовали локальную `.venv` на Apple M4 с 16 ГБ общей памяти. Обучение и плотный поиск выполнялись на PyTorch MPS в FP32 с SDPA. Seed эксперимента равен **42**.

| Компонент | Версия прогона |
|---|---|
| macOS | 15.5 |
| Python | 3.13.2 |
| PyTorch | 2.14.1 |
| Transformers | 5.19.0 |
| PEFT | 0.21.2 |
| Accelerate | 1.15.0 |
| rank-bm25 | 0.2.2 |
| NumPy | 2.5.4 |
| pandas | 3.0.6 |
| PyArrow | 26.0.0 |
| Tokenizers | 0.23.3 |
| Hugging Face Hub | 1.33.0 |

Полная конфигурация с версиями, параметрами и хешами скриптов записана во время исходного прогона. [Конфигурация](config.json)

## Корпус кандидатов

Файл `data/legal_chunks_20k.parquet` содержит 305 375 чанков. Оценка и демо используют один срез из **3 009 уникальных чанков семи документов**. Для вопроса RU доступны все 1 692 чанка RU из среза, для KK — все 1 317 чанков KK. Чанки упорядочиваются по строковому `chunk_id`, который также определяет порядок при равенстве баллов.

| `source_id` из QA | `document_id` корпуса |
|---|---:|
| `K030000442_` | 15924 |
| `K950001000_` | 1 |
| `K990000409_` | 3559 |
| `Z000000132_` | 7833 |
| `Z030000461_` | 16159 |
| `Z060000124_` | 26556 |
| `Z990000453_` | 4054 |

SHA‑256 корпуса — `b65485a62e7f9707049bd325f6c3b900c1b96eee565b3ac35fbedc10fc41515a`.

## Frozen benchmark

Данные и зафиксированные копии скриптов находятся в `data/frozen/baseline_v1/`. Проверка выполняется из корня проекта, поскольку манифест содержит пути от корня.

```bash
shasum -a 256 -c data/frozen/baseline_v1/SHA256SUMS.txt
shasum -a 256 data/legal_chunks_20k.parquet models/baseline_v1_lora/epoch-02/adapter_model.safetensors
```

| Файл | SHA‑256 |
|---|---|
| `retrieval_train_clean.parquet` | `498e05dd5f4134a855639dc635db906bc36b0cddd5766cb600437e3dea7bd4c8` |
| `retrieval_dev_clean.parquet` | `75480782da82945ad7aea5e4ea5ee433fbf185f685db2d3d68c815ca400a384a` |
| `retrieval_test_clean.parquet` | `cc396afb4afed353ba07e2dbf79b9f78d35b23f161be95ad6d2b27628b807155` |
| `bge_m3_training_records_clean.jsonl` | `48e16a984302f502cc9dd656f1a98e111b2ced6ade400759c9b9636934a40d5f` |
| `repair_all_splits.py` | `39ef4f99c31a16ed79dbecf020fcd3ace6d6f5aa6a49c623b8f832485ab46e39` |
| `build_clean_splits.py` | `84363cee991ab2ac506b22bc06bcf90f42174d20b4a2fecf0e226925e70efb31` |
| `prepare_training_data.py` | `c2285cd159384f2c2f8f2007e5762527a1527481617d9bdc37483c75add5c82c` |

Исходный манифест хранится в локальном frozen bundle. [Локальная структура данных](../../data/README.md#local-layout)

## Модель и чекпойнт

- Базовая модель — `BAAI/bge-m3`, ревизия `5617a9f61b028005a4858fdac845db406aefb181`. Для демо эта ревизия должна присутствовать в локальном кеше Hugging Face. [Закреплённая ревизия](https://huggingface.co/BAAI/bge-m3/tree/5617a9f61b028005a4858fdac845db406aefb181)
- Выбранный адаптер — `models/baseline_v1_lora/epoch-02/adapter_model.safetensors`. SHA‑256 — `206589ac8bb8b55fb2b207dff496653d82ba65ab958bc7893af943b278877982`. В том же каталоге находится `adapter_config.json`.
- В `selection.json` зафиксирована эпоха 2 с DEV Recall@10 0,9918032786885246 и остановкой после эпохи 4. [Выбор чекпойнта](selection.json)

## Как проверить пакет и запустить демо

Скрипты исходного эксперимента находятся в `scripts/train_bge_m3_lora.py` и `scripts/evaluate_test.py`. CSV и PNG этого пакета построены из сохранённых JSON. Для сверки чисел используйте указанные первичные файлы.

| Производный файл | Сохранённый источник |
|---|---|
| `metrics_table.csv`, `comparison_test_metrics.png`, `recall10_by_language.png` | [test_metrics.json](test_metrics.json) |
| `dev_history.csv`, `dev_training_history.png` | [dev_history.json](dev_history.json) и [selection.json](selection.json) |

CSV используют UTF‑8, запятую как разделитель и точку в десятичных числах. На эпохе 0 значение `train_loss` пустое, поскольку эта строка описывает zero-shot до обучения. В `dev_history.csv` значения 1 в столбцах `selected` и `early_stop` отмечают выбранную эпоху и точку остановки соответственно.

Команда ниже запускает локальный поиск с готовым адаптером и текущим корпусом.

```bash
python scripts/demo_search.py --interactive
```

Для работы плотного поиска нужен терминал с доступом к Metal. BM25 работает без GPU. [Проверенные запросы и команды для показа](demo_queries.md)
