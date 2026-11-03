# Данные: что где лежит и где взять

Папка `data/` в git **не хранится** (3.6 ГБ). На новой машине её надо
восстановить, затем проверить: `python training/check_data.py`.

## Обязательное (публичное)

| Папка | Что это | Размер | Где взять |
|---|---|---|---|
| `data/dfire/` | D-Fire, train/val/test + labels | ~3.0 ГБ | официальная страница датасета D-Fire |
| `data/afse/` | AFSE, train/valid (+ исходный test, он не holdout) | ~23 МБ | экспорт Roboflow-проекта AFSE |
| `data/afse_test_ours/` | AFSE test **со свапом классов** (`0<->1`), вечный holdout | ~4 МБ | собрать один раз из AFSE test и больше не трогать |

Классы везде наши: `0: smoke`, `1: fire`. AFSE исходно наоборот —
свап обязателен, иначе метрики holdout бессмысленны.

## Своё (перекинуть с исходной машины)

| Папка | Что это | Размер |
|---|---|---|
| `data/custom/` + `data/custom_raw/` | 24 кадра с дрона + исходные видео | ~9 МБ |
| `data/web_work/images/` | 129 размеченных веб-фото (jpg + txt рядом) | ~24 МБ |
| `data/merged_v3/` | собранный train (custom x30, AFSE x5, web x8) | ~416 МБ |

`merged_v3` можно не копировать, а пересобрать: `python training/build_mixed_v3.py`.

## Вода (второй рабочий поток, веса огня заморожены)

| Папка | Что это | Статус |
|---|---|---|
| `data/seadronessee/` | SeaDronesSee OD v2 (HF-зеркало `dronefreak/SeaDronesSee`, YOLO, CC0): train 8930 / val 1547, классы swimmer/boat/jetski/life_saving_appliances/buoy | качается через `training/fetch_seadronessee.py` |

Официальные ссылки датасета мертвы (Nextcloud 404) — используется
HF-зеркало. Holdout для воды: официальный val (разметка есть, test без
разметки). Вывод разведки: пловец медианно 50x40 px на 4K (12 px на
инференсе 960 — мелко, вопрос 960 vs 1280 открыт); огненная модель над
водой шумит (красное->fire, рябь->smoke до 0.59) — режимы миссий
раздельные, одновременно работает только одна модель.

## Не брать в обучение (лежат рядом, только для зондов)

- `data/flame1/` — Roboflow-копия «FLAME»: камины/мангалы, не воздух.
- `data/essential_v3/` — переупакованный AFSE (test = наш holdout!).
- `data/probe_external/`, `data/probe_flame/` — внешние проверки.

## Веса

Финальные веса лежат в GitHub Release (не в git):
`runs/mixed_v3_final/weights/best.pt` (~19 МБ). Для проверки достаточно
весов + `data/dfire/test` + `data/afse_test_ours/`.
