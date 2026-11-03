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

## Не брать в обучение (лежат рядом, только для зондов)

- `data/flame1/` — Roboflow-копия «FLAME»: камины/мангалы, не воздух.
- `data/essential_v3/` — переупакованный AFSE (test = наш holdout!).
- `data/probe_external/`, `data/probe_flame/` — внешние проверки.

## Веса

Финальные веса лежат в GitHub Release (не в git):
`runs/mixed_v3_final/weights/best.pt` (~19 МБ). Для проверки достаточно
весов + `data/dfire/test` + `data/afse_test_ours/`.
