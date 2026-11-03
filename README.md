# Samolet: обнаружение лесных пожаров с БПЛА

Проект самолёта-патрульного БАС с бортовой нейросетью для обнаружения
дыма и огня. Участвует в конкурсе ИКаР (номинация «ИКаР-БАС») и в
пилоте для лесничества.

## Текущее состояние

Программная часть ИИ готова и проверена:

- **Модель:** YOLO11s, обучена на D-Fire + 24 своих кадра с дрона + AFSE + 129 веб-фото.
- **Веса:** `runs/mixed_v3_final/weights/best.pt` (финальная),
  предыдущие: `runs/mixed_dfire_custom/weights/best.pt`,
  базовая `runs/dfire_yolo11s_960/weights/best.pt`.
- **Метрики (D-Fire test, conf=0.1):** огонь P 0.715 / R 0.655 / mAP50 0.670;
  дым P 0.831 / R 0.772 / mAP50 0.797. Регрессии базы нет.
- **Внешние зонды (окт-2026):** Wikimedia-кадры: дым выжигания 0.92,
  лес без огня — тихо; белые голые стволы дают ложный smoke 0.2–0.36
  (новый класс негативов). Шлейф на весь кадр давится контекстом (0.14),
  кропом — 0.84 (нужен тайлинг). Единственный пропуск holdout — тление
  без видимого пламени (шум разметки AFSE), на всех масштабах 0.084–0.087.
- **Essential v3 = переупакованный AFSE:** test совпал с holdout 42/42
  (там же: огонь 28/29, дым 10/10, пустые 3/3 тихо), train/valid — те же
  кадры, что в `data/afse`. В обучение не брать (дубли).
- **Roboflow-«FLAME» — не воздушный:** мангалы, камин, горящий диван.
  Модель их игнорирует — корректно. Настоящий nadir-FLAME только на
  IEEE DataPort (нужна регистрация).
- **Воздушные кадры:** база видела ~1 очаг на кадр, дообученная — почти
  все (78 → 508 детекций на 22 кадрах). База не пострадала.
- **Скорость:** 3.8 мс/кадр на GPU. На Raspberry Pi 5 (NCNN) ожидается
  100–200 мс, то есть 5–10 кадров/с.
- **Рецепт борьбы с пропусками:** conf=0.1 для огня, TTA, трек по
  кадрам (тревога при 3 из 5), дым как ранний триггер, кроп + повторный
  детект.

## Структура

- `data/dfire/` — датасет D-Fire (train/val/test, YOLO-формат)
- `data/custom/` — свои 24 кадра с дрона (train/val)
- `data/custom_raw/` — исходные кадры из видео
- `data/afse/` — датасет AFSE (вид с дрона, train/valid/test)
- `data/afse_test_ours/` — AFSE test со свапом классов, вечный holdout
- `data/web_work/images/` — 129 размеченных веб-фото
- `data/merged_v3/` + `data/merged_v3.yaml` — объединённый train (D-Fire + custom + AFSE + web)
- `data/flame1/` — Roboflow-копия «FLAME» (НЕ настоящий nadir-FLAME: камины,
  диваны, мангалы; для обучения не годится, только как чужой зонд)
- `data/essential_v3/` — переупакованный AFSE (train/valid = те же кадры,
  test = наш holdout; нового не даёт, в обучение не брать)
- `data/probe_external/`, `data/probe_flame/` — внешние зонды (не в train!)
- `data/dfire.yaml`, `data/custom.yaml`, `data/mixed.yaml` — конфиги
- `training/` — скрипты обучения и оценки
- `runs/.../weights/best.pt` — веса
- `видео пожары/` — исходные видео

## Быстрый старт на новой машине

```
git clone <repo> samolet && cd samolet
python -m venv .venv && .venv\Scripts\activate        # linux: python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt                        # для CUDA см. шапку requirements.txt
# восстановить data/ по инструкции DATA.md, затем:
python training/check_data.py                          # проверка раскладки
python training/eval_model.py <веса из Release>        # единый тест, отчёт в reports/
```

Тест-батарея: D-Fire test (строго `batch=1` — батчевый val виснет),
AFSE holdout (тревога на кадре), отчёт `reports/eval_*/report.md`.
Приёмка: fire-recall D-Fire не хуже 0.655, holdout не хуже 28/29.

## Окружение

Используется окружение проекта `robot`:

```
C:\Users\pc\robot\.venv\Scripts\python.exe
```

В нём: PyTorch 2.7.1+cu118, ultralytics 8.4.187, pymavlink 2.4.50.

## Обучение

```
python training\train.py          # YOLO11n, 640, D-Fire
python training\train_v2.py       # YOLO11s, 960, D-Fire
python training\train_mixed.py    # D-Fire + свои кадры (финальная)
```

## Оценка

```
python training\tune_threshold.py runs\mixed_dfire_custom\weights\best.pt
python training\tile_eval.py runs\mixed_dfire_custom\weights\best.pt
```

## Следующие шаги

- Экспорт модели в NCNN и замер скорости на Raspberry Pi 5.
- Сбор новых воздушных кадров для независимой проверки.
- Бортовой конвейер: камера → инференс → GPS (MAVLink) → тревога.
- Наземная часть: карта очагов.
