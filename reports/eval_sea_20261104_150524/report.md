# Eval SEA 20261104_150524 PASS

weights: `C:\Users\pc\samolet\runs\sea_yolo11n_960\weights\best.pt` | conf=0.25 imgsz=960 | v1 после правки базы под conf=0.25 | 108s

## SeaDronesSee val

| | P | R | mAP50 | alarm |
|---|---|---|---|---|
| swimmer | 0.840 | 0.803 | 0.759 | 1253/1286 |
| boat | 0.933 | 0.939 | 0.939 | 915/915 |
| jetski | 0.870 | 0.884 | 0.858 | 294/320 |
| life_saving_appliances | 0.835 | 0.433 | 0.448 | 145/191 |
| buoy | 0.863 | 0.777 | 0.814 | 348/352 |

Пропуски swimmer (6): `2035`, `2109`, `2241`, `3465`, `3466`, `3467`

## Кросс-домен (водная на суше)

- afse_holdout: 71 боксов / 42 кадров
- dfire150: 49 боксов / 150 кадров
