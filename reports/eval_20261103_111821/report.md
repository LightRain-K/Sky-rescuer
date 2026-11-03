# Eval 20261103_111821 PASS

weights: `C:\Users\pc\samolet\runs\mixed_v3_final\weights\best.pt` | conf=0.1 imgsz=960 | первый прогон после переезда yaml на относительные пути | 74s

## D-Fire test (batch=1)

| | P | R | mAP50 | base R/mAP |
|---|---|---|---|---|
| fire | 0.715 | 0.655 | 0.670 | 0.655/0.670 |
| smoke | 0.831 | 0.772 | 0.797 | 0.772/0.797 |

## AFSE holdout (alarm per frame)

| conf | fire | smoke-only | quiet |
|---|---|---|---|
| 0.1 | 28/29 | 10/10 | 3/3 |
| 0.05 | 29/29 | 10/10 | 3/3 |

Пропуски огня (1): `Original-Sized-RGB-Frame-45854-_jpg.rf.671ef670a6f` (smoke 0.14)

Аннотации пропусков лежат рядом с отчётом.
