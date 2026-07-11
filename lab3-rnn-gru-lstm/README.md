# Лабораторная 3 — RNN, GRU, LSTM для прогнозирования энергопотребления

Три рекуррентные архитектуры реализованы с нуля на чистом NumPy — включая ручной вывод и код backpropagation through time (BPTT) для каждой из них, без использования `torch.nn.RNN/GRU/LSTM`.

**Задача**: прогнозирование `Usage_kWh` на датасете [Steel Industry Energy Consumption](http://archive.ics.uci.edu/ml/datasets/Steel+Industry+Energy+Consumption+Dataset).

## Реализованные модели

- [`models/rnn.py`](models/rnn.py) — Vanilla RNN (`tanh`-активация, полный BPTT с grad clipping)
- [`models/gru.py`](models/gru.py) — GRU (update/reset гейты, кандидат состояния)
- [`models/lstm.py`](models/lstm.py) — LSTM (input/forget/output гейты, cell state)

## Результаты

| Модель | MSE | RMSE | R² |
|---|---|---|---|
| RNN | 0.1065 | 0.3264 | **0.8790** |
| GRU | 0.1140 | 0.3376 | 0.8705 |
| LSTM | 0.1212 | 0.3481 | 0.8623 |

Простая Vanilla RNN на этом датасете и горизонте прогноза показала себя не хуже, а по метрикам — даже немного лучше GRU/LSTM: короткие временные зависимости в почасовых данных энергопотребления не требуют сложных механизмов управления памятью, которые GRU/LSTM привносят ценой большего числа параметров и более сложной оптимизации.

Графики обучения и предсказания vs факт — в `results/{rnn,gru,lstm}/`.

## Запуск

```bash
python train.py --model rnn   # или gru / lstm
```
