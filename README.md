# Deep Learning — лабораторные работы

Курс по глубокому обучению: нейросетевые архитектуры, реализованные с нуля на чистом NumPy (без PyTorch/TensorFlow) — свёрточные, рекуррентные сети и генеративные модели, с ручным выводом и кодом backpropagation для каждой архитектуры.

Курс включал 6 лабораторных работ, в репозитории представлены 4 из них — лаб. 5 и 6 не выполнялись (см. ниже).

## Лабораторная 1 — Многослойный перцептрон (классификация грибов)

MLP с нуля (2 скрытых слоя, сигмоида, обычный градиентный спуск без оптимизаторов), accuracy/precision/recall/F1/ROC-AUC на датасете UCI Mushroom.

📄 [`lab1-mlp-mushroom/`](lab1-mlp-mushroom)

## Лабораторная 2 — Сверточная нейросеть (MNIST)

CNN с нуля (свёртка, ReLU, max-pooling, ручной backward), матрица ошибок → accuracy/precision/recall/F1, ROC-AUC (one-vs-rest), предварительный t-SNE.

📄 [`lab2-cnn-mnist/`](lab2-cnn-mnist)

## Лабораторная 3 — RNN, GRU, LSTM (прогноз энергопотребления)

Три рекуррентные архитектуры с нуля, включая полный ручной BPTT для каждой. Сравнение по MSE/RMSE/R² на датасете Steel Industry Energy Consumption.

📄 [`lab3-rnn-gru-lstm/`](lab3-rnn-gru-lstm)

## Лабораторная 4 — GAN + VAE (восстановление изображений MNIST)

Трёхкомпонентная сеть (Encoder + Decoder + Discriminator) с нуля: VAE с reparameterization trick и KL-warmup, дообученный adversarial-лоссом для более чётких реконструкций.

📄 [`lab4-gan-vae/`](lab4-gan-vae)

## Стек

NumPy (вся математика нейросетей — свёртки, пулинг, BPTT, backprop — реализована вручную), Matplotlib, Plotly (t-SNE), Pandas, scikit-learn (только вспомогательные метрики/t-SNE, не сами модели)
