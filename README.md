# Самостоятельное задание №1

Тема: разработка модели выявления признаков фальсификации или монтажа на цифровых изображениях для предотвращения мошенничества.

В репозитории есть отчет и рабочая нейросеть на PyTorch:

- `Самостоятельное_задание_1_отчет_CNN_фальсификация_изображений.docx` - отчет для защиты.
- `image_forgery_cnn/` - код обучения, предсказания и Grad-CAM.
- `image_forgery_cnn_runpod.zip` - архив с кодом для загрузки на RunPod.

## Модель

Используется `EfficientNetB0` с transfer learning:

1. Загружается CNN с предобученными ImageNet-весами.
2. Исходная classifier-head заменяется на бинарный классификатор `original / forged`.
3. Первые эпохи обучается только новая голова.
4. Затем backbone размораживается и выполняется fine-tuning с меньшим learning rate.

## Датасет

Основной датасет: CASIA 2.0 image tampering dataset.

Для запуска примерно на один час на RunPod с RTX PRO 6000 используется ограничение:

- до `5000` оригинальных изображений;
- до `5000` изображений с монтажом;
- вход `384 x 384`;
- `16` эпох;
- `batch-size 96`.

## Быстрый запуск на RunPod

```bash
cd image_forgery_cnn
pip install -r requirements-runpod.txt
bash run_rtx_pro_6000_hour.sh
```

После обучения появятся:

- `best_model.pt` - лучшая модель;
- `latest_model.pt` - последняя модель;
- `history.csv` - история обучения;
- `training_curves.png` - графики обучения;
- `confusion_matrix.png` - матрица ошибок;
- `test_metrics.json` - итоговые метрики.

## Предсказание

```bash
python predict_forgery_cnn.py \
  --checkpoint runs/casia_rtx_pro_6000_hour/best_model.pt \
  --input /path/to/image_or_folder \
  --output-csv runs/casia_rtx_pro_6000_hour/predictions.csv
```

## Интерпретация через Grad-CAM

```bash
python gradcam_forgery_cnn.py \
  --checkpoint runs/casia_rtx_pro_6000_hour/best_model.pt \
  --image /path/to/test_image.jpg \
  --output runs/casia_rtx_pro_6000_hour/gradcam_example.png
```
