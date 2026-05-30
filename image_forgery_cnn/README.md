# CNN для выявления фальсификации и монтажа изображений

Проект обучает бинарный классификатор `original / forged` на базе `EfficientNetB0` с трансферным обучением. Он подходит для защиты задания: есть заморозка backbone, fine-tuning, метрики, матрица ошибок, сохранение checkpoint и Grad-CAM.

## Целевой запуск: RTX PRO 6000

Основной пресет рассчитан на RunPod с NVIDIA RTX PRO 6000 Blackwell. У этой карты большой запас видеопамяти, поэтому вместо короткого учебного запуска на 2000 изображений используется более содержательный режим примерно на один час:

- Датасет: `divg07/casia-20-image-tampering-detection-dataset`
- Ограничение: `--max-per-class 5000`, то есть до 10000 изображений суммарно.
- Размер входа: `384 x 384`, чтобы модель видела больше локальных следов монтажа.
- Эпохи: `16`, первые `2` эпохи обучается только классификационная голова.
- Batch size: `96`.
- Смешанная точность включена по умолчанию.

Фактическое время зависит от скорости диска, загрузки датасета Kaggle и конкретного образа RunPod. Если обучение идет заметно быстрее часа, увеличьте `--epochs` до `20`. Если дольше, уменьшите `--max-per-class` до `3500`.

## Установка на RunPod

```bash
cd image_forgery_cnn
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-runpod.txt
```

## Обучение

Готовая команда:

```bash
bash run_rtx_pro_6000_hour.sh
```

То же самое явно:

```bash
python train_forgery_cnn.py \
  --dataset-slug divg07/casia-20-image-tampering-detection-dataset \
  --output-dir runs/casia_rtx_pro_6000_hour \
  --max-per-class 5000 \
  --epochs 16 \
  --freeze-epochs 2 \
  --batch-size 96 \
  --image-size 384 \
  --num-workers 12 \
  --input-mode rgb
```

Если датасет уже скачан вручную:

```bash
python train_forgery_cnn.py \
  --data-dir /workspace/CASIA2 \
  --output-dir runs/casia_rtx_pro_6000_hour \
  --max-per-class 5000 \
  --epochs 16 \
  --freeze-epochs 2 \
  --batch-size 96 \
  --image-size 384
```

Ожидаемая структура может быть `Au/Tp`, `original/forged`, `authentic/tampered` или `real/fake`. Маски и папки `ground truth` автоматически пропускаются.

## Быстрая проверка

Для проверки пайплайна без Kaggle можно создать маленький synthetic demo dataset:

```bash
python create_demo_dataset.py --output demo_data --per-class 64
python train_forgery_cnn.py \
  --data-dir demo_data \
  --output-dir runs/demo_smoke \
  --epochs 2 \
  --freeze-epochs 0 \
  --batch-size 16 \
  --max-per-class 64 \
  --image-size 224 \
  --no-pretrained \
  --num-workers 0
```

Для очень быстрого реального датасета можно использовать:

```bash
python train_forgery_cnn.py \
  --dataset-slug prajnar3/image-forgery-detection-dataset-splicing \
  --output-dir runs/splicing_quick \
  --max-per-class 120 \
  --epochs 5 \
  --freeze-epochs 1 \
  --batch-size 32 \
  --image-size 224
```

## Предсказание

```bash
python predict_forgery_cnn.py \
  --checkpoint runs/casia_rtx_pro_6000_hour/best_model.pt \
  --input /path/to/image_or_folder \
  --output-csv runs/casia_rtx_pro_6000_hour/predictions.csv
```

## Grad-CAM

```bash
python gradcam_forgery_cnn.py \
  --checkpoint runs/casia_rtx_pro_6000_hour/best_model.pt \
  --image /path/to/test_image.jpg \
  --output runs/casia_rtx_pro_6000_hour/gradcam_example.png
```

## Что сохраняется после обучения

- `best_model.pt` и `latest_model.pt` - веса модели и параметры препроцессинга.
- `train_split.csv`, `val_split.csv`, `test_split.csv` - разбиение датасета.
- `history.csv` и `training_curves.png` - динамика обучения.
- `test_metrics.json` - accuracy, precision, recall, F1, ROC-AUC, confusion matrix.
- `confusion_matrix.png` - матрица ошибок для отчета.

## Логика модели

`EfficientNetB0` выбран как компактная CNN-архитектура: она быстрее тяжелых ResNet-вариантов, хорошо переносит ImageNet-признаки на прикладную задачу и оставляет запас GPU-памяти для входа `384 x 384`. Первые эпохи обучается только классификационная голова, затем backbone размораживается и дообучается с меньшим learning rate.
