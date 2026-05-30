# Image Forgery CNN

Рабочая нейросеть для выявления признаков фальсификации или монтажа на цифровых изображениях.

Модель классифицирует изображение как:

- `original` - оригинальное изображение;
- `forged` - изображение с признаками монтажа, вставки, copy-move или другой локальной фальсификации.

## Состав проекта

| Файл | Назначение |
| --- | --- |
| `train_forgery_cnn.py` | обучение EfficientNetB0, расчет метрик, сохранение checkpoint |
| `predict_forgery_cnn.py` | предсказание для одного изображения или папки |
| `gradcam_forgery_cnn.py` | Grad-CAM визуализация решения модели |
| `create_demo_dataset.py` | генерация маленького synthetic dataset для smoke-test |
| `requirements-runpod.txt` | зависимости для RunPod |
| `run_rtx_pro_6000_hour.sh` | готовый запуск примерно на час для RTX PRO 6000 |

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-runpod.txt
```

Если `kagglehub` на RunPod падает из-за несовместимой версии `kagglesdk`, скрипт автоматически переключится на обычный `kaggle` CLI. Для этого должен быть доступ к Kaggle API.

Можно задать токен так:

```bash
export KAGGLE_API_TOKEN="your_token"
```

Или сохранить токен в файл:

```bash
mkdir -p ~/.kaggle
echo "your_token" > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```

Для старого формата `kaggle.json` также подходят переменные:

```bash
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_key"
```

На Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements-runpod.txt
```

## Основное обучение на RTX PRO 6000

```bash
bash run_rtx_pro_6000_hour.sh
```

Команда внутри скрипта:

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

Если обучение идет слишком быстро, можно увеличить `--epochs` до `20`. Если обучение занимает больше часа, можно уменьшить `--max-per-class` до `3500`.

## Локальный датасет

Если CASIA уже скачан вручную:

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

Скрипт понимает структуры папок:

- `Au/Tp`;
- `original/forged`;
- `authentic/tampered`;
- `real/fake`.

Папки с масками и `ground truth` автоматически пропускаются.

## Быстрая проверка без Kaggle

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

## Артефакты после обучения

В папке запуска сохраняются:

- `best_model.pt`;
- `latest_model.pt`;
- `train_split.csv`, `val_split.csv`, `test_split.csv`;
- `history.csv`;
- `training_curves.png`;
- `confusion_matrix.png`;
- `test_metrics.json`.

## Почему EfficientNetB0

EfficientNetB0 выбран как компактная CNN-архитектура с хорошим балансом качества и скорости. Она быстрее тяжелых ResNet-вариантов, поддерживает transfer learning и при входе `384 x 384` позволяет лучше сохранить локальные признаки монтажа: границы вставки, отличия текстуры, следы сжатия и локальные артефакты.
