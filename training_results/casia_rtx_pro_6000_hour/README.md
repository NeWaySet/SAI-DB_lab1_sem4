# Результаты обучения CASIA RTX PRO 6000

Обучение выполнено на подмножестве CASIA 2.0:

- всего изображений: `10000`;
- original: `5000`;
- forged: `5000`;
- train: `7000`;
- validation: `1500`;
- test: `1500`.

## Итоговые метрики на test

| Метрика | Значение |
| --- | ---: |
| Accuracy | `0.8447` |
| Precision forged | `0.8327` |
| Recall forged | `0.8627` |
| F1 forged | `0.8474` |
| ROC-AUC | `0.9225` |
| Loss | `0.4209` |

Матрица ошибок:

|  | Pred original | Pred forged |
| --- | ---: | ---: |
| True original | `620` | `130` |
| True forged | `103` | `647` |

## Файлы

- `run_config.json` - параметры запуска.
- `dataset_summary.json` - информация о датасете и split.
- `history.csv` - история обучения.
- `training_curves.png` - графики обучения.
- `test_metrics.json` - итоговые метрики.
- `confusion_matrix.png` - матрица ошибок.
- `gradcam_example.png` - пример Grad-CAM интерпретации.

Веса модели `best_model.pt` и `latest_model.pt` не добавлены в репозиторий, чтобы не раздувать историю Git. Они находятся в локальном архиве результатов обучения.
