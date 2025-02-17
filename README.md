## Описание проекта

Этот проект реализует **семантическую сегментацию** изображений с использованием **Unet++** и энкодера **EfficientNet-B4**.  
Обучение проводилось на датасете [Pascal-part](https://drive.google.com/file/d/1unIkraozhmsFtkfneZVhw8JMOQ8jv78J/view?usp=sharing).
Обученная модель доступна по ссылке: [exp1](https://drive.google.com/file/d/1glQs_W22PaAuMfY4U8TOwxh7vUYSGtJ4/view?usp=sharing)

### Итоговые результаты после 60 эпох:

- **Train Loss**: `0.2028`
- **Val Loss**: `0.2597`
- **mIoU Level 0 (body)**: `0.8532`
- **mIoU Level 1 (upper/lower body)**: `0.8234`
- **mIoU Level 2 (leaf classes)**: `0.6648`


## Структура проекта

### **Основные директории**
```
├── data
│   ├── classes.txt
│   ├── gt_masks
│   ├── JPEGImages
│   ├── train_id.txt
│   └── val_id.txt
├── models
│   └── unetpp.py
├── runs
│   └── exp1
│       ├── best.pth
│       ├── history.json
│       ├── last.pth
│       ├── miou_level0.png
│       ├── miou_level1.png
│       ├── miou_level2.png
│       ├── train_loss.png
│       └── val_loss.png
├── config.py
├── datasets.py
├── evaluate.py
├── train.py
├── utils.py
├── LICENSE
├── loss.py
├── metrics.py
├── predict.py
├── Test_assignment.md
├── requirements.txt
```


## **Запуск проекта**

### Установка зависимостей
Перед запуском установите необходимые библиотеки:

```
pip install -r requirements.txt
```

### Обучение модели
Запуск обучения на train.py:

```
python train.py
```

### Оценка модели
Оценить качество модели на валидационных данных:

```
python evaluate.py
```

### Предсказания на новых изображениях
Запуск предсказания:
```
python predict.py --model_path runs/exp1/best.pth --image_path data/test.jpg --output_path results/predicted_mask.png
```


## **Технические детали**

- **Модель**: Unet++ с энкодером EfficientNet-B4.
- **Функция потерь**: Комбинация Dice Loss + Focal Loss.
- **Оптимизатор**: AdamW.
- **LR Scheduler**: Cosine Annealing LR.
- **Размер входных изображений**: 512x512.


## **Возможные улучшения**

- **Добавление аугментации данных** для повышения обобщающей способности модели.
- **Эксперименты с различными энкодерами** (Swin Transformer, ResNet).
- **Внедрение post-processing** (CRF, медианная фильтрация).
- **Использование semi-supervised learning**.

