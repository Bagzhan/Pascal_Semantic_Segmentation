import numpy as np

def compute_iou(pred, gt, cls):
    """
    Вычисляет IoU для одного класса.
    """
    intersection = np.logical_and(pred == cls, gt == cls).sum()
    union = np.logical_or(pred == cls, gt == cls).sum()
    if union == 0:
        return np.nan
    return intersection / union

def map_to_level0(mask):
    """
    Уровень 0: объединяем все пиксели, не являющиеся background (0), в один класс (1).
    """
    body_mask = (mask > 0).astype(np.uint8)
    return body_mask

def map_to_level1(mask):
    """
    Уровень 1: разбиваем тело на две части.
    - Классы {1, 2, 4, 6} (low_hand, torso, head, up_hand) -> 0 (upper_body)
    - Классы {3, 5} (low_leg, up_leg) -> 1 (lower_body)
    - background (0) -> -1 (игнорируется)
    """
    mapped = np.full_like(mask, fill_value=-1, dtype=np.int8)
    upper_body = [1, 2, 4, 6]
    lower_body = [3, 5]
    for cls in upper_body:
        mapped[mask == cls] = 0
    for cls in lower_body:
        mapped[mask == cls] = 1
    return mapped

def map_to_level2(mask):
    """
    Уровень 2: детальная сегментация по 6 классам.
    Перенумеровываем классы:
      1 -> 0, 2 -> 1, 3 -> 2, 4 -> 3, 5 -> 4, 6 -> 5.
    Остальные (background) назначаются -1.
    """
    mapping = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
    mapped = np.full_like(mask, fill_value=-1, dtype=np.int8)
    for orig, new in mapping.items():
        mapped[mask == orig] = new
    return mapped

def compute_miou(pred_mask, gt_mask, level):
    """
    Вычисляет mIoU для заданного уровня иерархии.
    
    :param pred_mask: предсказанная маска (numpy array)
    :param gt_mask: истинная маска (numpy array)
    :param level: 0, 1 или 2 (соответствует уровням иерархии)
    :return: mIoU для выбранного уровня
    """
    if level == 0:
        pred_mapped = map_to_level0(pred_mask)
        gt_mapped = map_to_level0(gt_mask)
        # Вычисляем IoU для класса "body" (1)
        return compute_iou(pred_mapped, gt_mapped, 1)
    
    elif level == 1:
        pred_mapped = map_to_level1(pred_mask)
        gt_mapped = map_to_level1(gt_mask)
        iou_list = []
        # Для классов: 0 (upper_body) и 1 (lower_body)
        for cls in [0, 1]:
            # Игнорируем пиксели с меткой -1 (background)
            valid = (gt_mapped != -1)
            if valid.sum() == 0:
                continue
            intersection = np.logical_and(pred_mapped == cls, gt_mapped == cls)[valid].sum()
            union = np.logical_or(pred_mapped == cls, gt_mapped == cls)[valid].sum()
            if union == 0:
                iou_list.append(np.nan)
            else:
                iou_list.append(intersection / union)
        return np.nanmean(iou_list)
    
    elif level == 2:
        pred_mapped = map_to_level2(pred_mask)
        gt_mapped = map_to_level2(gt_mask)
        iou_list = []
        # Для классов от 0 до 5 (6 классов)
        for cls in range(6):
            valid = (gt_mapped != -1)
            if valid.sum() == 0:
                continue
            intersection = np.logical_and(pred_mapped == cls, gt_mapped == cls)[valid].sum()
            union = np.logical_or(pred_mapped == cls, gt_mapped == cls)[valid].sum()
            if union == 0:
                iou_list.append(np.nan)
            else:
                iou_list.append(intersection / union)
        return np.nanmean(iou_list)

def compute_hierarchical_miou(preds, targets):
    """
    Обёртка, которая принимает только предсказанные и истинные маски,
    а затем вычисляет mIoU для трёх уровней:
      - mIoU^0: для объединённого класса body
      - mIoU^1: для верхней и нижней частей тела
      - mIoU^2: для детальной сегментации (6 классов)
    
    :param preds: предсказанная маска (numpy array)
    :param targets: истинная маска (numpy array)
    :return: кортеж (miou_level0, miou_level1, miou_level2)
    """
    miou_level0 = compute_miou(preds, targets, level=0)
    miou_level1 = compute_miou(preds, targets, level=1)
    miou_level2 = compute_miou(preds, targets, level=2)
    return {
        "miou_level0": miou_level0,
        "miou_level1": miou_level1,
        "miou_level2": miou_level2
    }