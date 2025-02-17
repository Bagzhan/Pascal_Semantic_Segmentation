import os
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from config import *
from datasets import AlbumentationsSegmentationDataset, get_dataloaders
from loss import FocalLoss, DiceLoss
from metrics import compute_hierarchical_miou
from models.unetpp import get_model
from utils import get_experiment_folder, save_checkpoint, save_plot, save_history, save_config

device = DEVICE

data_dir = 'data'
images_dir = os.path.join(data_dir, 'JPEGImages')
masks_dir = os.path.join(data_dir, 'gt_masks')
    
train_ids_file = os.path.join(data_dir, 'train_id.txt')
val_ids_file = os.path.join(data_dir, 'val_id.txt')

# Считываем идентификаторы (ожидается, что в файлах указаны имена файлов без расширения)
with open(train_ids_file, 'r') as f:
    train_ids = [line.strip() for line in f if line.strip()]
with open(val_ids_file, 'r') as f:
    val_ids = [line.strip() for line in f if line.strip()]

# Получаем полный список имен изображений в папке (отсортированный)
all_image_names = sorted([f for f in os.listdir(images_dir) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

# Вычисляем индексы: сравниваем базовые имена (без расширения) с train/val id
train_indices = [i for i, name in enumerate(all_image_names) 
                    if os.path.splitext(name)[0] in train_ids]
val_indices = [i for i, name in enumerate(all_image_names) 
                if os.path.splitext(name)[0] in val_ids]

print("Количество обучающих примеров:", len(train_indices))
print("Количество валидационных примеров:", len(val_indices))

# --- Аугментации ---
train_transform = A.Compose([
    A.Resize(height=512, width=512),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(height=512, width=512),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# --- Датасеты и DataLoader'ы ---
train_dataset = AlbumentationsSegmentationDataset(images_dir, masks_dir, transform=train_transform, indices=train_indices)
val_dataset = AlbumentationsSegmentationDataset(images_dir, masks_dir, transform=val_transform, indices=val_indices)
train_loader, val_loader = get_dataloaders(train_dataset, val_dataset, BATCH_SIZE, workers=4)

# --- Инициализация модели ---
model = get_model().to(device)

# --- Функция потерь ---
criterion = lambda outputs, masks: 0.5 * FocalLoss(gamma=2, alpha=torch.tensor([0.1, 1, 1, 1, 1, 1, 1]).to(device))(outputs, masks) + \
                                  0.5 * DiceLoss()(outputs, masks)

optimizer = optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_MAX, eta_min=ETA_MIN)

num_epochs = NUM_EPOCHS
scaler = torch.cuda.amp.GradScaler()
best_miou = 0.0

# --- История обучения ---
history = {
    "train_loss": [],
    "val_loss": [],
    "miou_level0": [],
    "miou_level1": [],
    "miou_level2": []
}

# --- Создание папки эксперимента ---
exp_dir = get_experiment_folder()
# save_config(vars(), exp_dir)

# --- Обучение ---
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    model.train()
    train_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False)
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device).long()
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

    train_loss /= len(train_loader)
    
    # --- Валидация ---
    model.eval()
    val_loss = 0.0
    all_preds, all_trues = [], []

    val_pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}", leave=False)
    with torch.no_grad():
        for images, masks in val_pbar:
            images, masks = images.to(device), masks.to(device).long()

            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)

            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_trues.append(masks.cpu().numpy())

            val_pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

    val_loss /= len(val_loader)
    
    # --- Вычисление метрик ---
    all_preds = np.concatenate(all_preds, axis=0)
    all_trues = np.concatenate(all_trues, axis=0)
    
    hier_metrics = compute_hierarchical_miou(all_preds, all_trues)
    
    print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Val Loss: {val_loss:.4f}")
    print(f"  mIoU Level 0 (body): {hier_metrics['miou_level0']:.4f}")
    print(f"  mIoU Level 1 (upper/lower body): {hier_metrics['miou_level1']:.4f}")
    print(f"  mIoU Level 2 (leaf classes): {hier_metrics['miou_level2']:.4f}")

    # --- Сохранение истории ---
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["miou_level0"].append(hier_metrics["miou_level0"])
    history["miou_level1"].append(hier_metrics["miou_level1"])
    history["miou_level2"].append(hier_metrics["miou_level2"])

    # --- Сохранение чекпоинтов ---
    save_checkpoint({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': val_loss,
        'history': history
    }, exp_dir, filename=LAST_CHECKPOINT)

    if hier_metrics["miou_level2"] > best_miou:
        best_miou = hier_metrics["miou_level2"]
        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
            'history': history,
            'best_miou': best_miou
        }, exp_dir, filename=BEST_CHECKPOINT)
        print("Best model updated and saved.")

    # --- Обновление lr ---
    scheduler.step()

# --- Финальное сохранение истории и графиков ---
save_history(history, exp_dir)
save_plot(history["train_loss"], "Train Loss", "Loss", "train_loss.png", exp_dir)
save_plot(history["val_loss"], "Val Loss", "Loss", "val_loss.png", exp_dir)
save_plot(history["miou_level0"], "mIoU Level 0", "mIoU", "miou_level0.png", exp_dir)
save_plot(history["miou_level1"], "mIoU Level 1", "mIoU", "miou_level1.png", exp_dir)
save_plot(history["miou_level2"], "mIoU Level 2", "mIoU", "miou_level2.png", exp_dir)

print(f"Training completed. Results saved in {exp_dir}")
