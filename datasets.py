import os
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset

class AlbumentationsSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, indices=None):
        """
        Args:
            images_dir (str): Путь к папке с изображениями (JPEG).
            masks_dir (str): Путь к папке с масками (.npy).
            transform (albumentations.Compose, optional): Трансформации, применяемые к изображению и маске.
            indices (list, optional): Список индексов для выбора части датасета.
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        self.image_names = sorted([f for f in os.listdir(images_dir) 
                                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        self.mask_names = sorted([os.path.splitext(f)[0] + '.npy' for f in self.image_names])
        
        if len(self.image_names) == 0:
            raise ValueError("В указанной папке не найдено изображений.")
        if len(self.image_names) != len(self.mask_names):
            raise ValueError("Количество изображений и масок не совпадает!")
        
        if indices is not None:
            self.image_names = [self.image_names[i] for i in indices]
            self.mask_names = [self.mask_names[i] for i in indices]
            
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_names[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_names[idx])
        
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError("Не удалось загрузить изображение: " + img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = np.load(mask_path)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            mask = torch.from_numpy(mask).long()
            
        return image, mask

# --- Функция для создания DataLoader'ов ---
def get_dataloaders(train_dataset, val_dataset, batch_size, workers):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    return train_loader, val_loader