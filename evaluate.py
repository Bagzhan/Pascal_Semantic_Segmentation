import torch
from model import get_model
from datasets import get_dataloaders
from config import *


model = get_model().to(DEVICE)
checkpoint = torch.load("runs/expX/best_checkpoint.pth", map_location=DEVICE)
model.load_state_dict(checkpoint)
model.eval()

_, val_loader = get_dataloaders(BATCH_SIZE)

total = 0
correct = 0
with torch.no_grad():
    for images, masks in val_loader:
        images, masks = images.to(DEVICE), masks.to(DEVICE).long()
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == masks).sum().item()
        total += masks.numel()

print(f"Accuracy: {correct / total:.4f}")

