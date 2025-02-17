import torch
import argparse
import numpy as np
import cv2
import os
from models.unetpp import get_model
from torchvision import transforms
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = (512, 512)

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0).to(DEVICE)

def predict(model, image_path, output_path, class_colors=None):
    model.eval()
    image = preprocess_image(image_path)

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            output = model(image)
            pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    pred_mask_colored = create_colored_mask(pred_mask, class_colors)
    cv2.imwrite(output_path, pred_mask_colored)

def create_colored_mask(mask, class_colors=None):
    if class_colors is None:
        class_colors = {
            0: (0, 0, 0),
            1: (255, 0, 0),
            2: (0, 255, 0),
            3: (0, 0, 255),
            4: (255, 255, 0),
            5: (255, 0, 255),
            6: (0, 255, 255)
        }

    h, w = mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_idx, color in class_colors.items():
        colored_mask[mask == class_idx] = color

    return colored_mask

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)

    args = parser.parse_args()

    model = get_model(num_classes=7).to(DEVICE)
    checkpoint = torch.load(args.model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    predict(model, args.image_path, args.output_path)
