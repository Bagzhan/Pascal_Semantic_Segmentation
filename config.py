import torch


NUM_EPOCHS = 60
BATCH_SIZE = 16
LR = 5e-5
T_MAX = 20
ETA_MIN = 1e-6
NUM_CLASSES = 7


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


LAST_CHECKPOINT = "last.pth"
BEST_CHECKPOINT = "best.pth"

