import os
import json
import torch
import matplotlib.pyplot as plt

def get_experiment_folder(base_dir="runs"):
    """Создает папку для нового эксперимента (exp1, exp2, exp3...)"""
    os.makedirs(base_dir, exist_ok=True)
    existing_exps = [d for d in os.listdir(base_dir) if d.startswith("exp")]
    exp_nums = sorted([int(d[3:]) for d in existing_exps if d[3:].isdigit()])
    
    next_exp_num = (exp_nums[-1] + 1) if exp_nums else 1
    exp_dir = os.path.join(base_dir, f"exp{next_exp_num}")
    os.makedirs(exp_dir, exist_ok=True)

    return exp_dir

def save_checkpoint(state, exp_dir, filename="last_checkpoint.pth"):
    """Сохраняет чекпоинт модели"""
    torch.save(state, os.path.join(exp_dir, filename))

def save_plot(data, title, ylabel, filename, exp_dir):
    """Сохраняет график"""
    plt.figure(figsize=(8, 5))
    plt.plot(data, label=title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(exp_dir, filename))
    plt.close()

def save_history(history, exp_dir):
    """Сохраняет историю обучения"""
    with open(os.path.join(exp_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=4)

def save_config(config, exp_dir):
    """Сохраняет конфигурацию"""
    with open(os.path.join(exp_dir, "config.txt"), "w") as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")

