"""
Обучение ResNet18 на скриншотах сделок Тимура: LONG_ENTRY / SHORT_ENTRY / NO_ENTRY.
Работает на CPU (без GPU). Вход: ml_vision/data/{train,val}/<класс>/*.jpg|png
Выход: ml_vision/models/model.pt

ВНИМАНИЕ: на момент написания в train/ всего 25 файлов (по факту ~10 уникальных
сделок, дублированных со скриншотов с разных устройств), val/ — пусто. ResNet18
на такой выборке гарантированно переобучится, а accuracy на val ничего не будет
значить статистически. Скрипт написан заранее ("когда придут данные — сразу
запустим"), но реальное обучение имеет смысл начинать от сотен размеченных
примеров на класс.
"""
import os, sys, time, argparse, random
sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

BASE_DIR   = "/root/trading/AI/ml_vision"
TRAIN_DIR  = os.path.join(BASE_DIR, "data", "train")
VAL_DIR    = os.path.join(BASE_DIR, "data", "val")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pt")

CLASSES    = ["LONG_ENTRY", "NO_ENTRY", "SHORT_ENTRY"]  # алфавитный порядок ImageFolder
IMG_SIZE   = 224
BATCH_SIZE = 8
EPOCHS     = 50
LR         = 1e-4
SEED       = 42
VAL_FALLBACK_FRAC = 0.2  # если val/ пуста — берём эту долю из train для оценки

random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs(MODEL_DIR, exist_ok=True)


def build_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def count_images(root):
    if not os.path.isdir(root):
        return 0
    n = 0
    for cls in os.listdir(root):
        cdir = os.path.join(root, cls)
        if os.path.isdir(cdir):
            n += len([f for f in os.listdir(cdir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=LR)
    args = ap.parse_args()

    device = torch.device("cpu")
    print(f"Устройство: {device} (CPU, GPU не используется)")

    n_train_raw = count_images(TRAIN_DIR)
    n_val_raw = count_images(VAL_DIR)
    print(f"Найдено изображений: train={n_train_raw}, val={n_val_raw}")

    if n_train_raw == 0:
        print("ОШИБКА: в data/train/ нет изображений. Сначала разложи скриншоты по классам.")
        sys.exit(1)

    train_ds_full = datasets.ImageFolder(TRAIN_DIR, transform=build_transforms(train=True))
    print(f"Классы (по ImageFolder): {train_ds_full.classes}")

    if n_val_raw == 0:
        print(f"\n[WARN] data/val/ пуста — отделяю {VAL_FALLBACK_FRAC:.0%} из train "
              f"для оценки (fallback, НЕ полноценная валидация).")
        idx = list(range(len(train_ds_full)))
        random.shuffle(idx)
        n_val = max(1, int(len(idx) * VAL_FALLBACK_FRAC))
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]

        # для val нужен отдельный датасет без train-аугментаций
        val_ds_full = datasets.ImageFolder(TRAIN_DIR, transform=build_transforms(train=False))

        train_ds = Subset(train_ds_full, train_idx)
        val_ds = Subset(val_ds_full, val_idx)
        classes = train_ds_full.classes
    else:
        val_ds_full = datasets.ImageFolder(VAL_DIR, transform=build_transforms(train=False))
        train_ds = train_ds_full
        val_ds = val_ds_full
        classes = train_ds_full.classes

    print(f"Обучение: {len(train_ds)} изображений  |  Оценка: {len(val_ds)} изображений")
    if len(train_ds) < 30:
        print("[WARN] Меньше 30 обучающих изображений — ResNet18 переобучится почти "
              "гарантированно. Метрики ниже носят демонстрационный характер.")

    train_loader = DataLoader(train_ds, batch_size=min(args.batch_size, max(1, len(train_ds))),
                               shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=min(args.batch_size, max(1, len(val_ds))),
                             shuffle=False, num_workers=0)

    # ── Модель: ResNet18, предобученные веса ImageNet, замена головы под 3 класса ──
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
    except Exception as e:
        print(f"[WARN] не удалось скачать предобученные веса ({e}), учу с нуля.")
        model = models.resnet18(weights=None)

    num_classes = len(classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    best_val_acc = -1.0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, running_correct, running_n = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            running_correct += (out.argmax(1) == labels).sum().item()
            running_n += imgs.size(0)
        scheduler.step()

        train_loss = running_loss / max(1, running_n)
        train_acc = running_correct / max(1, running_n)

        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            model.eval()
            val_correct, val_n = 0, 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    out = model(imgs)
                    val_correct += (out.argmax(1) == labels).sum().item()
                    val_n += imgs.size(0)
            val_acc = val_correct / max(1, val_n)
            print(f"Эпоха {epoch:3d}/{args.epochs}  loss={train_loss:.4f}  "
                  f"train_acc={train_acc:.3f}  val_acc={val_acc:.3f}")
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                torch.save({"model_state": model.state_dict(),
                            "classes": classes,
                            "img_size": IMG_SIZE}, MODEL_PATH)

    dt = time.time() - t0
    print(f"\nОбучение завершено за {dt:.0f}с.")
    print(f"Лучшая val_acc: {best_val_acc:.3f}  (на {len(val_ds)} изображениях)")
    print(f"Модель сохранена: {MODEL_PATH}")

    if len(val_ds) < 20:
        print("\n[ВАЖНО] val-выборка меньше 20 изображений — accuracy статистически "
              "ненадёжна (доверительный интервал огромен). Не считай эту цифру "
              "финальным качеством модели, пока не наберётся полноценный val/ "
              "(десятки-сотни изображений на класс).")


if __name__ == "__main__":
    main()
