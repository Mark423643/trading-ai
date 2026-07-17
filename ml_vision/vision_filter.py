"""
vision_filter.py — интеграция Vision-модели (ResNet18) в живой сканер бота.

Идея: механический сигнал (уровень + ЛП/пробой) сначала проходит через
классификатор, обученный на реальных сделках Тимура, и только затем —
через Approval Mode (человек в NTFY). Vision-фильтр — дополнительное сито
ДО человека, не замена ему.

ВАЖНО (см. ml_vision/train_model.py): текущая модель обучена на ~10
уникальных сделках — это демонстрационный прототип, а не рабочий фильтр.
Не включай VisionFilter в реальный отсев сигналов, пока val-выборка не
вырастет минимум до нескольких десятков уникальных примеров на класс.

Использование:
    from vision_filter import VisionFilter
    vf = VisionFilter()
    result = vf.predict("GAZP", "2026-07-16 14:00", 150.0, "LONG")
    # {'decision': 'ENTRY'|'NO_ENTRY', 'confidence': 0-100, 'class_probs': {...}, 'screenshot_path': ...}
    if vf.is_entry("GAZP", "2026-07-16 14:00", 150.0, "LONG", threshold=0.7):
        ...
"""
import os
import sys

import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from screenshot_chart import load_ticker_df, slice_window, render_chart  # noqa: E402

DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pt")
PRED_DIR = os.path.join(BASE_DIR, "data", "predictions")

DIRECTION_TO_CLASS = {"LONG": "LONG_ENTRY", "SHORT": "SHORT_ENTRY"}

_EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class VisionFilter:
    """Загружает обученную ResNet18 и классифицирует скриншот сигнала."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self.device = torch.device("cpu")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Модель не найдена: {model_path}. "
                f"Сначала запусти ml_vision/train_model.py"
            )

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.classes = checkpoint["classes"]          # напр. ['LONG_ENTRY','NO_ENTRY','SHORT_ENTRY']
        self.img_size = checkpoint.get("img_size", 224)

        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(self.classes))
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        self.model = model.to(self.device)

        os.makedirs(PRED_DIR, exist_ok=True)

    def _screenshot(self, ticker: str, bar_time: str, level: float, direction: str) -> str:
        """Рендерит M5-график вокруг сигнала (100 баров, тёмная тема) и
        возвращает путь к PNG."""
        df = load_ticker_df(ticker, "M5")
        window = slice_window(df, bar_time, bars_before=99, bars_after=0)
        if len(window) == 0:
            raise ValueError(f"Пустое окно данных для {ticker} @ {bar_time}")

        ts_str = str(bar_time).replace(" ", "_").replace(":", "")
        fname = f"{ticker.upper()}_{ts_str}_{direction}_pred.png"
        out_path = os.path.join(PRED_DIR, fname)

        render_chart(window, out_path, level=level,
                     title=f"{ticker.upper()} M5 — {direction} @ {level}")
        return out_path

    def predict(self, ticker: str, bar_time: str, level: float, direction: str,
                keep_screenshot: bool = True) -> dict:
        """
        1. Делает скриншот графика вокруг сигнала.
        2. Прогоняет через ResNet18.
        3. Возвращает решение по целевому классу (LONG_ENTRY/SHORT_ENTRY,
           в зависимости от direction) и вероятности по всем трём классам.
        """
        direction = direction.upper()
        if direction not in DIRECTION_TO_CLASS:
            raise ValueError(f"direction должен быть 'LONG' или 'SHORT', получено: {direction}")
        target_class = DIRECTION_TO_CLASS[direction]

        screenshot_path = self._screenshot(ticker, bar_time, level, direction)

        img = Image.open(screenshot_path).convert("RGB")
        x = _EVAL_TRANSFORM(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1).squeeze(0)

        class_probs = {cls: float(probs[i]) * 100.0 for i, cls in enumerate(self.classes)}
        pred_idx = int(torch.argmax(probs))
        pred_class = self.classes[pred_idx]
        pred_conf = class_probs[pred_class]

        decision = "ENTRY" if pred_class == target_class else "NO_ENTRY"

        if not keep_screenshot:
            try:
                os.remove(screenshot_path)
            except OSError:
                pass

        return {
            "decision": decision,
            "confidence": round(pred_conf, 2),
            "predicted_class": pred_class,
            "target_class": target_class,
            "class_probs": {k: round(v, 2) for k, v in class_probs.items()},
            "screenshot_path": screenshot_path if keep_screenshot else None,
        }

    def is_entry(self, ticker: str, bar_time: str, level: float, direction: str,
                 threshold: float = 0.7) -> bool:
        """True, если модель уверенно (>threshold) предсказала целевой класс
        (LONG_ENTRY/SHORT_ENTRY по direction)."""
        result = self.predict(ticker, bar_time, level, direction, keep_screenshot=False)
        if result["decision"] != "ENTRY":
            return False
        return (result["confidence"] / 100.0) > threshold


if __name__ == "__main__":
    print("[TEST] VisionFilter — самопроверка...")
    vf = VisionFilter()
    print(f"[TEST] Модель загружена: {vf.model_path}")
    print(f"[TEST] Классы: {vf.classes}")

    result = vf.predict("GAZP", "2026-07-16 14:00", 150.0, "LONG")
    print(f"[TEST] predict(): {result}")

    entry = vf.is_entry("GAZP", "2026-07-16 14:00", 150.0, "LONG", threshold=0.7)
    print(f"[TEST] is_entry() (threshold=0.7): {entry}")
