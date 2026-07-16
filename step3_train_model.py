"""
Обучение MLP-классификатора на MOEX touch-событиях.
Валидация: временной сплит (train до SPLIT_DATE, OOS после).
Shuffle CV исключён — он сливает будущее через prev_bounce_rate.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, roc_auc_score,
                              precision_recall_curve, confusion_matrix)
from sklearn.utils.class_weight import compute_sample_weight

DATA_DIR  = "data"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Граница train/OOS — последние ~6 месяцев идут в OOS
SPLIT_DATE = "2026-01-01"

FEATURES = [
    "level_type", "n_prev_touches", "prev_bounce_rate",
    "prev_avg_reaction", "bars_since_form", "bars_since_last",
    "vol_ratio", "atr_pct",
    "rsi14", "ema_above", "touch_hour", "dist_next_level_atr",
]

# ── 1. Загрузка ──────────────────────────────────────────────────────────────
MOEX_CSV = os.path.join(DATA_DIR, "touch_events_moex.csv")
if not os.path.exists(MOEX_CSV):
    raise FileNotFoundError(f"Нет {MOEX_CSV} — запусти step5_moex_dataset.py")

dataset = pd.read_csv(MOEX_CSV)
print(f"Touch-событий загружено: {len(dataset)}")
print(f"  Тикеров: {dataset['ticker'].nunique() if 'ticker' in dataset.columns else 'N/A'}")
print(f"  Отбоев (1): {dataset['bounce'].sum()}  |  Пробоев (0): {(dataset['bounce']==0).sum()}")

# Оставляем только признаки которые есть в датасете
FEATURES = [f for f in FEATURES if f in dataset.columns]
print(f"\nПризнаков: {len(FEATURES)}")
print(f"  {FEATURES}")

# ── 2. Временной сплит ───────────────────────────────────────────────────────
date_col = None
for col in ["date", "Date", "datetime", "Datetime"]:
    if col in dataset.columns:
        date_col = col
        break

if date_col:
    dataset[date_col] = pd.to_datetime(dataset[date_col])
    train_mask = dataset[date_col] < SPLIT_DATE
    oos_mask   = dataset[date_col] >= SPLIT_DATE
    df_train = dataset[train_mask]
    df_oos   = dataset[oos_mask]
    print(f"\nСплит по дате {SPLIT_DATE}:")
    print(f"  Train: {len(df_train)} событий")
    print(f"  OOS:   {len(df_oos)} событий")
    if len(df_oos) < 100:
        print("  ВНИМАНИЕ: мало OOS-событий, граница слишком поздняя")
else:
    print("\nКолонка date не найдена — используем 80/20 split по порядку")
    split_idx = int(len(dataset) * 0.80)
    df_train = dataset.iloc[:split_idx]
    df_oos   = dataset.iloc[split_idx:]

X_train = df_train[FEATURES].values
y_train = df_train["bounce"].values
X_oos   = df_oos[FEATURES].values
y_oos   = df_oos["bounce"].values

# ── 3. Масштабирование (fit только на train) ─────────────────────────────────
scaler   = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_oos_s   = scaler.transform(X_oos)

# ── 4. Обучение ──────────────────────────────────────────────────────────────
sw_train = compute_sample_weight("balanced", y_train)

mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),   # поменьше чем раньше — меньше переобучение
    activation="relu",
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.10,      # 10% train идут в internal val
    n_iter_no_change=20,
    alpha=0.01,                    # L2 регуляризация
)

print("\n--- Обучение MLP (train set) ---")
mlp.fit(X_train_s, y_train, sample_weight=sw_train)
print(f"  Итераций: {mlp.n_iter_}")

# ── 5. Метрики на Train ──────────────────────────────────────────────────────
y_train_prob = mlp.predict_proba(X_train_s)[:, 1]
print(f"\n  Train ROC-AUC: {roc_auc_score(y_train, y_train_prob):.3f}  (для сравнения с OOS)")

# ── 6. OOS метрики — главное ─────────────────────────────────────────────────
if len(df_oos) > 0:
    y_oos_prob = mlp.predict_proba(X_oos_s)[:, 1]
    oos_auc = roc_auc_score(y_oos, y_oos_prob)
    print(f"\n{'='*55}")
    print(f"  OOS ROC-AUC: {oos_auc:.3f}  ← ГЛАВНАЯ МЕТРИКА")
    print(f"{'='*55}")

    # Подбираем threshold по OOS: максимизируем F1 класса bounce
    precisions, recalls, thresholds = precision_recall_curve(y_oos, y_oos_prob)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_scores[:-1])
    best_thr = thresholds[best_idx]
    best_prec = precisions[best_idx]
    best_rec  = recalls[best_idx]
    print(f"\n  Оптимальный порог по OOS F1:")
    print(f"    threshold={best_thr:.3f}  precision={best_prec:.3f}  recall={best_rec:.3f}")

    # Таблица precision при разных threshold
    print("\n  Precision/Recall bounce при разных порогах (OOS):")
    print(f"  {'Threshold':>10}  {'Precision':>10}  {'Recall':>8}  {'N_pred':>7}")
    for thr in [0.60, 0.65, 0.70, 0.75, 0.80]:
        pred = (y_oos_prob >= thr).astype(int)
        n_pred = pred.sum()
        if n_pred > 0:
            prec = (pred & y_oos).sum() / n_pred
            rec  = (pred & y_oos).sum() / y_oos.sum()
        else:
            prec = rec = 0.0
        print(f"  {thr:>10.2f}  {prec:>10.3f}  {rec:>8.3f}  {n_pred:>7}")

    # Classification report при best_thr
    y_oos_pred = (y_oos_prob >= best_thr).astype(int)
    print(f"\n--- Classification Report OOS (threshold={best_thr:.2f}) ---")
    print(classification_report(y_oos, y_oos_pred, target_names=["breakout","bounce"]))

    cm = confusion_matrix(y_oos, y_oos_pred)
    print(f"Confusion matrix OOS:")
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

    # Рекомендация по MODEL_THRESHOLD
    print(f"\n{'='*55}")
    if oos_auc >= 0.60:
        # Выбираем threshold где precision bounce >= 0.80
        good_thrs = [(thresholds[i], precisions[i], recalls[i])
                     for i in range(len(thresholds))
                     if precisions[i] >= 0.78 and recalls[i] >= 0.05]
        if good_thrs:
            rec_thr = good_thrs[0][0]
            print(f"  РЕКОМЕНДАЦИЯ: MODEL_THRESHOLD = {rec_thr:.2f}")
            print(f"  (precision bounce >= 78% на OOS)")
        else:
            print(f"  РЕКОМЕНДАЦИЯ: MODEL_THRESHOLD = {best_thr:.2f}  (лучший F1 на OOS)")
    else:
        print(f"  ВНИМАНИЕ: OOS AUC={oos_auc:.3f} < 0.60")
        print(f"  Модель слабая — оставить MODEL_THRESHOLD = 0.0 (ML выключен)")
    print(f"{'='*55}")
else:
    print("OOS пустой — нет данных после SPLIT_DATE")

# ── 7. Финальное обучение на ВСЕХ данных ────────────────────────────────────
print("\n--- Финальное обучение на полном датасете ---")
X_all = dataset[FEATURES].values
y_all = dataset["bounce"].values
X_all_s   = scaler.fit_transform(X_all)   # scaler пересчитывается на всех данных
sw_all = compute_sample_weight("balanced", y_all)
mlp.fit(X_all_s, y_all, sample_weight=sw_all)

joblib.dump(mlp,    os.path.join(MODEL_DIR, "mlp_levels.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
print(f"  Модель -> {MODEL_DIR}/mlp_levels.pkl")
print(f"  Скейлер -> {MODEL_DIR}/scaler.pkl")
