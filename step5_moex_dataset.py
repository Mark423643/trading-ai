"""
Step 5 MOEX v2 — датасет на российских голубых фишках.
Pivot-уровни ищутся на дневных барах (как в сканере).
Touch-события собираются на ЧАСОВЫХ барах (как в сканере).
Новые признаки: RSI(14), EMA20 выше/ниже, час дня, расстояние до соседнего уровня.
Итого 12 признаков. MLP обучается с балансировкой классов (sample_weight).
"""
import pandas as pd
import numpy as np
import os
import time
import requests
import joblib
from datetime import datetime, timedelta
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

DATA_DIR  = "data"
MODEL_DIR = "models"
os.makedirs(os.path.join(DATA_DIR, "moex_tickers"), exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

MOEX_TICKERS = [
    "SBER", "GAZP", "LKOH", "NVTK", "ROSN",
    "GMKN", "VTBR", "RUAL", "MTSS", "YNDX",
    "ALRS", "MGNT", "TATN", "CHMF", "NLMK",
    "PIKK", "MAGN", "AFLT", "MOEX", "IRAO",
    "FEES", "HYDR", "RTKM", "SNGS", "PHOR",
    "AKRN",
]

LOOKBACK   = 10     # pivot lookback (дневные бары)
TOUCH_TOL  = 0.003  # допуск касания (0.3 %)
REACT_BARS = 5      # часовых баров для оценки реакции

MOEX_URL = ("https://iss.moex.com/iss/engines/stock/markets/shares"
            "/boards/TQBR/securities/{ticker}/candles.json")

FEATURES = [
    "level_type", "n_prev_touches", "prev_bounce_rate",
    "prev_avg_reaction", "bars_since_form", "bars_since_last",
    "vol_ratio", "atr_pct",
    "rsi14", "ema_above", "touch_hour", "dist_next_level_atr",
]

# ────────────────────────────────────────────────────
# Загрузка данных с MOEX ISS
# ────────────────────────────────────────────────────
def fetch_moex_candles(ticker, interval, days):
    """interval=24 → дневные, interval=60 → часовые."""
    date_to   = datetime.now().strftime("%Y-%m-%d")
    date_from = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    url    = MOEX_URL.format(ticker=ticker)
    chunks = []
    start  = 0
    while True:
        try:
            r = requests.get(url, params={
                "interval": interval,
                "from":     date_from,
                "till":     date_to,
                "start":    start,
            }, timeout=20)
            r.raise_for_status()
            data = r.json()
            cols = data["candles"]["columns"]
            rows = data["candles"]["data"]
            if not rows:
                break
            chunks.append(pd.DataFrame(rows, columns=cols))
            start += len(rows)
            if len(rows) < 500:
                break
            time.sleep(0.2)
        except Exception as e:
            print(f"  [ошибка {ticker} i={interval}]: {e}")
            break
    if not chunks:
        return None
    df = pd.concat(chunks, ignore_index=True)
    df = df.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume", "begin": "Date",
    })
    df["Date"] = pd.to_datetime(df["Date"])
    return (df.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]]
              .astype(float).sort_index().dropna())

# ────────────────────────────────────────────────────
# Технические индикаторы
# ────────────────────────────────────────────────────
def calc_atr(df, period=14):
    prev = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev).abs(),
        (df["Low"]  - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calc_rsi(series, period=14):
    """Wilder RSI через EWM."""
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - 100 / (1 + rs)

def find_pivots(df, lookback):
    pivots = []
    for i in range(lookback, len(df) - lookback):
        wh = df["High"].iloc[i - lookback:i + lookback + 1]
        wl = df["Low"].iloc[i - lookback:i + lookback + 1]
        if df["High"].iloc[i] == wh.max():
            pivots.append({"date": df.index[i], "level": df["High"].iloc[i], "type": 0})
        if df["Low"].iloc[i] == wl.min():
            pivots.append({"date": df.index[i], "level": df["Low"].iloc[i],  "type": 1})
    return pivots

# ────────────────────────────────────────────────────
# Сбор touch-событий на часовых барах
# ────────────────────────────────────────────────────
def collect_touches(df1d, dfh, pivots, ticker):
    if dfh is None or len(dfh) < 40:
        return []

    # Дневной ATR → маппинг на часовые бары по дате
    daily_atr_series = calc_atr(df1d)
    daily_atr_map    = dict(zip(df1d.index.strftime("%Y-%m-%d"), daily_atr_series))

    dfh = dfh.copy()
    dfh["date_key"]  = dfh.index.strftime("%Y-%m-%d")
    dfh["ATR_daily"] = dfh["date_key"].map(daily_atr_map).ffill()
    dfh["RSI14"]     = calc_rsi(dfh["Close"])
    dfh["EMA20_h"]   = dfh["Close"].ewm(span=20, adjust=False).mean()
    dfh["AvgVol20"]  = dfh["Volume"].rolling(20).mean()

    all_levels = np.array([pv["level"] for pv in pivots])
    records    = []

    for pv in pivots:
        level_price = pv["level"]
        level_type  = pv["type"]
        pivot_date  = pv["date"]

        # Только часовые бары после даты формирования уровня
        dfh_sub = dfh[dfh.index > pivot_date]
        if len(dfh_sub) < REACT_BARS + 1:
            continue

        prev_touches = []
        i = 0

        while i < len(dfh_sub) - REACT_BARS:
            bar       = dfh_sub.iloc[i]
            atr_daily = bar["ATR_daily"]

            if pd.isna(atr_daily) or atr_daily == 0:
                i += 1
                continue

            dist = min(
                abs(bar["Close"] - level_price),
                abs(bar["Low"]   - level_price),
                abs(bar["High"]  - level_price),
            ) / (level_price + 1e-9)

            if dist < TOUCH_TOL:
                future = dfh_sub["Close"].iloc[i + 1:i + 1 + REACT_BARS]
                if len(future) < REACT_BARS:
                    break

                above      = bar["Close"] > level_price
                last_close = future.iloc[-1]
                bounce     = int((above and last_close > level_price) or
                                 (not above and last_close < level_price))

                future_abs   = (future - level_price).abs()
                reaction_atr = future_abs.max() / (atr_daily + 1e-9)

                n_prev            = len(prev_touches)
                prev_bounce_rate  = np.mean([t["bounce"] for t in prev_touches]) if n_prev > 0 else 0.5
                prev_avg_reaction = np.mean([t["reaction_atr"] for t in prev_touches]) if n_prev > 0 else 0.0

                # bars_since_form / bars_since_last в "дневных единицах" (÷8 ч/день)
                bars_since_form = max(1, i // 8)
                bars_since_last = max(1, (i - prev_touches[-1]["bar_idx"]) // 8) if n_prev > 0 else bars_since_form

                avg_vol20 = bar["AvgVol20"]
                vol_ratio = bar["Volume"] / avg_vol20 if (not pd.isna(avg_vol20) and avg_vol20 > 0) else 1.0
                atr_pct   = atr_daily / (level_price + 1e-9)

                # ── Новые признаки ──────────────────────────────
                rsi14      = bar["RSI14"] if not pd.isna(bar["RSI14"]) else 50.0
                ema_above  = int(bar["Close"] > bar["EMA20_h"]) if not pd.isna(bar["EMA20_h"]) else 0
                touch_hour = dfh_sub.index[i].hour

                other_lvls = all_levels[np.abs(all_levels - level_price) > level_price * 0.001]
                if len(other_lvls) > 0 and atr_daily > 0:
                    dist_next = float(np.min(np.abs(other_lvls - bar["Close"])) / atr_daily)
                else:
                    dist_next = 10.0
                dist_next = min(dist_next, 20.0)

                records.append({
                    "ticker":              ticker,
                    "date":                dfh_sub.index[i].strftime("%Y-%m-%d"),
                    "level_type":          level_type,
                    "n_prev_touches":      n_prev,
                    "prev_bounce_rate":    round(prev_bounce_rate, 4),
                    "prev_avg_reaction":   round(prev_avg_reaction, 4),
                    "bars_since_form":     bars_since_form,
                    "bars_since_last":     bars_since_last,
                    "vol_ratio":           round(vol_ratio, 3),
                    "atr_pct":             round(atr_pct, 5),
                    "rsi14":               round(rsi14, 2),
                    "ema_above":           ema_above,
                    "touch_hour":          touch_hour,
                    "dist_next_level_atr": round(dist_next, 3),
                    "bounce":              bounce,
                })

                prev_touches.append({
                    "bar_idx":      i,
                    "bounce":       bounce,
                    "reaction_atr": reaction_atr,
                    "volume":       bar["Volume"],
                })
                i += REACT_BARS
            else:
                i += 1

    return records

# ────────────────────────────────────────────────────
# 1. Загрузка и обработка каждого тикера
# ────────────────────────────────────────────────────
all_records = []

print("=" * 60)
print("  MOEX Dataset Builder v2 — 12 признаков (часовые касания)")
print("=" * 60)

for ticker in MOEX_TICKERS:
    daily_cache  = os.path.join(DATA_DIR, "moex_tickers", f"{ticker.lower()}_daily.csv")
    hourly_cache = os.path.join(DATA_DIR, "moex_tickers", f"{ticker.lower()}_hourly.csv")

    # Дневные данные (для pivot)
    if os.path.exists(daily_cache):
        df1d = pd.read_csv(daily_cache, index_col=0, parse_dates=True)
        df1d = df1d[["Open", "High", "Low", "Close", "Volume"]].astype(float).sort_index().dropna()
    else:
        df1d = fetch_moex_candles(ticker, interval=24, days=730)
        if df1d is None or df1d.empty:
            print(f"[пропуск] {ticker}: нет дневных данных")
            continue
        df1d.to_csv(daily_cache)
        time.sleep(0.3)

    # Часовые данные (для касаний)
    if os.path.exists(hourly_cache):
        dfh = pd.read_csv(hourly_cache, index_col=0, parse_dates=True)
        dfh = dfh[["Open", "High", "Low", "Close", "Volume"]].astype(float).sort_index().dropna()
    else:
        dfh = fetch_moex_candles(ticker, interval=60, days=365)
        if dfh is None or dfh.empty:
            print(f"[пропуск] {ticker}: нет часовых данных")
            continue
        dfh.to_csv(hourly_cache)
        time.sleep(0.4)

    if len(df1d) < 60 or len(dfh) < 40:
        print(f"[пропуск] {ticker}: мало данных (daily={len(df1d)}, hourly={len(dfh)})")
        continue

    pivots  = find_pivots(df1d, LOOKBACK)
    records = collect_touches(df1d, dfh, pivots, ticker)
    all_records.extend(records)

    print(f"{ticker:6s}: daily={len(df1d):4d}, hourly={len(dfh):5d}, "
          f"pivot={len(pivots):3d}, touches={len(records):4d}")

# ────────────────────────────────────────────────────
# 2. Объединённый датасет
# ────────────────────────────────────────────────────
dataset = pd.DataFrame(all_records)
out_csv = os.path.join(DATA_DIR, "touch_events_moex.csv")
dataset.to_csv(out_csv, index=False)

print(f"\n{'='*60}")
print(f"Итого touch-событий: {len(dataset)}")
print(f"  Отбоев  (1): {dataset['bounce'].sum()}")
print(f"  Пробоев (0): {(dataset['bounce']==0).sum()}")
print(f"  Тикеров:     {dataset['ticker'].nunique()}")
print(f"Датасет: {out_csv}")

if len(dataset) < 50:
    print("ВНИМАНИЕ: слишком мало данных для обучения.")

# ────────────────────────────────────────────────────
# 3. Обучение MLP с балансировкой классов
# ────────────────────────────────────────────────────
X = dataset[FEATURES].values
y = dataset["bounce"].values

sample_weights = compute_sample_weight("balanced", y)

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation="relu",
    max_iter=1000,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=30,
)

print(f"\n--- Кросс-валидация (StratifiedKFold k=5, balanced) ---")
min_class = int(dataset["bounce"].value_counts().min())
n_splits  = min(5, min_class)

if n_splits < 2:
    print("Мало примеров — обучаем без CV.")
    mlp.fit(X_scaled, y, sample_weight=sample_weights)
    cv_scores = None
else:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = cross_val_score(mlp, X_scaled, y, cv=cv, scoring="roc_auc")
    print(f"ROC-AUC по фолдам: {cv_scores.round(3)}")
    print(f"Среднее ROC-AUC:   {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    mlp.fit(X_scaled, y, sample_weight=sample_weights)

y_pred = mlp.predict(X_scaled)
y_prob = mlp.predict_proba(X_scaled)[:, 1]
print(f"\n--- Classification Report (train) ---")
print(classification_report(y, y_pred, target_names=["breakout", "bounce"]))
print(f"Train ROC-AUC: {roc_auc_score(y, y_prob):.3f}")

# ────────────────────────────────────────────────────
# 4. Сохранение модели
# ────────────────────────────────────────────────────
joblib.dump(mlp,    os.path.join(MODEL_DIR, "mlp_levels.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
print(f"\nМодель:  {MODEL_DIR}/mlp_levels.pkl")
print(f"Скейлер: {MODEL_DIR}/scaler.pkl")

# ────────────────────────────────────────────────────
# 5. Статистика по тикерам
# ────────────────────────────────────────────────────
print(f"\n--- Touch-события по тикерам ---")
summary = (dataset.groupby("ticker")
           .agg(touches=("bounce", "count"),
                bounces=("bounce", "sum"),
                bounce_rate=("bounce", "mean"))
           .sort_values("touches", ascending=False))
summary["bounce_rate"] = summary["bounce_rate"].round(3)
print(summary.to_string())

print(f"\n--- Средние значения новых признаков ---")
for feat in ["rsi14", "ema_above", "touch_hour", "dist_next_level_atr"]:
    print(f"  {feat:22s}: mean={dataset[feat].mean():.3f}  std={dataset[feat].std():.3f}")
