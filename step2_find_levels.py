import pandas as pd
import numpy as np
import os

DATA_DIR = "data"
LOOKBACK = 10        # баров для определения локального хая/лоя
TOUCH_TOL = 0.003   # 0.3% — расстояние до уровня считается касанием
REACTION_BARS = 5   # баров после касания для оценки реакции

# ───────────────────────────────────────────────
# 1. Загрузка данных
# ───────────────────────────────────────────────
df = pd.read_csv(os.path.join(DATA_DIR, "mstr_daily.csv"), header=[0, 1], index_col=0)
df.index = pd.to_datetime(df.index)

# Убираем мультииндекс колонок (Price/Ticker → плоские имена)
df.columns = [col[0] for col in df.columns]
df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
df = df.sort_index().dropna()

print(f"Дневных баров загружено: {len(df)}")

# ───────────────────────────────────────────────
# 2. ATR(14) дневной
# ───────────────────────────────────────────────
high, low, close = df["High"], df["Low"], df["Close"]
prev_close = close.shift(1)
tr = pd.concat([
    high - low,
    (high - prev_close).abs(),
    (low - prev_close).abs()
], axis=1).max(axis=1)
df["ATR14"] = tr.rolling(14).mean()

# ───────────────────────────────────────────────
# 3. Поиск pivot-уровней (локальные хаи и лои)
# ───────────────────────────────────────────────
def find_pivots(df, lookback):
    pivots = []
    for i in range(lookback, len(df) - lookback):
        window_high = df["High"].iloc[i - lookback:i + lookback + 1]
        window_low  = df["Low"].iloc[i - lookback:i + lookback + 1]

        if df["High"].iloc[i] == window_high.max():
            pivots.append({
                "bar_idx": i,
                "date": df.index[i],
                "level": df["High"].iloc[i],
                "type": "resistance"
            })
        if df["Low"].iloc[i] == window_low.min():
            pivots.append({
                "bar_idx": i,
                "date": df.index[i],
                "level": df["Low"].iloc[i],
                "type": "support"
            })
    return pd.DataFrame(pivots)

pivots = find_pivots(df, LOOKBACK)
print(f"Pivot-уровней найдено: {len(pivots)}")

# ───────────────────────────────────────────────
# 4. Признаки силы для каждого уровня
# ───────────────────────────────────────────────
def analyze_level(level_price, pivot_bar, df, tol=TOUCH_TOL, react_bars=REACTION_BARS):
    """
    Для уровня считаем касания ПОСЛЕ его формирования.
    Касание: цена подошла в пределах tol% от уровня.
    Реакция: движение цены за react_bars баров после касания.
    Отбой vs пробой: закрытие через react_bars баров по ту же сторону = отбой.
    """
    touches = []
    i = pivot_bar + 1  # анализируем только бары после формирования уровня

    while i < len(df) - react_bars:
        bar = df.iloc[i]
        dist = abs(bar["Close"] - level_price) / level_price

        # Касание: цена подошла к уровню
        touched = (dist < tol or
                   abs(bar["Low"] - level_price) / level_price < tol or
                   abs(bar["High"] - level_price) / level_price < tol)

        if touched:
            atr = df["ATR14"].iloc[i]
            future = df["Close"].iloc[i + 1:i + 1 + react_bars]

            if len(future) < react_bars or atr == 0 or np.isnan(atr):
                i += 1
                continue

            # Реакция: макс отклонение от уровня за следующие react_bars баров
            reaction_pct = (future - level_price).abs().max() / atr

            # Сторона при касании
            above = bar["Close"] > level_price

            # Отбой: цена ушла в сторону от уровня и не вернулась за react_bars
            last_close = future.iloc[-1]
            bounce = (above and last_close > level_price) or (not above and last_close < level_price)

            touches.append({
                "bar_idx": i,
                "date": df.index[i],
                "volume": bar["Volume"],
                "reaction_atr": round(reaction_pct, 3),
                "bounce": int(bounce),
                "atr": atr,
            })
            i += react_bars  # пропускаем реакцию, чтобы не считать одно касание дважды
        else:
            i += 1

    return touches

records = []
for _, row in pivots.iterrows():
    touches = analyze_level(row["level"], row["bar_idx"], df)

    if not touches:
        n_touches = 0
        avg_reaction = 0.0
        avg_volume = 0.0
        avg_gap_bars = 0.0
        n_bounces = 0
        n_breakouts = 0
        bounce_rate = 0.0
    else:
        t_df = pd.DataFrame(touches)
        n_touches = len(t_df)
        avg_reaction = t_df["reaction_atr"].mean()
        avg_volume = t_df["volume"].mean()
        n_bounces = t_df["bounce"].sum()
        n_breakouts = n_touches - n_bounces
        bounce_rate = n_bounces / n_touches if n_touches > 0 else 0

        idxs = t_df["bar_idx"].values
        avg_gap_bars = float(np.diff(idxs).mean()) if len(idxs) > 1 else 0.0

    records.append({
        "date": row["date"],
        "level": round(row["level"], 4),
        "type": row["type"],
        "n_touches": n_touches,
        "avg_reaction_atr": round(avg_reaction, 3),
        "avg_volume": round(avg_volume, 0),
        "avg_gap_bars": round(avg_gap_bars, 1),
        "n_bounces": n_bounces,
        "n_breakouts": n_breakouts,
        "bounce_rate": round(bounce_rate, 3),
    })

levels_df = pd.DataFrame(records)
levels_df = levels_df.sort_values("date").reset_index(drop=True)

# ───────────────────────────────────────────────
# 5. Сохранение
# ───────────────────────────────────────────────
out_path = os.path.join(DATA_DIR, "levels.csv")
levels_df.to_csv(out_path, index=False)
print(f"Уровни сохранены: {out_path}  ({len(levels_df)} записей)")

# ───────────────────────────────────────────────
# 6. Превью
# ───────────────────────────────────────────────
print("\n" + "=" * 70)
print("Первые 10 уровней:")
print("=" * 70)
print(levels_df.head(10).to_string(index=False))

print("\n" + "=" * 70)
print("Топ-5 уровней по числу касаний:")
print("=" * 70)
print(levels_df.nlargest(5, "n_touches").to_string(index=False))

print("\n" + "=" * 70)
print("Распределение bounce_rate (уровни с 2+ касаниями):")
active = levels_df[levels_df["n_touches"] >= 2]
print(f"  Всего уровней с 2+ касаниями: {len(active)}")
print(f"  Среднее bounce_rate:           {active['bounce_rate'].mean():.2%}")
print(f"  Уровней с bounce_rate >= 0.7:  {(active['bounce_rate'] >= 0.7).sum()}")
