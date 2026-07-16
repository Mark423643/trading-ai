"""
Step 7 — Hourly backtest на 2 годах данных.
Скачивает 1h свечи MSTR за 730 дней (макс yfinance),
запускает полный пайплайн: уровни → условия → модель → симуляция.
"""
import pandas as pd
import numpy as np
import os
import joblib
import yfinance as yf
from datetime import datetime, timedelta

DATA_DIR  = "data"
MODEL_DIR = "models"

TICKER = "MSTR"

# ── параметры (адаптированы под 1h) ─────────────────
ATR_EXHAUSTION      = 0.30
LEVEL_DIST          = 0.005     # 0.5% — шире чем 5м (0.3%), hourly бары крупнее
SMALL_BODY          = 0.40
VOID_MULTIPLIER     = 1.0       # 1.0× дневной ATR (было 1.5 — слишком жёстко)
STOP_ATR_FRAC       = 0.10
ATR_PERIOD          = 14
RR_TARGET           = 3.0
MAX_TRADE_BARS      = 35        # ~5 торговых дней на 1h
LEVEL_COOLDOWN_BARS = 7         # ~1 день на 1h
LOOKBACK_PIVOT      = 10        # для дневных пивотов
TOUCH_TOL           = 0.003
TREND_EMA           = 20        # дневной EMA для фильтра тренда
FB_LOOKBACK         = 5         # баров для ложного пробоя (шире для hourly)

# ───────────────────────────────────────────────────────
# 1. Загрузка данных
# ───────────────────────────────────────────────────────
def load_ohlcv(path):
    df = pd.read_csv(path, header=[0, 1], index_col=0)
    df.index = pd.to_datetime(df.index)
    df.columns = [col[0] for col in df.columns]
    return df[["Open", "High", "Low", "Close", "Volume"]].astype(float).sort_index().dropna()

def calc_atr(df, period):
    prev = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev).abs(),
        (df["Low"]  - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calc_rsi(series, period=14):
    """Wilder RSI через EWM (идентично step11b_paper_trading.py)."""
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - 100 / (1 + rs)

def to_date_str(ts):
    if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
        return ts.tz_convert("America/New_York").strftime("%Y-%m-%d")
    return ts.strftime("%Y-%m-%d")

# Дневные данные (уровни + тренд)
df1d = load_ohlcv(os.path.join(DATA_DIR, "mstr_daily.csv"))
df1d["ATR_daily"] = calc_atr(df1d, ATR_PERIOD)
df1d["EMA20"]     = df1d["Close"].ewm(span=TREND_EMA, adjust=False).mean()
df1d["trend"]     = np.where(df1d["Close"] > df1d["EMA20"], 1, -1)

daily_atr_map   = dict(zip(pd.to_datetime(df1d.index).strftime("%Y-%m-%d"), df1d["ATR_daily"]))
daily_trend_map = dict(zip(pd.to_datetime(df1d.index).strftime("%Y-%m-%d"), df1d["trend"]))

print(f"Дневных баров: {len(df1d)}  ({df1d.index[0].date()} — {df1d.index[-1].date()})")

# Часовые данные (730 дней — максимум yfinance)
hourly_cache = os.path.join(DATA_DIR, "mstr_1h.csv")
if os.path.exists(hourly_cache):
    dfh = pd.read_csv(hourly_cache, header=[0, 1], index_col=0)
    dfh.index = pd.to_datetime(dfh.index, utc=True)
    dfh.columns = [col[0] for col in dfh.columns]
    dfh = dfh[["Open", "High", "Low", "Close", "Volume"]].astype(float).sort_index().dropna()
    print(f"[кэш] Часовых баров: {len(dfh)}")
else:
    print("Скачиваю часовые данные (730 дней)...")
    end   = datetime.now()
    start = end - timedelta(days=729)
    dfh = yf.download(TICKER, start=start.strftime("%Y-%m-%d"),
                      end=end.strftime("%Y-%m-%d"),
                      interval="1h", auto_adjust=True, progress=False)
    if dfh.empty:
        raise RuntimeError("Часовые данные не загружены")
    dfh.to_csv(hourly_cache)
    dfh.index = pd.to_datetime(dfh.index)
    dfh.columns = [col[0] for col in dfh.columns]
    dfh = dfh[["Open", "High", "Low", "Close", "Volume"]].astype(float).sort_index().dropna()
    print(f"[загружено] Часовых баров: {len(dfh)}")

print(f"Период hourly: {to_date_str(dfh.index[0])} — {to_date_str(dfh.index[-1])}")

# Привязываем дневной ATR и тренд к часовым барам
dfh["date_key"]    = [to_date_str(ts) for ts in dfh.index]
dfh["ATR_daily"]   = dfh["date_key"].map(daily_atr_map).ffill()
dfh["daily_trend"] = dfh["date_key"].map(daily_trend_map).ffill()
dfh["ATR"]         = calc_atr(dfh, ATR_PERIOD)

# Дополнительные индикаторы для 12-признаковой модели
dfh["RSI14"]   = calc_rsi(dfh["Close"])
dfh["EMA20_h"] = dfh["Close"].ewm(span=TREND_EMA, adjust=False).mean()

valid_atr = dfh["ATR_daily"].notna().sum()
print(f"Баров с дневным ATR: {valid_atr}/{len(dfh)}")

# ───────────────────────────────────────────────────────
# 2. Pivot-уровни с дневного ТФ
# ───────────────────────────────────────────────────────
def find_pivots(df, lookback):
    pivots = []
    for i in range(lookback, len(df) - lookback):
        wh = df["High"].iloc[i - lookback:i + lookback + 1]
        wl = df["Low"].iloc[i - lookback:i + lookback + 1]
        if df["High"].iloc[i] == wh.max():
            pivots.append({"date": df.index[i], "level": df["High"].iloc[i], "type": "resistance"})
        if df["Low"].iloc[i] == wl.min():
            pivots.append({"date": df.index[i], "level": df["Low"].iloc[i],  "type": "support"})
    return pd.DataFrame(pivots)

pivots_df   = find_pivots(df1d, LOOKBACK_PIVOT)
level_prices = pivots_df["level"].values
level_types  = pivots_df["type"].values
pivot_dates  = pd.to_datetime(pivots_df["date"]).dt.strftime("%Y-%m-%d").values

print(f"Pivot-уровней: {len(pivots_df)}")

# ───────────────────────────────────────────────────────
# 3. Модель
# ───────────────────────────────────────────────────────
mlp    = joblib.load(os.path.join(MODEL_DIR, "mlp_levels.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

# Предвычислим bounce_rate для каждого уровня из levels.csv
levels_csv = pd.read_csv(os.path.join(DATA_DIR, "levels.csv"))
level_meta = {row["level"]: row for _, row in levels_csv.iterrows()}

def get_model_prob(level_price, level_type_str, h_idx):
    meta = level_meta.get(level_price)
    if meta is None:
        # Уровень не в levels.csv — используем нейтральные признаки
        n_prev, bounce_rate, avg_reaction = 0, 0.5, 0.0
    else:
        n_prev       = meta["n_touches"]
        bounce_rate  = meta["bounce_rate"]
        avg_reaction = meta["avg_reaction_atr"]

    ltype    = 0 if level_type_str == "resistance" else 1
    atr_day  = dfh["ATR_daily"].iloc[h_idx]
    atr_pct  = atr_day / level_price if level_price != 0 else 0
    vol20    = dfh["Volume"].iloc[max(0, h_idx-20):h_idx].mean()
    vol_ratio = dfh["Volume"].iloc[h_idx] / vol20 if vol20 > 0 else 1.0

    # bars_since_form / bars_since_last в дневных барах
    cur_date   = dfh["date_key"].iloc[h_idx]
    pv_date    = pivot_dates[np.where(level_prices == level_price)[0][0]] if level_price in level_prices else cur_date
    all_dates  = dfh["date_key"].unique()
    bars_form  = max(1, len([d for d in all_dates if d >= pv_date]))
    bars_last  = max(1, n_prev)

    # rsi14 (из dfh["RSI14"], если пусто — дефолт 50.0)
    rsi14 = dfh["RSI14"].iloc[h_idx] if "RSI14" in dfh.columns else 50.0
    if pd.isna(rsi14):
        rsi14 = 50.0

    # ema_above (1, если Close > EMA20_h, иначе 0)
    ema_above = 0
    if "EMA20_h" in dfh.columns:
        ema_val = dfh["EMA20_h"].iloc[h_idx]
        if not pd.isna(ema_val):
            ema_above = int(dfh["Close"].iloc[h_idx] > ema_val)

    # touch_hour (час из dfh.index[h_idx].hour)
    touch_hour = dfh.index[h_idx].hour

    # dist_next: расстояние до ближайшего соседнего уровня в ATR, ограничено 20.0, дефолт 10.0
    other_lvls = level_prices[np.abs(level_prices - level_price) > level_price * 0.001]
    if len(other_lvls) > 0 and atr_day > 0:
        dist_next = float(np.min(np.abs(other_lvls - dfh["Close"].iloc[h_idx])) / atr_day)
    else:
        dist_next = 10.0
    dist_next = min(dist_next, 20.0)

    feat = np.array([[ltype, n_prev, bounce_rate, avg_reaction,
                      bars_form, bars_last, vol_ratio, atr_pct,
                      rsi14, ema_above, touch_hour, dist_next]])
    return mlp.predict_proba(scaler.transform(feat))[0][1]

# ───────────────────────────────────────────────────────
# 4. Условия стратегии
# ───────────────────────────────────────────────────────
def slow_approach(i, lookback=5):
    if i < lookback:
        return False
    atr = dfh["ATR"].iloc[i]
    if np.isnan(atr) or atr == 0:
        return False
    bodies = (dfh["Close"].iloc[i-lookback:i] - dfh["Open"].iloc[i-lookback:i]).abs()
    return bodies.max() < 1.5 * atr

def false_breakout(i, level_price, is_long, lookback=3):
    if i < lookback + 1:
        return False
    cur = dfh["Close"].iloc[i]
    for j in range(i - lookback, i):
        prev = dfh["Close"].iloc[j]
        if is_long  and cur >= level_price and prev < level_price:
            return True
        if not is_long and cur <= level_price and prev > level_price:
            return True
    return False

# ───────────────────────────────────────────────────────
# 5. Генерация сигналов
# ───────────────────────────────────────────────────────
def generate_signals(model_threshold):
    signals = []
    level_last_signal = {}
    filter_counts = {"total": 0, "atr": 0, "level": 0, "body": 0,
                     "slow": 0, "trend": 0, "cooldown": 0, "void": 0, "fb": 0, "model": 0}

    for i in range(20, len(dfh) - MAX_TRADE_BARS - 1):
        bar     = dfh.iloc[i]
        atr_h   = bar["ATR"]
        atr_day = bar["ATR_daily"]
        trend   = bar["daily_trend"]
        filter_counts["total"] += 1

        if np.isnan(atr_h) or np.isnan(atr_day) or atr_day == 0:
            continue
        if atr_h >= ATR_EXHAUSTION * atr_day:
            continue
        filter_counts["atr"] += 1

        dists = np.abs(level_prices - bar["Close"]) / bar["Close"]
        min_dist_idx = np.argmin(dists)
        if dists[min_dist_idx] >= LEVEL_DIST:
            continue
        filter_counts["level"] += 1

        level_price    = level_prices[min_dist_idx]
        level_type_str = level_types[min_dist_idx]
        is_long        = (level_type_str == "support")

        # Тренд-фильтр
        if not np.isnan(trend):
            if is_long and trend < 0:
                continue
            if not is_long and trend > 0:
                continue
        filter_counts["trend"] += 1

        # Cooldown
        last_bar = level_last_signal.get(level_price, -LEVEL_COOLDOWN_BARS - 1)
        if i - last_bar < LEVEL_COOLDOWN_BARS:
            continue
        filter_counts["cooldown"] += 1

        # Маленькие тела
        bodies = (dfh["Close"].iloc[i-5:i] - dfh["Open"].iloc[i-5:i]).abs()
        if bodies.mean() >= SMALL_BODY * atr_h:
            continue
        filter_counts["body"] += 1

        # Медленный подход
        if not slow_approach(i):
            continue
        filter_counts["slow"] += 1

        # Пустота впереди
        cands = level_prices[level_prices > level_price] if is_long else level_prices[level_prices < level_price]
        if len(cands) == 0:
            continue
        next_lvl  = cands[np.argmin(np.abs(cands - bar["Close"]))]
        void_dist = abs(next_lvl - level_price)
        if void_dist <= VOID_MULTIPLIER * atr_day:
            continue
        filter_counts["void"] += 1

        # Ложный пробой / ретест
        if not false_breakout(i, level_price, is_long, lookback=FB_LOOKBACK):
            continue
        filter_counts["fb"] += 1

        # Модель
        prob = get_model_prob(level_price, level_type_str, i)
        if prob < model_threshold:
            continue
        filter_counts["model"] += 1

        entry  = bar["Close"]
        stop   = (level_price - STOP_ATR_FRAC * atr_day) if is_long else (level_price + STOP_ATR_FRAC * atr_day)
        risk   = abs(entry - stop)
        if risk == 0:
            continue
        target = entry + RR_TARGET * risk if is_long else entry - RR_TARGET * risk

        level_last_signal[level_price] = i
        signals.append({
            "bar_idx":    i,
            "datetime":   dfh.index[i],
            "entry":      entry,
            "stop":       stop,
            "target":     target,
            "risk":       risk,
            "long":       is_long,
            "level":      level_price,
            "level_type": level_type_str,
            "model_prob": round(prob, 3),
            "trend":      int(trend) if not np.isnan(trend) else 0,
        })

    return signals, filter_counts

# ───────────────────────────────────────────────────────
# 6. Симуляция сделки
# ───────────────────────────────────────────────────────
def simulate_trade(sig):
    start     = sig["bar_idx"] + 1
    if start >= len(dfh):
        return None
    entry     = dfh["Open"].iloc[start]
    stop      = sig["stop"]
    target    = sig["target"]
    is_long   = sig["long"]
    risk      = abs(entry - stop)
    if risk == 0:
        return None

    for j in range(start, min(start + MAX_TRADE_BARS, len(dfh))):
        hi = dfh["High"].iloc[j]
        lo = dfh["Low"].iloc[j]
        if is_long:
            sl_hit = lo <= stop
            tp_hit = hi >= target
        else:
            sl_hit = hi >= stop
            tp_hit = lo <= target

        if sl_hit and tp_hit:
            outcome, exit_price = "SL", stop
        elif tp_hit:
            outcome, exit_price = "TP", target
        elif sl_hit:
            outcome, exit_price = "SL", stop
        else:
            continue

        pnl_r = (exit_price - entry) / risk if is_long else (entry - exit_price) / risk
        return {**sig, "entry": round(entry, 4), "exit": round(exit_price, 4),
                "outcome": outcome, "pnl_r": round(pnl_r, 3), "bars_held": j - start}

    # Таймаут
    exit_price = dfh["Close"].iloc[min(start + MAX_TRADE_BARS - 1, len(dfh)-1)]
    pnl_r = (exit_price - entry) / risk if is_long else (entry - exit_price) / risk
    return {**sig, "entry": round(entry, 4), "exit": round(exit_price, 4),
            "outcome": "TIMEOUT", "pnl_r": round(pnl_r, 3), "bars_held": MAX_TRADE_BARS}

# ───────────────────────────────────────────────────────
# 7. Метрики
# ───────────────────────────────────────────────────────
def calc_metrics(tdf):
    if len(tdf) == 0:
        return {}
    wins   = tdf[tdf["outcome"] == "TP"]
    losses = tdf[tdf["outcome"].isin(["SL", "TIMEOUT"])]
    eq     = tdf["pnl_r"].cumsum().values
    peak   = np.maximum.accumulate(eq)
    dd     = eq - peak

    streaks, curr = [], 0
    for r in tdf["pnl_r"]:
        curr = curr + 1 if r < 0 else 0
        streaks.append(curr)

    std_r = tdf["pnl_r"].std()
    return {
        "n_trades":        len(tdf),
        "winrate":         round(len(wins) / len(tdf), 3),
        "avg_win_r":       round(wins["pnl_r"].mean(), 3)   if len(wins) > 0 else 0,
        "avg_loss_r":      round(losses["pnl_r"].mean(), 3) if len(losses) > 0 else 0,
        "total_r":         round(tdf["pnl_r"].sum(), 3),
        "profit_factor":   round(wins["pnl_r"].sum() / losses["pnl_r"].abs().sum(), 3)
                           if len(losses) > 0 and losses["pnl_r"].abs().sum() > 0 else 0,
        "max_drawdown_r":  round(dd.min(), 3),
        "max_consec_loss": max(streaks) if streaks else 0,
        "sharpe":          round(tdf["pnl_r"].mean() / std_r * np.sqrt(len(tdf)), 3) if std_r > 0 else 0,
        "equity_final":    round(eq[-1], 3),
        "n_long":          int((tdf["long"] == True).sum()),
        "n_short":         int((tdf["long"] == False).sum()),
        "tp_count":        len(wins),
        "sl_count":        len(losses[losses["outcome"]=="SL"]),
        "timeout_count":   len(losses[losses["outcome"]=="TIMEOUT"]),
    }

# ───────────────────────────────────────────────────────
# 8. Запуск
# ───────────────────────────────────────────────────────
thresholds  = [0.50, 0.55, 0.60]
all_results = {}

for thr in thresholds:
    sigs, fcounts = generate_signals(thr)
    trades = [simulate_trade(s) for s in sigs]
    trades = [t for t in trades if t is not None]
    tdf    = pd.DataFrame(trades)
    all_results[thr] = {
        "trades":  tdf,
        "metrics": calc_metrics(tdf),
        "filters": fcounts,
    }

# ───────────────────────────────────────────────────────
# 9. Отчёт
# ───────────────────────────────────────────────────────
print("\n" + "="*68)
print(f"БЭКТЕСТ  {TICKER}  HOURLY  2 года")
print("="*68)

metric_labels = [
    ("n_trades",        "Сделок"),
    ("n_long",          "  Лонгов"),
    ("n_short",         "  Шортов"),
    ("tp_count",        "  TP"),
    ("sl_count",        "  SL"),
    ("timeout_count",   "  Timeout"),
    ("winrate",         "Winrate"),
    ("avg_win_r",       "Avg Win (R)"),
    ("avg_loss_r",      "Avg Loss (R)"),
    ("total_r",         "Total P&L (R)"),
    ("profit_factor",   "Profit Factor"),
    ("max_drawdown_r",  "Max Drawdown (R)"),
    ("max_consec_loss", "Макс убыт. подряд"),
    ("sharpe",          "Sharpe (R)"),
    ("equity_final",    "Equity Final (R)"),
]

header = f"{'Метрика':<22}" + "".join(f"  {'Порог '+str(t):>10}" for t in thresholds)
print(header)
print("-"*68)
for key, label in metric_labels:
    row = f"{label:<22}"
    for thr in thresholds:
        m   = all_results[thr]["metrics"]
        val = m.get(key, "—")
        if key == "winrate" and isinstance(val, float):
            row += f"  {val:>10.1%}"
        elif isinstance(val, float):
            row += f"  {val:>10.3f}"
        else:
            row += f"  {val:>10}"
    print(row)
print("="*68)

# Воронка фильтров (для порога 0.50)
fc = all_results[0.50]["filters"]
print(f"\n--- Воронка фильтров (порог 0.50) ---")
print(f"  Всего 1h баров:         {fc['total']}")
print(f"  После ATR-фильтра:      {fc['atr']}  ({fc['atr']/fc['total']:.1%})")
print(f"  После фильтра уровня:   {fc['level']}")
print(f"  После тренд-фильтра:    {fc['trend']}")
print(f"  После cooldown:         {fc['cooldown']}")
print(f"  После малых тел:        {fc['body']}")
print(f"  После медл. подхода:    {fc['slow']}")
print(f"  После пустоты:          {fc['void']}")
print(f"  После ложн. пробоя:     {fc['fb']}")
print(f"  После модели (>0.50):   {fc['model']}")

# Детальные сделки лучшего порога
best_thr = max(thresholds, key=lambda t: all_results[t]["metrics"].get("total_r", -999))
best_df  = all_results[best_thr]["trades"]

if len(best_df) > 0:
    out_path = os.path.join(DATA_DIR, "backtest_hourly.csv")
    best_df.to_csv(out_path, index=False)
    print(f"\nСделки (порог {best_thr}) сохранены: {out_path}")

    cols = ["datetime", "long", "level", "entry", "exit", "outcome", "pnl_r", "bars_held", "model_prob"]
    print(f"\n--- Все сделки (порог {best_thr}, {len(best_df)} шт.) ---")
    print(best_df[cols].to_string(index=False))

    # Equity curve ASCII
    eq = best_df["pnl_r"].cumsum().round(2).tolist()
    if len(eq) <= 40:
        print(f"\n--- Equity curve (R) ---")
        print("Сд:  " + " ".join(f"{i+1:>4}" for i in range(len(eq))))
        print("Eq:  " + " ".join(f"{v:>4.1f}" for v in eq))

    # P&L на капитал
    capital, risk_pct = 10_000, 0.01
    cap = capital
    for r in best_df["pnl_r"]:
        cap += r * cap * risk_pct
    print(f"\nP&L ($10 000, риск 1%/сделку):")
    print(f"  Итог: ${cap:,.0f}  ({(cap/capital-1):+.1%})")
else:
    print(f"\nСделок нет даже при пороге {best_thr}.")
    print("Попробуй: снизить LEVEL_DIST до 0.005 или убрать тренд-фильтр.")
