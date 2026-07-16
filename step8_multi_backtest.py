"""
Step 8 — Мульти-тикерный hourly backtest.
15 тикеров × 2 года hourly = полноценная статистика.
"""
import pandas as pd
import numpy as np
import os
import joblib
import yfinance as yf
from datetime import datetime, timedelta

DATA_DIR  = "data"
MODEL_DIR = "models"
TICK_DIR  = os.path.join(DATA_DIR, "tickers")
os.makedirs(TICK_DIR, exist_ok=True)

TICKERS = [
    "MSTR", "NVDA", "TSLA", "AAPL", "MSFT",
    "META", "AMZN", "GOOGL", "AMD",  "COIN",
    "PLTR", "SNOW", "CRWD", "SMCI", "IONQ",
]

# ── параметры (из step7) ─────────────────────────────
ATR_EXHAUSTION      = 0.30
LEVEL_DIST          = 0.005
SMALL_BODY          = 0.40
VOID_MULTIPLIER     = 1.0
STOP_ATR_FRAC       = 0.10
ATR_PERIOD          = 14
RR_TARGET           = 3.0
MAX_TRADE_BARS      = 35
LEVEL_COOLDOWN_BARS = 7
LOOKBACK_PIVOT      = 10
TREND_EMA           = 20
FB_LOOKBACK         = 5

mlp    = joblib.load(os.path.join(MODEL_DIR, "mlp_levels.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

# ───────────────────────────────────────────────────────
# Утилиты
# ───────────────────────────────────────────────────────
def to_date_str(ts):
    if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
        return ts.tz_convert("America/New_York").strftime("%Y-%m-%d")
    return ts.strftime("%Y-%m-%d")

def calc_atr(df, period):
    prev = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev).abs(),
        (df["Low"]  - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def parse_csv(path, utc=False):
    df = pd.read_csv(path, header=[0, 1], index_col=0)
    df.index = pd.to_datetime(df.index, utc=utc)
    df.columns = [col[0] for col in df.columns]
    return df[["Open","High","Low","Close","Volume"]].astype(float).sort_index().dropna()

def download(ticker, interval, period_days, cache_path, utc=False):
    if os.path.exists(cache_path):
        return parse_csv(cache_path, utc=utc)
    end   = datetime.now()
    start = end - timedelta(days=period_days - 1)
    raw   = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                        end=end.strftime("%Y-%m-%d"),
                        interval=interval, auto_adjust=True, progress=False)
    if raw.empty:
        return None
    raw.to_csv(cache_path)
    df = parse_csv(cache_path, utc=utc)
    return df

def find_pivots(df, lookback):
    rows = []
    for i in range(lookback, len(df) - lookback):
        wh = df["High"].iloc[i-lookback:i+lookback+1]
        wl = df["Low"].iloc[i-lookback:i+lookback+1]
        if df["High"].iloc[i] == wh.max():
            rows.append({"date": df.index[i], "level": df["High"].iloc[i], "type": "resistance"})
        if df["Low"].iloc[i] == wl.min():
            rows.append({"date": df.index[i], "level": df["Low"].iloc[i],  "type": "support"})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["date","level","type"])

# ───────────────────────────────────────────────────────
# Условия стратегии (работают на переданном dfh)
# ───────────────────────────────────────────────────────
def slow_approach(dfh, i, lookback=5):
    if i < lookback:
        return False
    atr = dfh["ATR"].iloc[i]
    if np.isnan(atr) or atr == 0:
        return False
    bodies = (dfh["Close"].iloc[i-lookback:i] - dfh["Open"].iloc[i-lookback:i]).abs()
    return bodies.max() < 1.5 * atr

def false_breakout(dfh, i, level_price, is_long, lookback=FB_LOOKBACK):
    if i < lookback + 1:
        return False
    cur = dfh["Close"].iloc[i]
    for j in range(i - lookback, i):
        prev = dfh["Close"].iloc[j]
        if is_long     and cur >= level_price and prev < level_price:
            return True
        if not is_long and cur <= level_price and prev > level_price:
            return True
    return False

def get_model_prob(dfh, level_price, level_type_str, i,
                   n_touches, bounce_rate, avg_reaction, pv_date_str):
    ltype     = 0 if level_type_str == "resistance" else 1
    atr_day   = dfh["ATR_daily"].iloc[i]
    atr_pct   = atr_day / level_price if level_price != 0 else 0
    vol20     = dfh["Volume"].iloc[max(0, i-20):i].mean()
    vol_ratio = dfh["Volume"].iloc[i] / vol20 if vol20 > 0 else 1.0
    all_dates = dfh["date_key"].unique()
    bars_form = max(1, len([d for d in all_dates if d >= pv_date_str]))
    bars_last = max(1, n_touches)
    feat = np.array([[ltype, n_touches, bounce_rate, avg_reaction,
                      bars_form, bars_last, vol_ratio, atr_pct]])
    return mlp.predict_proba(scaler.transform(feat))[0][1]

# ───────────────────────────────────────────────────────
# Пайплайн для одного тикера
# ───────────────────────────────────────────────────────
def run_ticker(ticker, model_threshold=0.50):
    # Загрузка / кэш данных
    daily_path  = os.path.join(TICK_DIR, f"{ticker.lower()}_daily.csv")
    hourly_path = os.path.join(TICK_DIR, f"{ticker.lower()}_1h.csv")

    df1d = download(ticker, "1d",  730, daily_path,  utc=False)
    dfh  = download(ticker, "1h",  730, hourly_path, utc=True)

    if df1d is None or dfh is None or len(df1d) < 30 or len(dfh) < 100:
        return [], f"{ticker}: недостаточно данных"

    # Дневной ATR + тренд
    df1d["ATR_daily"] = calc_atr(df1d, ATR_PERIOD)
    df1d["EMA20"]     = df1d["Close"].ewm(span=TREND_EMA, adjust=False).mean()
    df1d["trend"]     = np.where(df1d["Close"] > df1d["EMA20"], 1, -1)

    d_atr   = dict(zip(pd.to_datetime(df1d.index).strftime("%Y-%m-%d"), df1d["ATR_daily"]))
    d_trend = dict(zip(pd.to_datetime(df1d.index).strftime("%Y-%m-%d"), df1d["trend"]))

    # Привязка к hourly
    dfh["date_key"]    = [to_date_str(ts) for ts in dfh.index]
    dfh["ATR_daily"]   = dfh["date_key"].map(d_atr).ffill()
    dfh["daily_trend"] = dfh["date_key"].map(d_trend).ffill()
    dfh["ATR"]         = calc_atr(dfh, ATR_PERIOD)

    # Pivot-уровни
    pivots = find_pivots(df1d, LOOKBACK_PIVOT)
    if len(pivots) == 0:
        return [], f"{ticker}: нет пивотов"

    lev_prices = pivots["level"].values
    lev_types  = pivots["type"].values
    lev_dates  = pd.to_datetime(pivots["date"]).dt.strftime("%Y-%m-%d").values

    # Простой подсчёт касаний (bounce_rate) по дневным данным
    def touch_stats(level_price, pivot_bar_approx):
        bounces, total = 0, 0
        for k in range(pivot_bar_approx + 1, len(df1d) - 5):
            bar = df1d.iloc[k]
            dist = min(abs(bar["Close"]-level_price), abs(bar["Low"]-level_price),
                       abs(bar["High"]-level_price)) / level_price
            if dist < 0.003:
                future = df1d["Close"].iloc[k+1:k+6]
                if len(future) < 5:
                    break
                above  = bar["Close"] > level_price
                bounce = int((above and future.iloc[-1] > level_price) or
                             (not above and future.iloc[-1] < level_price))
                bounces += bounce
                total   += 1
                k += 5
        br = bounces / total if total > 0 else 0.5
        ar = 1.0  # упрощение: avg_reaction = 1 ATR
        return total, br, ar

    # Кэш статистики по уровням (избегаем пересчёт)
    stats_cache = {}
    for idx, row in pivots.iterrows():
        stats_cache[row["level"]] = touch_stats(row["level"], idx)

    # Генерация сигналов
    signals = []
    level_last_signal = {}

    for i in range(20, len(dfh) - MAX_TRADE_BARS - 1):
        bar     = dfh.iloc[i]
        atr_h   = bar["ATR"]
        atr_day = bar["ATR_daily"]
        trend   = bar["daily_trend"]

        if np.isnan(atr_h) or np.isnan(atr_day) or atr_day == 0:
            continue
        if atr_h >= ATR_EXHAUSTION * atr_day:
            continue

        dists = np.abs(lev_prices - bar["Close"]) / bar["Close"]
        mi    = np.argmin(dists)
        if dists[mi] >= LEVEL_DIST:
            continue

        level_price    = lev_prices[mi]
        level_type_str = lev_types[mi]
        is_long        = (level_type_str == "support")
        pv_date_str    = lev_dates[mi]

        # Тренд
        if not np.isnan(trend):
            if is_long and trend < 0:
                continue
            if not is_long and trend > 0:
                continue

        # Cooldown
        if i - level_last_signal.get(level_price, -LEVEL_COOLDOWN_BARS-1) < LEVEL_COOLDOWN_BARS:
            continue

        # Маленькие тела
        bodies = (dfh["Close"].iloc[i-5:i] - dfh["Open"].iloc[i-5:i]).abs()
        if bodies.mean() >= SMALL_BODY * atr_h:
            continue

        # Медленный подход
        if not slow_approach(dfh, i):
            continue

        # Пустота
        cands = lev_prices[lev_prices > level_price] if is_long else lev_prices[lev_prices < level_price]
        if len(cands) == 0:
            continue
        void_dist = abs(cands[np.argmin(np.abs(cands - bar["Close"]))] - level_price)
        if void_dist <= VOID_MULTIPLIER * atr_day:
            continue

        # Ложный пробой
        if not false_breakout(dfh, i, level_price, is_long):
            continue

        # Модель
        n_t, br, ar = stats_cache.get(level_price, (0, 0.5, 1.0))
        prob = get_model_prob(dfh, level_price, level_type_str, i, n_t, br, ar, pv_date_str)
        if prob < model_threshold:
            continue

        entry  = bar["Close"]
        stop   = (level_price - STOP_ATR_FRAC * atr_day) if is_long else (level_price + STOP_ATR_FRAC * atr_day)
        risk   = abs(entry - stop)
        if risk == 0:
            continue
        target = entry + RR_TARGET * risk if is_long else entry - RR_TARGET * risk

        level_last_signal[level_price] = i
        signals.append({
            "ticker":     ticker,
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
        })

    # Симуляция
    trades = []
    for sig in signals:
        start = sig["bar_idx"] + 1
        if start >= len(dfh):
            continue
        entry   = dfh["Open"].iloc[start]
        stop    = sig["stop"]
        target  = sig["target"]
        is_long = sig["long"]
        risk    = abs(entry - stop)
        if risk == 0:
            continue

        outcome, exit_price, bars_held = "TIMEOUT", None, MAX_TRADE_BARS
        for j in range(start, min(start + MAX_TRADE_BARS, len(dfh))):
            hi, lo = dfh["High"].iloc[j], dfh["Low"].iloc[j]
            sl_hit = (lo <= stop) if is_long else (hi >= stop)
            tp_hit = (hi >= target) if is_long else (lo <= target)
            if sl_hit and tp_hit:
                outcome, exit_price, bars_held = "SL", stop, j - start; break
            elif tp_hit:
                outcome, exit_price, bars_held = "TP", target, j - start; break
            elif sl_hit:
                outcome, exit_price, bars_held = "SL", stop, j - start; break

        if exit_price is None:
            exit_price = dfh["Close"].iloc[min(start + MAX_TRADE_BARS - 1, len(dfh)-1)]

        pnl_r = (exit_price - entry) / risk if is_long else (entry - exit_price) / risk
        trades.append({**sig, "entry": round(entry,4), "exit": round(exit_price,4),
                       "outcome": outcome, "pnl_r": round(pnl_r,3), "bars_held": bars_held})

    return trades, f"{ticker}: сигналов={len(signals)}, сделок={len(trades)}"

# ───────────────────────────────────────────────────────
# Запуск по всем тикерам
# ───────────────────────────────────────────────────────
print("Запуск мульти-тикерного бэктеста...\n")
all_trades = []

for ticker in TICKERS:
    trades, msg = run_ticker(ticker, model_threshold=0.50)
    print(f"  {msg}")
    all_trades.extend(trades)

df_all = pd.DataFrame(all_trades)

if len(df_all) == 0:
    print("\nСделок нет. Попробуй снизить MODEL_THRESHOLD или VOID_MULTIPLIER.")
    exit()

df_all.to_csv(os.path.join(DATA_DIR, "backtest_multi.csv"), index=False)

# ───────────────────────────────────────────────────────
# Агрегированные метрики
# ───────────────────────────────────────────────────────
def metrics(df):
    if len(df) == 0:
        return {}
    wins   = df[df["outcome"] == "TP"]
    losses = df[df["outcome"].isin(["SL","TIMEOUT"])]
    eq     = df["pnl_r"].cumsum().values
    peak   = np.maximum.accumulate(eq)
    dd     = (eq - peak).min()
    gp     = wins["pnl_r"].sum()
    gl     = losses["pnl_r"].abs().sum()
    std_r  = df["pnl_r"].std()

    streaks, cur = [], 0
    for r in df["pnl_r"]:
        cur = cur + 1 if r < 0 else 0
        streaks.append(cur)

    return {
        "n_trades":       len(df),
        "winrate":        round(len(wins)/len(df), 3),
        "avg_win_r":      round(wins["pnl_r"].mean(), 3)   if len(wins) > 0  else 0,
        "avg_loss_r":     round(losses["pnl_r"].mean(), 3) if len(losses) > 0 else 0,
        "total_r":        round(df["pnl_r"].sum(), 3),
        "profit_factor":  round(gp/gl, 3) if gl > 0 else 0,
        "max_dd_r":       round(dd, 3),
        "max_streak":     max(streaks) if streaks else 0,
        "sharpe":         round(df["pnl_r"].mean()/std_r*np.sqrt(len(df)), 3) if std_r > 0 else 0,
        "equity_final":   round(eq[-1], 3),
        "n_long":         int((df["long"]==True).sum()),
        "n_short":        int((df["long"]==False).sum()),
        "n_tp":           len(wins),
        "n_sl":           int((df["outcome"]=="SL").sum()),
        "n_timeout":      int((df["outcome"]=="TIMEOUT").sum()),
    }

m = metrics(df_all)

print(f"\n{'='*60}")
print(f"СВОДНЫЕ РЕЗУЛЬТАТЫ  ({len(TICKERS)} тикеров, hourly, ~2 года)")
print(f"{'='*60}")
print(f"  Сделок всего:        {m['n_trades']}  (лонг: {m['n_long']}, шорт: {m['n_short']})")
print(f"  TP / SL / Timeout:   {m['n_tp']} / {m['n_sl']} / {m['n_timeout']}")
print(f"  Winrate:             {m['winrate']:.1%}")
print(f"  Avg Win (R):         {m['avg_win_r']:.3f}")
print(f"  Avg Loss (R):        {m['avg_loss_r']:.3f}")
print(f"  Total P&L (R):       {m['total_r']:.3f}")
print(f"  Profit Factor:       {m['profit_factor']:.3f}")
print(f"  Max Drawdown (R):    {m['max_dd_r']:.3f}")
print(f"  Max убыт. подряд:    {m['max_streak']}")
print(f"  Sharpe (R):          {m['sharpe']:.3f}")
print(f"  Equity Final (R):    {m['equity_final']:.3f}")

# Капитал $10k, риск 1%
capital, rp = 10_000, 0.01
cap = capital
for r in df_all["pnl_r"]:
    cap += r * cap * rp
print(f"\n  P&L ($10k, 1%/сделку): ${cap:,.0f}  ({(cap/capital-1):+.1%})")

# ───────────────────────────────────────────────────────
# По тикерам
# ───────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("РЕЗУЛЬТАТЫ ПО ТИКЕРАМ")
print(f"{'='*60}")
print(f"{'Тикер':<7} {'Сделок':>7} {'Winrate':>9} {'Total R':>9} {'PF':>7} {'Sharpe':>8}")
print("-"*60)

for t in TICKERS:
    sub = df_all[df_all["ticker"] == t]
    if len(sub) == 0:
        print(f"{t:<7} {'—':>7}")
        continue
    ms = metrics(sub)
    wr = f"{ms['winrate']:.0%}"
    print(f"{t:<7} {ms['n_trades']:>7} {wr:>9} {ms['total_r']:>9.2f} "
          f"{ms['profit_factor']:>7.2f} {ms['sharpe']:>8.3f}")

# ───────────────────────────────────────────────────────
# Все сделки
# ───────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("ВСЕ СДЕЛКИ")
print(f"{'='*60}")
cols = ["ticker","datetime","long","level","entry","exit","outcome","pnl_r","bars_held","model_prob"]
print(df_all[cols].to_string(index=False))

# Equity curve (ASCII, до 50 точек)
eq = df_all["pnl_r"].cumsum().round(2).tolist()
step = max(1, len(eq) // 50)
sample = eq[::step]
print(f"\n--- Equity curve (R), каждые {step} сделок ---")
print(" ".join(f"{v:+.1f}" for v in sample))
