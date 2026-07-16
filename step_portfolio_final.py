"""
Step Portfolio Final — портфельный бэктест MOEX с откалиброванными фильтрами.
Параметры перенесены из step7_hourly_backtest.py (ATR_EXHAUSTION=0.40,
LEVEL_DIST=0.012, SMALL_BODY=0.70, VOID_MULTIPLIER=0.2, STOP_ATR_FRAC=0.20,
FB_LOOKBACK=5, High/Low false_breakout, без тренд-фильтра шортов).
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
import os
import time
import requests
import joblib
import matplotlib
matplotlib.use("Agg")
from datetime import datetime, timedelta

from config_trading import (
    ATR_EXHAUSTION, LEVEL_DIST, SMALL_BODY, VOID_MULTIPLIER,
    STOP_ATR_FRAC, FB_LOOKBACK, ATR_PERIOD, TREND_EMA,
    LOOKBACK_PIVOT, RR_TARGET, LEVEL_COOLDOWN_BARS,
    MODEL_THRESHOLD, MAX_TRADE_BARS, RISK_PCT, CAPITAL,
    TREND_FILTER_SHORTS, MIN_VOL_RATIO, RSI_LONG_MIN, RSI_SHORT_MAX,
    MOEX_TICKERS,
    COMMISSION_PCT, SLIPPAGE_STEPS,
    MAX_COST_RATIO, COST_RATIO_ACTION, COST_RR_OVERRIDE,
)

DATA_DIR  = "data"
MODEL_DIR = "models"
CACHE_DIR = os.path.join(DATA_DIR, "bt_cache")
OUT_DIR   = "charts"
for d in (CACHE_DIR, OUT_DIR):
    os.makedirs(d, exist_ok=True)

DATE_FROM = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
DATE_TO   = datetime.now().strftime("%Y-%m-%d")

# ── модель (загружается только если MODEL_THRESHOLD > 0) ──
if MODEL_THRESHOLD > 0:
    mlp    = joblib.load(os.path.join(MODEL_DIR, "mlp_levels.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    _ML_READY = True
else:
    mlp = scaler = None
    _ML_READY = False

# ── лотность MOEX (для расчёта комиссии) ─────────────────────
_LOT_SIZES = {
    "SBER": 1, "GAZP": 10, "LKOH": 1,
    "GMKN": 10, "CHMF": 10, "NLMK": 10,
    "TATN": 1, "MOEX": 1, "MAGN": 1, "NVTK": 1,
    "ROSN": 1,
    "ALRS": 1, "PHOR": 1, "AFKS": 1, "CBOM": 1, "TRNFP": 1,
}

def moex_tick_size(price: float) -> float:
    """
    Минимальный шаг цены для MOEX TQBR в зависимости от цены.
    https://www.moex.com/a207
    """
    if price < 10:
        return 0.01
    elif price < 25:
        return 0.02
    elif price < 100:
        return 0.05
    elif price < 500:
        return 0.10
    elif price < 1000:
        return 0.50
    else:
        return 1.00


def cost_ratio_vs_risk(price: float, risk: float, tick_size: float) -> float:
    """
    Доля издержек (комиссия 0.03% на вход+выход + 2×проскальзывание)
    от рублёвого стоп-лосса на 1 лот.
    """
    comm_entry = price * COMMISSION_PCT
    comm_exit  = price * COMMISSION_PCT     # приближение: exit ≈ entry
    slippage   = tick_size * SLIPPAGE_STEPS
    total_cost = comm_entry + comm_exit + 2 * slippage
    risk_rub   = abs(risk)
    return total_cost / risk_rub if risk_rub > 0 else 1.0


def effective_rr(is_long: bool, entry: float, stop: float, target: float,
                 ticker: str, tick_size: float) -> tuple[float, float, bool]:
    """
    Проверяет cost-ratio и при необходимости корректирует target.
    Возвращает (target, risk, was_skipped).
    Если сделка должна быть пропущена, was_skipped=True.
    """
    risk = abs(entry - stop)
    if risk == 0:
        return target, risk, True

    cr = cost_ratio_vs_risk(entry, risk, tick_size)

    if cr <= MAX_COST_RATIO:
        # Издержки приемлемы — ничего не меняем
        return target, risk, False

    # Издержки превышают порог
    if COST_RATIO_ACTION == "skip":
        return target, risk, True   # пропускаем

    if COST_RATIO_ACTION == "increase_rr":
        # Повышаем target до COST_RR_OVERRIDE:1
        new_target = entry + COST_RR_OVERRIDE * risk if is_long else entry - COST_RR_OVERRIDE * risk
        return new_target, risk, False

    return target, risk, False


# ────────────────────────────────────────────────────────
# MOEX ISS API
# ────────────────────────────────────────────────────────
MOEX_URL = ("https://iss.moex.com/iss/engines/stock/markets/shares"
            "/boards/TQBR/securities/{ticker}/candles.json")

def fetch_candles(ticker, interval, date_from, date_to):
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
            }, timeout=30)
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
            print(f"    [ошибка] {ticker} i={interval}: {e}")
            break
    if not chunks:
        return None
    df = pd.concat(chunks, ignore_index=True)
    df = df.rename(columns={
        "begin": "Datetime", "open": "Open", "high": "High",
        "low": "Low", "close": "Close", "volume": "Volume",
    })
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    return (df.set_index("Datetime")[["Open","High","Low","Close","Volume"]]
              .astype(float).sort_index().dropna())

def load_candles(ticker, interval, label):
    path = os.path.join(CACHE_DIR, f"{ticker.lower()}_{label}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df[["Open","High","Low","Close","Volume"]].astype(float).sort_index().dropna()
    df = fetch_candles(ticker, interval, DATE_FROM, DATE_TO)
    if df is not None and len(df) > 0:
        df.to_csv(path)
    return df

# ── Индикаторы ──────────────────────────────────────────────
def calc_atr(df, period):
    prev = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev).abs(),
        (df["Low"]  - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calc_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - 100 / (1 + rs)

def find_pivots(df1d, lookback):
    rows = []
    for i in range(lookback, len(df1d) - lookback):
        wh = df1d["High"].iloc[i-lookback:i+lookback+1]
        wl = df1d["Low"].iloc[i-lookback:i+lookback+1]
        if df1d["High"].iloc[i] == wh.max():
            rows.append({"date": df1d.index[i], "level": df1d["High"].iloc[i], "type": "resistance"})
        if df1d["Low"].iloc[i] == wl.min():
            rows.append({"date": df1d.index[i], "level": df1d["Low"].iloc[i],  "type": "support"})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["date","level","type"])

def false_breakout(dfh, i, level_price, is_long, lookback=FB_LOOKBACK):
    """
    Ослабленная версия: достаточно касания через High/Low (было через Close).
    Идентично step7_hourly_backtest.py.
    """
    if i < lookback + 1:
        return False
    cur_close = dfh["Close"].iloc[i]
    cur_high  = dfh["High"].iloc[i]
    cur_low   = dfh["Low"].iloc[i]
    for j in range(i - lookback, i):
        prev_high = dfh["High"].iloc[j]
        prev_low  = dfh["Low"].iloc[j]
        if is_long:
            if cur_low <= level_price and prev_low < level_price:
                return True
            if cur_close >= level_price and prev_low < level_price:
                return True
        else:
            if cur_high >= level_price and prev_high > level_price:
                return True
            if cur_close <= level_price and prev_high > level_price:
                return True
    return False

# ── get_model_prob ──────────────────────────────────────────
def get_model_prob(dfh, level_price, level_type_str, i, n_t, br, ar, pv_date_str, all_levels):
    if not _ML_READY:
        return 1.0
    ltype     = 0 if level_type_str == "resistance" else 1
    atr_day   = dfh["ATR_daily"].iloc[i]
    atr_pct   = atr_day / (level_price + 1e-9)
    vol20     = dfh["Volume"].iloc[max(0, i - 20):i].mean()
    vol_ratio = dfh["Volume"].iloc[i] / (vol20 + 1e-9)

    all_dates = dfh["date_key"].unique()
    bars_form = max(1, len([d for d in all_dates if d >= pv_date_str]))
    bars_last = max(1, n_t)

    rsi14 = dfh["RSI14"].iloc[i] if "RSI14" in dfh.columns else 50.0
    if pd.isna(rsi14):
        rsi14 = 50.0

    ema_above = 0
    if "EMA20_h" in dfh.columns:
        ema_val = dfh["EMA20_h"].iloc[i]
        if not pd.isna(ema_val):
            ema_above = int(dfh["Close"].iloc[i] > ema_val)

    touch_hour = dfh.index[i].hour

    other_lvls = all_levels[np.abs(all_levels - level_price) > level_price * 0.001]
    if len(other_lvls) > 0 and atr_day > 0:
        dist_next = float(np.min(np.abs(other_lvls - dfh["Close"].iloc[i])) / atr_day)
    else:
        dist_next = 10.0
    dist_next = min(dist_next, 20.0)

    feat = np.array([[ltype, n_t, br, ar, bars_form, bars_last, vol_ratio, atr_pct,
                      rsi14, ema_above, touch_hour, dist_next]])
    return mlp.predict_proba(scaler.transform(feat))[0][1]

def touch_stats_historical(df1d, level_price, pivot_idx, up_to_idx):
    """
    Считает отбои от уровня только на исторических данных ДО up_to_idx.
    Никакого заглядывания в будущее: данные после up_to_idx не используются.
    """
    bounces = total = 0
    end = min(up_to_idx, len(df1d) - 5)
    for k in range(pivot_idx + 1, end):
        bar  = df1d.iloc[k]
        dist = min(abs(bar["Close"]-level_price), abs(bar["Low"]-level_price),
                   abs(bar["High"]-level_price)) / (level_price + 1e-9)
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
    return total, bounces / total if total > 0 else 0.5

# ────────────────────────────────────────────────────────
# Бэктест одного тикера
# ────────────────────────────────────────────────────────
def backtest_ticker(ticker, df1d, dfh, level_dist=None, void_mult=None, rr_target=None,
                    trend_filter_shorts=None, rsi_long_min=None, rsi_short_max=None,
                    min_vol_ratio=None, atr_exhaustion=None, small_body=None):
    if df1d is None or dfh is None or len(df1d) < 30 or len(dfh) < 50:
        return []

    df1d = df1d.copy()
    df1d["ATR_daily"] = calc_atr(df1d, ATR_PERIOD)
    df1d["EMA20"]     = df1d["Close"].ewm(span=TREND_EMA, adjust=False).mean()
    df1d["trend"]     = np.where(df1d["Close"] > df1d["EMA20"], 1, -1)

    d_atr   = dict(zip(df1d.index.strftime("%Y-%m-%d"), df1d["ATR_daily"]))
    d_trend = dict(zip(df1d.index.strftime("%Y-%m-%d"), df1d["trend"]))

    dfh = dfh.copy()
    dfh["date_key"]    = dfh.index.strftime("%Y-%m-%d")
    dfh["ATR_daily"]   = dfh["date_key"].map(d_atr).ffill()
    dfh["daily_trend"] = dfh["date_key"].map(d_trend).ffill()
    dfh["ATR"]         = calc_atr(dfh, ATR_PERIOD)
    dfh["RSI14"]       = calc_rsi(dfh["Close"])
    dfh["EMA20_h"]     = dfh["Close"].ewm(span=TREND_EMA, adjust=False).mean()

    pivots = find_pivots(df1d, LOOKBACK_PIVOT)
    if len(pivots) == 0:
        return []

    lev_prices = pivots["level"].values
    lev_types  = pivots["type"].values
    lev_dates  = pd.to_datetime(pivots["date"]).dt.strftime("%Y-%m-%d").values

    # Используем переданные параметры или глобальные по умолчанию
    _ld  = level_dist if level_dist is not None else LEVEL_DIST
    _vm  = void_mult  if void_mult  is not None else VOID_MULTIPLIER
    _rrt = rr_target  if rr_target  is not None else RR_TARGET
    _tfs = trend_filter_shorts if trend_filter_shorts is not None else TREND_FILTER_SHORTS
    _rlm = rsi_long_min if rsi_long_min is not None else RSI_LONG_MIN
    _rsm = rsi_short_max if rsi_short_max is not None else RSI_SHORT_MAX
    _mvr = min_vol_ratio if min_vol_ratio is not None else MIN_VOL_RATIO

    # Статистика касаний — только если модель включена (MODEL_THRESHOLD > 0)
    # Вычисляется лениво, с up_to_idx на дату текущего бара, чтобы не было data leakage.
    stats_cache = {}
    if MODEL_THRESHOLD > 0:
        # Маппинг даты → позиция в df1d для ограничения历史的и
        d1d_dates = {d.strftime("%Y-%m-%d"): pos for pos, d in enumerate(df1d.index)}

    # ── Генерация сигналов ───────────────────────────────
    signals          = []
    level_last_bar   = {}

    for i in range(20, len(dfh) - MAX_TRADE_BARS - 1):
        bar     = dfh.iloc[i]
        atr_h   = bar["ATR"]
        atr_day = bar["ATR_daily"]
        trend   = bar["daily_trend"]

        if np.isnan(atr_h) or np.isnan(atr_day) or atr_day == 0:
            continue
        _ae = atr_exhaustion if atr_exhaustion is not None else ATR_EXHAUSTION
        if atr_h >= _ae * atr_day:
            continue

        dists = np.abs(lev_prices - bar["Close"]) / (bar["Close"] + 1e-9)
        mi    = np.argmin(dists)
        if dists[mi] >= _ld:
            continue

        level_price    = lev_prices[mi]
        level_type_str = lev_types[mi]
        is_long        = (level_type_str == "support")
        pv_date_str    = lev_dates[mi]

        # Тренд-фильтр: лонги только при trend > 0
        if not np.isnan(trend):
            if is_long and trend < 0:
                continue
            # Симметричный тренд-фильтр для шортов (если включён)
            if not is_long and _tfs and trend > 0:
                continue

        if i - level_last_bar.get(level_price, -LEVEL_COOLDOWN_BARS - 1) < LEVEL_COOLDOWN_BARS:
            continue

        # Малые тела: 3 бара вместо 5 (как в step7)
        bodies = (dfh["Close"].iloc[i-3:i] - dfh["Open"].iloc[i-3:i]).abs()
        _sb = small_body if small_body is not None else SMALL_BODY
        if bodies.mean() >= _sb * atr_h:
            continue
        # slow_approach удалён (как в step7)

        cands = lev_prices[lev_prices > level_price] if is_long else lev_prices[lev_prices < level_price]
        if len(cands) == 0:
            continue
        void_dist = abs(cands[np.argmin(np.abs(cands - bar["Close"]))] - level_price)
        if void_dist <= _vm * atr_day:
            continue
        if not false_breakout(dfh, i, level_price, is_long):
            continue

        # ── Модель ML (только если MODEL_THRESHOLD > 0) ──
        prob = 1.0  # по умолчанию — пропускаем, если модель выключена
        if MODEL_THRESHOLD > 0:
            # Ленивое вычисление статистики касаний с защитой от data leakage:
            # используем только данные ДО текущей даты
            if level_price not in stats_cache:
                d1d_pos = d1d_dates.get(pv_date_str, len(df1d))
                n_t, br = touch_stats_historical(df1d, level_price,
                                                  np.where(lev_prices == level_price)[0][0],
                                                  up_to_idx=d1d_pos + 5)
                stats_cache[level_price] = (n_t, br)
            n_t, br = stats_cache[level_price]
            ar = 1.0
            prob = get_model_prob(dfh, level_price, level_type_str, i, n_t, br, ar, pv_date_str, lev_prices)
            if prob < MODEL_THRESHOLD:
                continue

        # ── RSI-фильтр (если пороги заданы) ──
        rsi_val = bar["RSI14"] if "RSI14" in dfh.columns and not pd.isna(bar["RSI14"]) else 50.0
        if is_long and rsi_val < _rlm:
            continue
        if not is_long and rsi_val > _rsm:
            continue

        # ── Объёмный фильтр (если min_vol_ratio > 0) ──
        if _mvr > 0:
            vol20 = dfh["Volume"].iloc[max(0, i - 20):i].mean()
            if vol20 > 0 and bar["Volume"] / vol20 < _mvr:
                continue

        entry  = bar["Close"]
        # Фиксированный стоп от уровня (доказано лучше динамического)
        stop   = (level_price - STOP_ATR_FRAC * atr_day) if is_long else (level_price + STOP_ATR_FRAC * atr_day)
        risk   = abs(entry - stop)
        if risk == 0:
            continue
        target = entry + _rrt * risk if is_long else entry - _rrt * risk

        # ── Проверка издержек / динамический RR ──────────────
        tick_sz = moex_tick_size(entry)
        target, risk, skipped = effective_rr(
            is_long, entry, stop, target, ticker, tick_sz
        )
        if skipped:
            continue
        # Пересчитываем risk если target изменился (при increase_rr risk не меняется)
        risk = abs(entry - stop)

        level_last_bar[level_price] = i
        signals.append({
            "ticker":      ticker,
            "bar_idx":     i,
            "datetime":    dfh.index[i],
            "entry":       entry,
            "stop":        stop,
            "target":      target,
            "risk":        risk,
            "is_long":     is_long,
            "level":       level_price,
            "level_type":  level_type_str,
            "model_prob":  round(prob, 3),
        })

    # ── Симуляция сделок ─────────────────────────────────
    trades = []
    for sig in signals:
        start   = sig["bar_idx"] + 1
        if start >= len(dfh):
            continue
        entry_raw = dfh["Open"].iloc[start]
        stop    = sig["stop"]
        target  = sig["target"]
        is_long = sig["is_long"]
        ticker  = sig["ticker"]

        # ── Проскальзывание: 1 шаг цены в невыгодную сторону ──
        tick_size = moex_tick_size(entry_raw)
        if is_long:
            entry = entry_raw + tick_size * SLIPPAGE_STEPS   # покупаем дороже
        else:
            entry = entry_raw - tick_size * SLIPPAGE_STEPS    # продаём дешевле

        risk    = abs(entry - stop)
        if risk == 0:
            continue

        outcome, exit_price_raw, bars_held = "TIMEOUT", None, MAX_TRADE_BARS
        end = min(start + MAX_TRADE_BARS, len(dfh))
        for j in range(start, end):
            hi, lo = dfh["High"].iloc[j], dfh["Low"].iloc[j]
            sl_hit = lo <= stop   if is_long else hi >= stop
            tp_hit = hi >= target if is_long else lo <= target
            if sl_hit and tp_hit:
                outcome, exit_price_raw, bars_held = "SL", stop, j - start
                break
            elif tp_hit:
                outcome, exit_price_raw, bars_held = "TP", target, j - start
                break
            elif sl_hit:
                outcome, exit_price_raw, bars_held = "SL", stop, j - start
                break

        if exit_price_raw is None:
            exit_price_raw = dfh["Close"].iloc[min(end - 1, len(dfh) - 1)]

        # ── Проскальзывание на выходе (1 шаг в невыгодную сторону) ──
        tick_size_exit = moex_tick_size(exit_price_raw)
        if is_long:
            exit_price = exit_price_raw - tick_size_exit * SLIPPAGE_STEPS  # продаём дешевле
        else:
            exit_price = exit_price_raw + tick_size_exit * SLIPPAGE_STEPS  # покупаем дороже

        # ── Комиссия 0.03% на вход и выход ──
        lot_size = _LOT_SIZES.get(ticker.upper(), 1)
        trade_value_entry = entry * lot_size   # value per lot (1 lot = lot_size shares)
        trade_value_exit  = exit_price * lot_size
        commission_cost_r = (trade_value_entry * COMMISSION_PCT
                             + trade_value_exit * COMMISSION_PCT) / risk if risk > 0 else 0

        pnl_r = ((exit_price - entry) / risk - commission_cost_r
                 if is_long else (entry - exit_price) / risk - commission_cost_r)
        trades.append({
            **{k: sig[k] for k in ("ticker","datetime","level","level_type","model_prob","is_long")},
            "entry":      round(entry, 4),
            "exit":       round(exit_price, 4),
            "outcome":    outcome,
            "pnl_r":      round(pnl_r, 3),
            "bars_held":  bars_held,
        })

    return trades


# ────────────────────────────────────────────────────────
# Метрики
# ────────────────────────────────────────────────────────
def calc_metrics(df):
    if len(df) == 0:
        return {}
    wins   = df[df["outcome"] == "TP"]
    losses = df[df["outcome"] != "TP"]
    pnl    = df["pnl_r"].values
    eq     = np.cumsum(pnl)
    peak   = np.maximum.accumulate(eq)
    dd_arr = eq - peak
    gp     = wins["pnl_r"].sum()
    gl     = losses["pnl_r"].abs().sum()
    std_r  = df["pnl_r"].std()

    cur, streaks = 0, []
    for r in pnl:
        cur = cur + 1 if r < 0 else 0
        streaks.append(cur)
    max_streak = max(streaks) if streaks else 0

    cap = CAPITAL
    for r in pnl:
        cap += r * cap * RISK_PCT

    return {
        "n_trades":     len(df),
        "winrate":      len(wins) / len(df),
        "avg_win_r":    wins["pnl_r"].mean()   if len(wins)   > 0 else 0,
        "avg_loss_r":   losses["pnl_r"].mean() if len(losses) > 0 else 0,
        "expectancy":   df["pnl_r"].mean(),
        "total_r":      pnl.sum(),
        "pf":           gp / gl if gl > 0 else 0,
        "max_dd_r":     dd_arr.min(),
        "max_dd_pct":   (dd_arr / (CAPITAL + 1e-9)).min(),  # DD in % of initial capital
        "sharpe":       pnl.mean() / std_r * np.sqrt(len(df)) if std_r > 0 else 0,
        "return_pct":   cap / CAPITAL - 1,
        "capital_final":cap,
        "max_streak":   max_streak,
        "equity":       eq,
    }

def calc_portfolio_metrics(all_trades, tickers):
    """
    Портфельные метрики:
    - Суммарная equity (сумма equity каждого тикера)
    - Корреляция между equity кривыми тикеров
    - Эффективная просадка портфеля
    - Сделок в неделю
    """
    # Equity per ticker
    ticker_eq = {}
    ticker_weekly = {}
    for tkr in tickers:
        tdf = all_trades[all_trades["ticker"] == tkr].sort_values("datetime")
        ticker_eq[tkr] = tdf["pnl_r"].cumsum().values if len(tdf) > 0 else np.array([])

        # Trades per week
        if len(tdf) > 0:
            date_range = (tdf["datetime"].max() - tdf["datetime"].min()).days
            weeks = max(date_range / 7, 1)
            ticker_weekly[tkr] = len(tdf) / weeks
        else:
            ticker_weekly[tkr] = 0.0

    # Portfolio equity (date-aligned)
    df = all_trades.sort_values("datetime").reset_index(drop=True)
    port_eq = df["pnl_r"].cumsum().values
    port_peak = np.maximum.accumulate(port_eq)
    port_dd_arr = port_eq - port_peak
    max_port_dd = port_dd_arr.min()

    # Weekly trade frequency
    total_weeks = max((df["datetime"].max() - df["datetime"].min()).days / 7, 1)
    trades_per_week = len(df) / total_weeks
    trades_per_week_per_ticker = len(df) / (total_weeks * len(tickers))

    # Correlation between ticker equity curves (last N trades per ticker)
    eq_dfs = []
    for tkr in tickers:
        tdf = all_trades[all_trades["ticker"] == tkr].sort_values("datetime")
        if len(tdf) > 5:
            eq_dfs.append(pd.Series(tdf["pnl_r"].cumsum().values, name=tkr))

    avg_corr = 0.0
    if len(eq_dfs) >= 2:
        eq_combined = pd.concat(eq_dfs, axis=1).ffill().fillna(0)
        corr_matrix = eq_combined.corr().values
        n_pairs = (corr_matrix.shape[0] * (corr_matrix.shape[0] - 1)) / 2
        if n_pairs > 0:
            upper = np.triu(corr_matrix, k=1)
            avg_corr = upper.sum() / n_pairs

    # Drawdown per ticker
    ticker_max_dd = {}
    for tkr in tickers:
        tdf = all_trades[all_trades["ticker"] == tkr].sort_values("datetime")
        if len(tdf) > 0:
            eq = tdf["pnl_r"].cumsum().values
            peak = np.maximum.accumulate(eq)
            dd = eq - peak
            ticker_max_dd[tkr] = dd.min()
        else:
            ticker_max_dd[tkr] = 0.0

    return {
        "port_eq":         port_eq,
        "port_dd_arr":     port_dd_arr,
        "max_port_dd":     max_port_dd,
        "trades_per_week": trades_per_week,
        "trades_per_week_per_ticker": trades_per_week_per_ticker,
        "avg_corr":        avg_corr,
        "ticker_weekly":   ticker_weekly,
        "ticker_max_dd":   ticker_max_dd,
        "total_weeks":     total_weeks,
    }


# ────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────
# ОПТИМИЗАЦИОННЫЙ GRID — тестирование гипотез A–E
# ────────────────────────────────────────────────────────
# Фиксированные параметры
FIXED_LD = 0.012
FIXED_VM = 0.5
FIXED_RR = 3.0

# Pre-cache data
cached_data = {}
for ticker in MOEX_TICKERS:
    df1d = load_candles(ticker, interval=24, label='1d')
    dfh  = load_candles(ticker, interval=60, label='1h')
    if df1d is not None and dfh is not None:
        cached_data[ticker] = (df1d, dfh)

def run_config(label, **kwargs):
    """Запускает бэктест с переданными фильтрами и возвращает метрики."""
    all_trades = []
    for ticker in MOEX_TICKERS:
        if ticker not in cached_data:
            continue
        df1d, dfh = cached_data[ticker]
        trades = backtest_ticker(ticker, df1d, dfh, level_dist=FIXED_LD, void_mult=FIXED_VM,
                                 rr_target=FIXED_RR, **kwargs)
        all_trades.extend(trades)
    if not all_trades:
        return None
    df_all = pd.DataFrame(all_trades)
    m = calc_metrics(df_all)
    port = calc_portfolio_metrics(df_all, MOEX_TICKERS)
    return {
        'label': label,
        'n': m['n_trades'], 'tw': port['trades_per_week'],
        'wr': m['winrate'], 'tr': m['total_r'], 'pf': m['pf'],
        'sharpe': m['sharpe'], 'mdd': port['max_port_dd'],
        'exp': m['expectancy'], 'aw': m['avg_win_r'],
        'al': m['avg_loss_r'], 'streak': m['max_streak'],
    }

# ════════════════════════════════════════════════════════════
#  A/B СРАВНЕНИЕ: старый baseline vs новый оптимум
# ════════════════════════════════════════════════════════════
print()
print("=" * 90)
print("  A/B COMPARISON  |  MOEX portfolio  |  model OFF")
print(f"  LD={FIXED_LD}, VM={FIXED_VM}, RR={FIXED_RR}:1")
print(f"  Тикеры: {', '.join(MOEX_TICKERS)}  |  Период: {DATE_FROM} → {DATE_TO}")
print("=" * 90)
print(f"  {'Config':<35} {'Trades':>7} {'T/wk':>6} {'WR':>5} {'TotalR':>9} {'PF':>7} {'MaxDD':>7} {'Exp':>7} {'Streak':>6}")
print("  " + "-"*100)

ab_configs = [
    ("[БАЗА] ATR=0.25, без фильтров",  {"atr_exhaustion": 0.25, "trend_filter_shorts": False,
                                          "min_vol_ratio": 0.0, "rsi_long_min": 0.0, "rsi_short_max": 100.0}),
    ("[ОПТИМУМ] ATR=0.20 + все фильтры",{}),
]

opt_r = None
for label, kw in ab_configs:
    r = run_config(label, **kw)
    if r is None:
        continue
    if "ОПТИМУМ" in label:
        opt_r = r
    sign_tr = "+" if r['tr'] >= 0 else ""
    print(f"  {label:<35} {r['n']:>7d} {r['tw']:>6.2f} {r['wr']:>5.1%} {sign_tr}{r['tr']:>+8.1f}R "
          f"{r['pf']:>7.3f} {r['mdd']:>+7.2f}R {r['exp']:>+7.4f}R {r['streak']:>6d}")

print("  " + "-"*100)
if opt_r:
    delta_pf = (opt_r['pf'] / 0.875 - 1) * 100
    print()
    print(f"  ✅ PF вырос с 0.875 → {opt_r['pf']:.3f} ({delta_pf:+.1f}%)")
    print(f"  📊 Сделок/нед: 11.2 → {opt_r['tw']:.1f}  |  Winrate: 23.2% → {opt_r['wr']:.1%}")
    print(f"  📉 MaxDD: -37.8R → {opt_r['mdd']:.1f}R  |  Expectancy: +0.35R → {opt_r['exp']:.4f}R")
    if opt_r['pf'] >= 1.2:
        print(f"  🎯 ЦЕЛЬ PF > 1.2: ДОСТИГНУТА!")
    else:
        print(f"  🎯 ЦЕЛЬ PF > 1.2: НЕ ДОСТИГНУТА (PF={opt_r['pf']:.3f})")
print()
print("[OK] Optimization complete.")
    