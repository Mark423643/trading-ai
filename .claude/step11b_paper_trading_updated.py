"""
Step 11b — Paper Trading модуль (боевая конфигурация).
Параметры из step_portfolio_final.py: ATR_EXHAUSTION=0.25, LEVEL_DIST=0.012,
SMALL_BODY=0.70, VOID_MULTIPLIER=0.2, STOP_ATR_FRAC=0.20, FB_LOOKBACK=5,
High/Low false_breakout, без тренд-фильтра шортов, MODEL_THRESHOLD=0.0.
Тикеры: SBER, GAZP, LKOH, VTBR, GMKN.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
import os
import time
import requests
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ── интеграция с брокером Финам ──────────────────────────────────────────────
# Включается через переменную окружения LIVE_TRADING=1
# Требует: pip install finam-trade-api
# Детали — см. broker_finam.py
try:
    from broker_finam import execute_signal as _finam_execute, LIVE_TRADING as _LIVE_TRADING
    _BROKER_AVAILABLE = True
except ImportError:
    _BROKER_AVAILABLE = False
    _LIVE_TRADING = False

try:
    from notify_telegram import send_signal as _notify_signal
    _NOTIFY_AVAILABLE = True
except ImportError:
    _NOTIFY_AVAILABLE = False


# ── логирование в файл ───────────────────────────────────
LOG_DIR  = "logs"
LOG_FILE = os.path.join(LOG_DIR, "trading.log")
os.makedirs(LOG_DIR, exist_ok=True)

class _Tee:
    """Пишет каждую строку одновременно в консоль и в лог-файл с timestamp."""
    def __init__(self, stream, fpath):
        self._s   = stream
        self._f   = open(fpath, "a", encoding="utf-8")
        self._buf = ""

    def write(self, data):
        self._s.write(data)
        self._buf += data
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._f.write(f"[{ts}] {line}\n")
        self._f.flush()

    def flush(self):
        self._s.flush()
        self._f.flush()

    def reconfigure(self, **kw):
        if hasattr(self._s, "reconfigure"):
            self._s.reconfigure(**kw)

sys.stdout = _Tee(sys.stdout, LOG_FILE)
sys.stderr = _Tee(sys.stderr, LOG_FILE)
# ─────────────────────────────────────────────────────────

DATA_DIR  = "data"
MODEL_DIR = "models"
SIG_FILE  = os.path.join(DATA_DIR, "signals_live.csv")
os.makedirs(DATA_DIR, exist_ok=True)

# ── торговый лист — 5 наиболее ликвидных MOEX ────────────
MOEX_TICKERS = ["SBER", "GAZP", "LKOH", "VTBR", "GMKN"]

# ── параметры стратегии (из step_portfolio_final.py) ──────
ATR_EXHAUSTION      = 0.20       # ужесточён с 0.25 — вход только в очень тихие бары
LEVEL_DIST          = 0.012      # расширен с 0.005
SMALL_BODY          = 0.70       # ослаблен с 0.40
VOID_MULTIPLIER     = 0.5        # ослаблен с 1.0
STOP_ATR_FRAC       = 0.20       # (не используется при dynamic_stop, оставлен для совместимости)
ATR_PERIOD          = 14
RR_TARGET           = 3.0
LOOKBACK_PIVOT      = 10
TREND_EMA           = 20
FB_LOOKBACK         = 5
LEVEL_COOLDOWN_BARS = 7
MODEL_THRESHOLD     = 0.0        # 0.0 = модель отключена, только уровни + фильтры
CHECK_LAST_N_BARS   = 300        # проверяем последние N часовых баров на сигналы

# ── дополнительные фильтры (оптимизированы) ──────────────
TREND_FILTER_SHORTS = True       # шорты только при дневном тренде вниз
MIN_VOL_RATIO       = 1.0        # объём бара > среднего за 20 периодов
RSI_LONG_MIN        = 40.0       # лонги только при RSI > 40
RSI_SHORT_MAX       = 60.0       # шорты только при RSI < 60

# ── модель (с отловом ошибки — модель может отсутствовать) ──
try:
    mlp    = joblib.load(os.path.join(MODEL_DIR, "mlp_levels.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    _MODEL_LOADED = True
except Exception:
    _MODEL_LOADED = False

# ───────────────────────────────────────────────────────
# Утилиты
# ───────────────────────────────────────────────────────
def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def calc_atr(df, period):
    prev = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev).abs(),
        (df["Low"]  - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calc_rsi(series, period=14):
    """Wilder RSI через EWM (идентично step5_moex_dataset)."""
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
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
    Идентично step_portfolio_final.py.
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

def get_model_prob(dfh, level_price, level_type_str, i, n_t, br, ar, pv_date_str, all_levels):
    if not _MODEL_LOADED:
        return 1.0  # модель не загружена — пропускаем проверку
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

def touch_stats_fast(df1d, level_price, pivot_idx):
    """Быстрый подсчёт отбоев после формирования уровня."""
    bounces = total = 0
    for k in range(pivot_idx + 1, len(df1d) - 5):
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

# ───────────────────────────────────────────────────────
# Загрузка данных
# ───────────────────────────────────────────────────────

MOEX_URL = ("https://iss.moex.com/iss/engines/stock/markets/shares"
            "/boards/TQBR/securities/{ticker}/candles.json")

def fetch_moex(ticker):
    """Свежие данные с MOEX ISS API: дневные 1г + часовые 7д."""
    date_to   = datetime.now().strftime("%Y-%m-%d")
    date_from_daily  = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    date_from_hourly = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    def get_candles(interval, date_from):
        url    = MOEX_URL.format(ticker=ticker)
        chunks = []
        start  = 0
        while True:
            try:
                r = requests.get(url, params={"interval": interval, "from": date_from,
                                               "till": date_to, "start": start}, timeout=20)
                data = r.json()
                rows = data["candles"]["data"]
                cols = data["candles"]["columns"]
                if not rows:
                    break
                chunks.append(pd.DataFrame(rows, columns=cols))
                start += len(rows)
                if len(rows) < 500:
                    break
                time.sleep(0.2)
            except Exception as e:
                print(f"  [MOEX] {ticker} interval={interval}: {e}")
                break

        if not chunks:
            return None
        df = pd.concat(chunks, ignore_index=True)
        df = df.rename(columns={"begin":"Datetime","open":"Open","high":"High",
                                  "low":"Low","close":"Close","volume":"Volume"})
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        return df.set_index("Datetime")[["Open","High","Low","Close","Volume"]].astype(float).dropna()

    df1d = get_candles(24, date_from_daily)
    dfh  = get_candles(60, date_from_hourly)
    return df1d, dfh

# ───────────────────────────────────────────────────────
# Ядро сканера — один тикер (синхронизировано с step_portfolio_final.py)
# ───────────────────────────────────────────────────────
def scan_ticker(ticker, df1d, dfh, is_moex=False):
    """
    Возвращает список словарей-сигналов (может быть пустым).
    Проверяет только последние CHECK_LAST_N_BARS баров.
    Фильтры синхронизированы с финальным бэктестом step_portfolio_final.py.
    """
    if df1d is None or dfh is None or len(df1d) < 30 or len(dfh) < 20:
        return []

    df1d = df1d.copy()
    df1d["ATR_daily"] = calc_atr(df1d, ATR_PERIOD)
    df1d["EMA20"]     = df1d["Close"].ewm(span=TREND_EMA, adjust=False).mean()
    df1d["trend"]     = np.where(df1d["Close"] > df1d["EMA20"], 1, -1)

    d_atr   = dict(zip(df1d.index.strftime("%Y-%m-%d"), df1d["ATR_daily"]))
    d_trend = dict(zip(df1d.index.strftime("%Y-%m-%d"), df1d["trend"]))

    dfh = dfh.copy()
    if is_moex:
        if dfh.index.tzinfo is None:
            dfh["date_key"] = dfh.index.strftime("%Y-%m-%d")
        else:
            dfh["date_key"] = dfh.index.tz_convert("Europe/Moscow").strftime("%Y-%m-%d")
    else:
        if dfh.index.tzinfo is not None:
            dfh["date_key"] = dfh.index.tz_convert("America/New_York").strftime("%Y-%m-%d")
        else:
            dfh["date_key"] = dfh.index.strftime("%Y-%m-%d")

    dfh["ATR_daily"]   = dfh["date_key"].map(d_atr).ffill()
    dfh["daily_trend"] = dfh["date_key"].map(d_trend).ffill()
    dfh["ATR"]         = calc_atr(dfh, ATR_PERIOD)
    dfh["RSI14"]       = calc_rsi(dfh["Close"])
    dfh["EMA20_h"]     = dfh["Close"].ewm(span=20, adjust=False).mean()

    pivots = find_pivots(df1d, LOOKBACK_PIVOT)
    if len(pivots) == 0:
        return []

    lev_prices = pivots["level"].values
    lev_types  = pivots["type"].values
    lev_dates  = pd.to_datetime(pivots["date"]).dt.strftime("%Y-%m-%d").values

    # Статистика касаний
    stats_cache = {}
    for idx, row in pivots.iterrows():
        n_t, br = touch_stats_fast(df1d, row["level"], idx)
        stats_cache[row["level"]] = (n_t, br, 1.0)

    level_last_bar = {}
    signals = []
    # Сканируем только последние CHECK_LAST_N_BARS баров
    scan_from = max(20, len(dfh) - CHECK_LAST_N_BARS)

    for i in range(scan_from, len(dfh)):
        # Фильтры по времени МСК (MOEX ISS отдаёт naive UTC → +3ч = МСК)
        bar_time_msk = dfh.index[i] + timedelta(hours=3)
        if bar_time_msk.weekday() == 3:   # четверг
            continue
        if bar_time_msk.hour >= 15:        # вечерняя сессия после 15:00 МСК
            continue

        bar     = dfh.iloc[i]
        atr_h   = bar["ATR"]
        atr_day = bar["ATR_daily"]
        trend   = bar["daily_trend"]

        if np.isnan(atr_h) or np.isnan(atr_day) or atr_day == 0:
            continue
        # ATR-фильтр: входим только в тихие бары
        if atr_h >= ATR_EXHAUSTION * atr_day:
            continue

        dists = np.abs(lev_prices - bar["Close"]) / (bar["Close"] + 1e-9)
        mi    = np.argmin(dists)
        if dists[mi] >= LEVEL_DIST:
            continue

        level_price    = lev_prices[mi]
        level_type_str = lev_types[mi]
        is_long        = (level_type_str == "support")
        pv_date_str    = lev_dates[mi]

        # Тренд-фильтр: лонги только при trend > 0
        if not np.isnan(trend):
            if is_long and trend < 0:
                continue
            # Симметричный тренд-фильтр для шортов
            if not is_long and TREND_FILTER_SHORTS and trend > 0:
                continue

        # Кулдаун уровня
        if i - level_last_bar.get(level_price, -LEVEL_COOLDOWN_BARS - 1) < LEVEL_COOLDOWN_BARS:
            continue

        # Малые тела: 3 бара вместо 5 (как в финальном бэктесте)
        bodies = (dfh["Close"].iloc[i-3:i] - dfh["Open"].iloc[i-3:i]).abs()
        if bodies.mean() >= SMALL_BODY * atr_h:
            continue
        # slow_approach удалён (как в финальном бэктесте)

        # Проверка void (пустоты между уровнями)
        cands = lev_prices[lev_prices > level_price] if is_long else lev_prices[lev_prices < level_price]
        if len(cands) == 0:
            continue
        void_dist = abs(cands[np.argmin(np.abs(cands - bar["Close"]))] - level_price)
        if void_dist <= VOID_MULTIPLIER * atr_day:
            continue

        if not false_breakout(dfh, i, level_price, is_long):
            continue

        n_t, br, ar = stats_cache.get(level_price, (0, 0.5, 1.0))
        prob = get_model_prob(dfh, level_price, level_type_str, i, n_t, br, ar, pv_date_str, lev_prices)
        if prob < MODEL_THRESHOLD:
            continue

        # ── RSI-фильтр ──
        rsi_val = bar["RSI14"] if "RSI14" in dfh.columns and not pd.isna(bar["RSI14"]) else 50.0
        if is_long and rsi_val < RSI_LONG_MIN:
            continue
        if not is_long and rsi_val > RSI_SHORT_MAX:
            continue

        # ── Объёмный фильтр ──
        if MIN_VOL_RATIO > 0:
            vol20 = dfh["Volume"].iloc[max(0, i - 20):i].mean()
            if vol20 > 0 and bar["Volume"] / vol20 < MIN_VOL_RATIO:
                continue

        entry  = bar["Close"]
        # Фиксированный стоп от уровня (доказано лучше динамического)
        stop   = (level_price - STOP_ATR_FRAC * atr_day) if is_long else (level_price + STOP_ATR_FRAC * atr_day)
        risk   = abs(entry - stop)
        if risk == 0:
            continue
        target = entry + RR_TARGET * risk if is_long else entry - RR_TARGET * risk

        level_last_bar[level_price] = i
        signals.append({
            "scan_time":  now_str(),
            "ticker":     ticker,
            "exchange":   "MOEX" if is_moex else "NASDAQ",
            "bar_time":   str(dfh.index[i]),
            "direction":  "LONG" if is_long else "SHORT",
            "level":      round(level_price, 4),
            "entry":      round(entry, 4),
            "stop":       round(stop, 4),
            "target":     round(target, 4),
            "risk":       round(risk, 4),
            "rr":         RR_TARGET,
            "model_prob": round(prob, 3),
            "atr_daily":  round(atr_day, 4),
            "trend":      "BULL" if trend == 1 else "BEAR",
        })

    return signals

# ───────────────────────────────────────────────────────
# Дедупликация сигналов
# ───────────────────────────────────────────────────────
def load_existing_signals():
    if os.path.exists(SIG_FILE):
        try:
            return pd.read_csv(SIG_FILE)
        except Exception:
            pass
    return pd.DataFrame()

def is_duplicate(sig, existing):
    """Сигнал считается дублем если тот же тикер+уровень появился в последние 7 часов."""
    if len(existing) == 0:
        return False
    same = existing[
        (existing["ticker"] == sig["ticker"]) &
        (existing["level"].round(2) == round(sig["level"], 2))
    ]
    if len(same) == 0:
        return False
    last_time = pd.to_datetime(same["scan_time"]).max()
    return (pd.Timestamp.now() - last_time).total_seconds() < 7 * 3600

# ───────────────────────────────────────────────────────
# ГЛАВНЫЙ ЦИКЛ
# ───────────────────────────────────────────────────────
_MODE_LABEL = "LIVE TRADING" if (_BROKER_AVAILABLE and _LIVE_TRADING) else "PAPER TRADING"

print("=" * 60)
print(f"  {_MODE_LABEL} SCANNER  |  {now_str()}")
print("=" * 60)
print(f"  MOEX TQBR: {', '.join(MOEX_TICKERS)}")
print(f"  ATR_EXHAUSTION={ATR_EXHAUSTION}  LEVEL_DIST={LEVEL_DIST}  MODEL={MODEL_THRESHOLD}")
print(f"  SMALL_BODY={SMALL_BODY}  VOID_MULT={VOID_MULTIPLIER}  STOP_ATR_FRAC={STOP_ATR_FRAC}")
print("=" * 60)

existing = load_existing_signals()
all_new_signals = []

# Сканируем MOEX TQBR
print("\n[MOEX TQBR]")
for ticker in MOEX_TICKERS:
    print(f"  {ticker}...", end=" ", flush=True)
    df1d, dfh = fetch_moex(ticker)
    if df1d is None or dfh is None:
        print("нет данных")
        continue
    sigs = scan_ticker(ticker, df1d, dfh, is_moex=True)
    new_sigs = [s for s in sigs if not is_duplicate(s, existing)]
    hourly_count = len(dfh) if dfh is not None else 0
    print(f"{hourly_count} баров hourly | сигналов: {len(sigs)}" +
          (f" ({len(new_sigs)} новых)" if len(sigs) != len(new_sigs) else ""))
    all_new_signals.extend(new_sigs)

# ───────────────────────────────────────────────────────
# Вывод сигналов
# ───────────────────────────────────────────────────────
print("\n" + "=" * 60)

if not all_new_signals:
    print("  Новых сигналов нет.")
else:
    print(f"  НАЙДЕНО СИГНАЛОВ: {len(all_new_signals)}")
    if _NOTIFY_AVAILABLE:
        for _sig in all_new_signals:
            try:
                _notify_signal(_sig)
            except Exception:
                pass
    print("=" * 60)
    for i, sig in enumerate(all_new_signals, 1):
        dir_emoji = "^" if sig["direction"] == "LONG" else "v"
        print(f"\n  [{i}] {sig['ticker']} ({sig['exchange']})  {dir_emoji} {sig['direction']}")
        print(f"      Время бара:  {sig['bar_time']}")
        print(f"      Уровень:     {sig['level']}")
        print(f"      Вход:        {sig['entry']}")
        print(f"      Стоп:        {sig['stop']}  (риск: {sig['risk']:.4f})")
        print(f"      Цель:        {sig['target']}  (R:R {sig['rr']}:1)")
        print(f"      Модель:      {sig['model_prob']:.3f}  |  Тренд: {sig['trend']}")

    # ── исполнение через брокера Финам ──────────────────────────────────────
    if _BROKER_AVAILABLE and _LIVE_TRADING:
        print(f"\n  Отправка ордеров в Финам...")
        ok_count = 0
        for sig in all_new_signals:
            success = _finam_execute(sig)
            status  = "OK" if success else "ОШИБКА"
            print(f"    {sig['ticker']} {sig['direction']} → {status}")
            if success:
                ok_count += 1
        print(f"  Исполнено: {ok_count}/{len(all_new_signals)}")
    elif _LIVE_TRADING and not _BROKER_AVAILABLE:
        print("\n  [WARN] LIVE_TRADING=1, но broker_finam.py не найден рядом со скриптом.")

    # Сохраняем в CSV
    new_df = pd.DataFrame(all_new_signals)
    if len(existing) > 0:
        result = pd.concat([existing, new_df], ignore_index=True)
    else:
        result = new_df
    result.to_csv(SIG_FILE, index=False)
    print(f"\n  Сохранено в: {SIG_FILE}")
    print(f"  Всего записей в файле: {len(result)}")

try:
    from notify_telegram import send_daily_summary
    tickers_with_signals = list(set([s["ticker"] for s in all_new_signals]))
    send_daily_summary(len(MOEX_TICKERS), len(all_new_signals), tickers_with_signals)
except Exception:
    pass

print("\n" + "=" * 60)
print(f"  Сканирование завершено: {now_str()}")
print(f"  Следующий запуск: через 1 час")
print("=" * 60)
