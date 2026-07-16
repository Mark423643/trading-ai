"""
Step 11b — Paper Trading модуль (MOEX, H1, чистая геометрия уровней).
Параметры синхронизированы с откалиброванной стратегией step_portfolio_final.py.

Запускать вручную или по расписанию (Task Scheduler / cron) раз в час.
Скачивает свежие данные по всем тикерам, прогоняет через фильтры,
выводит сигналы в консоль, дописывает в signals_live.csv,
отслеживает открытые позиции и отправляет уведомления через ntfy.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
import os
import time
import math
import requests
import joblib
from datetime import datetime, timedelta, timezone
import warnings
from dotenv import load_dotenv
warnings.filterwarnings("ignore")

# ── ПРИНУДИТЕЛЬНАЯ загрузка .env ──────────────────────────────
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
if os.path.exists(_env_path):
    load_dotenv(_env_path, override=True)
else:
    load_dotenv(override=True)
# ──────────────────────────────────────────────────────────────

# ── интеграция с брокером Финам ──────────────────────────────
try:
    from broker_finam import execute_signal as _finam_execute, LIVE_TRADING as _LIVE_TRADING, FINAM_TOKEN, _symbol
    _BROKER_AVAILABLE = True
except Exception:
    _BROKER_AVAILABLE = False
    _LIVE_TRADING = False
    FINAM_TOKEN = ""
    _symbol = None

# ── ntfy уведомления ─────────────────────────────────────────
try:
    from notify_ntfy import (
        send_open_signal as _ntfy_open,
        send_close_signal as _ntfy_close,
        send_portfolio_radar as _ntfy_radar,
        generate_status_comment as _ntfy_comment,
    )
    _NTFY_AVAILABLE = True
except Exception as e:
    print(f"  [WARN] notify_ntfy не загружен: {e}")
    _NTFY_AVAILABLE = False

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
HIST_FILE = os.path.join(DATA_DIR, "trades_history.csv")
os.makedirs(DATA_DIR, exist_ok=True)

from config_trading import (
    ATR_EXHAUSTION, LEVEL_DIST, SMALL_BODY, VOID_MULTIPLIER,
    STOP_ATR_FRAC, FB_LOOKBACK, APPROACH_BARS, ATR_PERIOD, TREND_EMA,
    LOOKBACK_PIVOT, RR_TARGET, LEVEL_COOLDOWN_BARS,
    MODEL_THRESHOLD, MODEL_PROB_MAX,
    TREND_FILTER_SHORTS, MIN_VOL_RATIO, RSI_LONG_MIN, RSI_SHORT_MAX,
    TRADE_HOURS_BLOCK, TRADE_DAYS_BLOCK,
    MOEX_TICKERS,
    MOEX_FUTURES,
    MOEX_FUTURES_PERPETUAL,
)

CHECK_LAST_N_BARS   = 720        # последние 30 дней H1-баров (для сканирования/статистики)

# ── ГЕЙТ СВЕЖЕСТИ ДЛЯ РЕАЛЬНЫХ ОРДЕРОВ ──────────────────────────
# Реальный ордер выставляется ТОЛЬКО по сигналу с последнего бара, и только
# если этот бар не старше FRESH_MAX_AGE_HOURS часов. Без этого гейта сигнал
# из прошлого (напр. бар от 18 июня) выстрелил бы рыночным ордером по текущей
# цене со старым стопом — это и была причина слива на комиссиях (BANEP вошёл
# по 922.5 со стопом из июня 918.96).
FRESH_MAX_AGE_HOURS = 2.0

# ── загрузка модели (не используется при MODEL_THRESHOLD=0.0) ──
_ML_AVAILABLE = False
if MODEL_THRESHOLD > 0:
    try:
        mlp    = joblib.load(os.path.join(MODEL_DIR, "mlp_levels.pkl"))
        scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        _ML_AVAILABLE = True
    except Exception:
        print("  [WARN] Модель не загружена (файлы .pkl не найдены), "
              "но MODEL_THRESHOLD=0.0 — это не критично.")


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

def detect_lp(dfh, i, level_price, is_long, lookback=FB_LOOKBACK):
    """
    Ищет ЛП (ложный пробой) в последних `lookback` барах.
    ЛП лонг (поддержка): тень пробила ниже уровня, тело закрылось выше.
    ЛП шорт (сопротивление): тень пробила выше уровня, тело закрылось ниже.
    Дополнительно проверяет, что 3 бара ДО ЛП не пробивали уровень —
    т.е. это именно первый укол, а не продолжение пробоя.
    Возвращает (found: bool, lp_bar_idx: int).
    """
    if i < 4:
        return False, -1
    for j in range(max(i - lookback, 3), i + 1):
        lo = dfh["Low"].iloc[j]
        hi = dfh["High"].iloc[j]
        cl = dfh["Close"].iloc[j]
        op = dfh["Open"].iloc[j]
        if is_long:
            if lo < level_price and cl > level_price and op > level_price:
                # 3 бара до ЛП не пробивали уровень снизу
                pre = dfh["Low"].iloc[j-3:j]
                if (pre >= level_price * 0.998).all():
                    return True, j
        else:
            if hi > level_price and cl < level_price and op < level_price:
                # 3 бара до ЛП не пробивали уровень сверху
                pre = dfh["High"].iloc[j-3:j]
                if (pre <= level_price * 1.002).all():
                    return True, j
    return False, -1

def get_model_prob(dfh, level_price, level_type_str, i, n_t, br, ar, pv_date_str, all_levels):
    if not _ML_AVAILABLE:
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
    if pd.isna(rsi14): rsi14 = 50.0
    ema_above = 0
    if "EMA20_h" in dfh.columns:
        ema_val = dfh["EMA20_h"].iloc[i]
        if not pd.isna(ema_val): ema_above = int(dfh["Close"].iloc[i] > ema_val)
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

def moex_tick_size(price: float) -> float:
    """Минимальный шаг цены для MOEX TQBR."""
    if price < 10: return 0.01
    elif price < 25: return 0.02
    elif price < 100: return 0.05
    elif price < 500: return 0.10
    elif price < 1000: return 0.50
    else: return 1.00

def touch_stats_fast(df1d, level_price, pivot_idx):
    bounces = total = 0
    for k in range(pivot_idx + 1, len(df1d) - 5):
        bar  = df1d.iloc[k]
        dist = min(abs(bar["Close"]-level_price), abs(bar["Low"]-level_price),
                   abs(bar["High"]-level_price)) / (level_price + 1e-9)
        if dist < 0.003:
            future = df1d["Close"].iloc[k+1:k+6]
            if len(future) < 5: break
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
                if not rows: break
                chunks.append(pd.DataFrame(rows, columns=cols))
                start += len(rows)
                if len(rows) < 500: break
                time.sleep(0.2)
            except Exception as e:
                print(f"  [MOEX] {ticker} interval={interval}: {e}")
                break
        if not chunks: return None
        df = pd.concat(chunks, ignore_index=True)
        df = df.rename(columns={"begin":"Datetime","open":"Open","high":"High",
                                  "low":"Low","close":"Close","volume":"Volume"})
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        return df.set_index("Datetime")[["Open","High","Low","Close","Volume"]].astype(float).dropna()

    df1d = get_candles(24, date_from_daily)
    dfh  = get_candles(60, date_from_hourly)
    return df1d, dfh


def fetch_current_price(ticker: str, client=None) -> float | None:
    """Получает текущую live-цену через Finam API (get_last_quote).

    Возвращает float (Last Price) или None при ошибке/отсутствии токена/
    невалидном значении (NaN/Inf/0/stale timestamp).
    Если передан client (finam_trade_api.Client), использует его —
    иначе создаёт новый (медленнее при множественных вызовах).
    """
    if not _BROKER_AVAILABLE or not FINAM_TOKEN or _symbol is None:
        return None

    async def _get():
        try:
            symbol = _symbol(ticker, "MOEX")
            if client is not None:
                quote = await client.instruments.get_last_quote(symbol)
            else:
                from finam_trade_api import Client, TokenManager
                _c = Client(TokenManager(FINAM_TOKEN))
                quote = await _c.instruments.get_last_quote(symbol)

            price = float(quote.quote.last)

            # ── Валидация ────────────────────────────────────────
            if math.isnan(price) or math.isinf(price) or price <= 0:
                return None

            ts = quote.quote.timestamp
            if ts is not None:
                now_aware = datetime.now(ts.tzinfo) if ts.tzinfo else datetime.now()
                if (now_aware - ts).total_seconds() > 300:
                    return None  # цена устарела

            return price
        except Exception:
            return None

    import asyncio
    return asyncio.run(_get())


# ───────────────────────────────────────────────────────
# Фьючерсы MOEX FORTS
# ───────────────────────────────────────────────────────

FUTURES_URL = ("https://iss.moex.com/iss/engines/futures/markets/"
               "forts/boards/RFUD/securities/{ticker}/candles.json")

def fetch_futures(ticker):
    """Загружает дневные и часовые свечи для фьючерса."""
    date_to   = datetime.now().strftime("%Y-%m-%d")
    date_from_daily  = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    date_from_hourly = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")

    def get_candles(interval, date_from):
        url = FUTURES_URL.format(ticker=ticker)
        chunks = []; start = 0
        while True:
            try:
                r = requests.get(url, params={"interval": interval, "from": date_from,
                                               "till": date_to, "start": start}, timeout=20)
                data = r.json()
                rows = data["candles"]["data"]
                cols = data["candles"]["columns"]
                if not rows: break
                chunks.append(pd.DataFrame(rows, columns=cols))
                start += len(rows)
                if len(rows) < 500: break
                time.sleep(0.2)
            except Exception as e:
                print(f"  [FUT] {ticker} interval={interval}: {e}")
                break
        if not chunks: return None
        df = pd.concat(chunks, ignore_index=True)
        # futures имеют колонки begin/end вместо begin
        df = df.rename(columns={"begin":"Datetime","open":"Open","high":"High",
                                "low":"Low","close":"Close","volume":"Volume"})
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        return df.set_index("Datetime")[["Open","High","Low","Close","Volume"]].astype(float).dropna()

    df1d = get_candles(24, date_from_daily)
    dfh  = get_candles(60, date_from_hourly)
    return df1d, dfh


# ───────────────────────────────────────────────────────
# Управление позициями — SL/TP трекинг
# ───────────────────────────────────────────────────────

# Исторические PF тикеров (из бэктеста step_portfolio_final.py)
_HISTORICAL_PF = {
    "SBER": 0.497, "GAZP": 0.660, "LKOH": 0.813, "GMKN": 0.739,
    "TATN": 1.496, "MOEX": 1.192, "MAGN": 1.507, "NVTK": 1.198,
    "ALRS": 1.211, "PHOR": 0.659, "AFKS": 2.938, "CBOM": 1.684,
    "TRNFP": 1.955,
    "VTBR": 2.440, "MGNT": 2.610, "NLMK": 2.000,
    "AFLT": 2.310, "CHMF": 1.280, "FEES": 19.96,
    "HYDR": 2.170, "IRAO": 2.300, "PIKK": 1.300,
    "RUAL": 1.590, "SNGS": 1.930,
    "AKRN": 3.000, "BANEP": 1.420, "BSPB": 3.780,
    "CHGZ": 1.500, "CNRU": 1.500, "DATA": 83.14,
    "GLRX": 1.660, "LSRG": 1.220, "MVID": 1.860,
    "PLZL": 3.000, "RNFT": 6.000, "SELG": 6.590,
    "SPBE": 1.620, "TATNP": 4.050, "VKCO": 3.000,
    # ── Фьючерсы ─────────────────────────────────────────────────
    "MXU6": 1.500,  # Мини-индекс Мосбиржи
    "GDU6": 1.500,  # Золото
}


def load_positions() -> pd.DataFrame:
    """Загружает открытые позиции из signals_live.csv."""
    if os.path.exists(SIG_FILE):
        try:
            df = pd.read_csv(SIG_FILE)
            if len(df) > 0:
                # Убираем дубликаты: оставляем последнюю запись по тикеру+уровню
                df = df.sort_values("scan_time").drop_duplicates(
                    subset=["ticker", "level"], keep="last"
                ).reset_index(drop=True)
            return df
        except Exception:
            pass
    return pd.DataFrame()


def load_history() -> pd.DataFrame:
    """Загружает историю закрытых сделок."""
    if os.path.exists(HIST_FILE):
        try:
            return pd.read_csv(HIST_FILE)
        except Exception:
            pass
    return pd.DataFrame()


def check_positions(positions: pd.DataFrame) -> list[dict]:
    """
    Проверяет все открытые позиции на SL/TP.
    Использует Finam API для live-цен с защитой от дребезга.
    Возвращает список закрытых позиций с результатами.
    """
    closed = []
    if len(positions) == 0:
        return closed

    # ── Инициализируем один клиент Финам для всех цен ────────
    _finam_client = None
    if _BROKER_AVAILABLE and FINAM_TOKEN:
        try:
            from finam_trade_api import Client, TokenManager
            _finam_client = Client(TokenManager(FINAM_TOKEN))
        except Exception:
            _finam_client = None

    unique_tickers = positions["ticker"].unique()
    prices = {}
    for t in unique_tickers:
        # ── Двойная верификация цены (защита от шпилек) ──
        _p1 = fetch_current_price(t, client=_finam_client)
        if _p1 is not None and _p1 > 0:
            time.sleep(0.3)
            _p2 = fetch_current_price(t, client=_finam_client)
            if _p2 is not None and _p2 > 0:
                delta = abs(_p2 - _p1) / _p1
                if delta < 0.01:
                    prices[t] = _p2
                else:
                    prices[t] = (_p1 + _p2) / 2
                    print(f"  [SPIKE] {t}: {_p1:.2f}→{_p2:.2f} (ср={prices[t]:.2f})")
            else:
                prices[t] = _p1  # второй запрос упал — берём первый
        time.sleep(0.15)

    for idx, row in positions.iterrows():
        ticker = row["ticker"]
        current = prices.get(ticker)
        if current is None:
            continue

        entry = float(row["entry"])
        stop = float(row["stop"])
        target = float(row["target"])
        direction = row["direction"]
        is_long = direction == "LONG"
        risk = abs(entry - stop)
        if risk == 0:
            continue

        # Проверка SL/TP
        if is_long:
            sl_hit = current <= stop
            tp_hit = current >= target
        else:
            sl_hit = current >= stop
            tp_hit = current <= target

        if tp_hit:
            status = "TAKE_PROFIT"
            exit_price = target
            pnl_r = (target - entry) / risk if is_long else (entry - target) / risk
        elif sl_hit:
            status = "STOP_LOSS"
            exit_price = stop
            pnl_r = (stop - entry) / risk if is_long else (entry - stop) / risk
        else:
            continue

        closed.append({
            "row_idx": idx,
            "ticker": ticker,
            "direction": direction,
            "entry": entry,
            "exit_price": exit_price,
            "status": status,
            "pnl_r": pnl_r,
            "risk": risk,
            "level": row.get("level", ""),
            "scan_time": row.get("scan_time", ""),
            "close_time": now_str(),
        })
        print(f"  [CLOSE] {ticker} {direction}: {status} (PnL={pnl_r:+.2f}R)")

    return closed


def save_closed_and_active(positions: pd.DataFrame, closed: list[dict]):
    """Удаляет закрытые позиции из active, добавляет их в history."""
    if not closed:
        return positions

    # Сохраняем в историю
    history = load_history()
    new_history = []
    for c in closed:
        new_history.append({
            "ticker": c["ticker"],
            "direction": c["direction"],
            "entry": c["entry"],
            "exit_price": c["exit_price"],
            "status": c["status"],
            "pnl_r": round(c["pnl_r"], 3),
            "risk": c["risk"],
            "level": c["level"],
            "open_time": c["scan_time"],
            "close_time": c["close_time"],
        })
    if new_history:
        hist_df = pd.DataFrame(new_history)
        if len(history) > 0:
            history = pd.concat([history, hist_df], ignore_index=True)
        else:
            history = hist_df
        history.to_csv(HIST_FILE, index=False)
        print(f"  [HISTORY] {len(new_history)} сделок сохранено в {HIST_FILE}")

    # Удаляем закрытые позиции из активных
    closed_indices = [c["row_idx"] for c in closed]
    active = positions.drop(index=closed_indices, errors="ignore").reset_index(drop=True)
    active.to_csv(SIG_FILE, index=False)
    print(f"  [ACTIVE] {len(active)} позиций остаётся в {SIG_FILE}")

    return active


# ───────────────────────────────────────────────────────
# Ядро сканера — один тикер (MOEX only)
# ───────────────────────────────────────────────────────
def scan_ticker(ticker, df1d, dfh):
    if df1d is None or dfh is None or len(df1d) < 30 or len(dfh) < 20:
        return []

    df1d = df1d.copy()
    df1d["ATR_daily"] = calc_atr(df1d, ATR_PERIOD)
    df1d["EMA20"]     = df1d["Close"].ewm(span=TREND_EMA, adjust=False).mean()
    df1d["trend"]     = np.where(df1d["Close"] > df1d["EMA20"], 1, -1)

    d_atr   = dict(zip(df1d.index.strftime("%Y-%m-%d"), df1d["ATR_daily"]))
    d_trend = dict(zip(df1d.index.strftime("%Y-%m-%d"), df1d["trend"]))

    dfh = dfh.copy()
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

    stats_cache = {}
    for idx, row in pivots.iterrows():
        n_t, br = touch_stats_fast(df1d, row["level"], idx)
        stats_cache[row["level"]] = (n_t, br, 1.0)

    signals = []
    scan_from = max(20, len(dfh) - CHECK_LAST_N_BARS)

    # ── Живая цена из Finam (для последнего, самого актуального бара) ──
    _live_price = None
    if _BROKER_AVAILABLE and FINAM_TOKEN:
        _live_price = fetch_current_price(ticker)

    for i in range(scan_from, len(dfh)):
        bar     = dfh.iloc[i]
        atr_h   = bar["ATR"]
        atr_day = bar["ATR_daily"]
        trend   = bar["daily_trend"]

        if np.isnan(atr_h) or np.isnan(atr_day) or atr_day == 0:
            continue
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

        if not np.isnan(trend):
            if is_long and trend < 0:
                continue
            if not is_long and TREND_FILTER_SHORTS and trend > 0:
                continue

        if i - level_last_bar.get(level_price, -LEVEL_COOLDOWN_BARS - 1) < LEVEL_COOLDOWN_BARS:
            continue

        # ── ЛП: тень пробила уровень одним баром, тело закрылось обратно ──
        lp_found, lp_j = detect_lp(dfh, i, level_price, is_long)
        if not lp_found:
            continue

        # ── Медленный подход: тела баров ДО ЛП маленькие (нет импульса) ──
        _ap_start = max(0, lp_j - APPROACH_BARS)
        bodies = (dfh["Close"].iloc[_ap_start:lp_j] - dfh["Open"].iloc[_ap_start:lp_j]).abs()
        if len(bodies) > 0 and bodies.mean() >= SMALL_BODY * atr_h:
            continue

        # ── Впереди пусто: до следующего уровня есть запас хода ──
        cands = lev_prices[lev_prices > level_price] if is_long else lev_prices[lev_prices < level_price]
        if len(cands) == 0:
            continue
        void_dist = abs(cands[np.argmin(np.abs(cands - bar["Close"]))] - level_price)
        if void_dist <= VOID_MULTIPLIER * atr_day:
            continue

        n_t, br, ar = stats_cache.get(level_price, (0, 0.5, 1.0))
        prob = get_model_prob(dfh, level_price, level_type_str, i, n_t, br, ar, pv_date_str, lev_prices)
        if prob < MODEL_THRESHOLD:
            continue
        if prob > MODEL_PROB_MAX:
            continue

        rsi_val = bar["RSI14"] if "RSI14" in dfh.columns and not pd.isna(bar["RSI14"]) else 50.0
        if is_long and rsi_val < RSI_LONG_MIN:
            continue
        if not is_long and rsi_val > RSI_SHORT_MAX:
            continue

        if MIN_VOL_RATIO > 0:
            vol20 = dfh["Volume"].iloc[max(0, i - 20):i].mean()
            if vol20 > 0 and bar["Volume"] / vol20 < MIN_VOL_RATIO:
                continue

        # ── Стоп за тенью ЛП-бара (метод Тимура: стоп 10% ATR за тень) ──
        if is_long:
            shadow_extreme = dfh["Low"].iloc[lp_j]
            stop = min(shadow_extreme, level_price) - STOP_ATR_FRAC * atr_day
        else:
            shadow_extreme = dfh["High"].iloc[lp_j]
            stop = max(shadow_extreme, level_price) + STOP_ATR_FRAC * atr_day

        # ── Цена входа: живая цена из Finam для последнего бара ──
        is_last_bar = (i == len(dfh) - 1)

        if is_last_bar and _live_price is not None and _live_price > 0:
            entry = _live_price
            # Если живая цена ушла за уровень — сетап недействителен
            stop_invalid = (is_long and stop >= entry) or (not is_long and stop <= entry)
            if stop_invalid:
                print(f"      [SKIP] {ticker}: живая цена {entry:.4f} пробила уровень "
                      f"{level_price:.4f} — разворотный сетап недействителен")
                continue
            print(f"      [LIVE] {ticker}: entry={bar['Close']:.4f}(close) → {entry:.4f}(live)")
        else:
            entry = bar["Close"]

        risk   = abs(entry - stop)
        if risk == 0:
            continue
        target = entry + RR_TARGET * risk if is_long else entry - RR_TARGET * risk

        level_last_bar[level_price] = i

        # ── ГЕЙТ СВЕЖЕСТИ: в signals (→ реальные ордера) попадают ТОЛЬКО ──
        # сигналы с последнего бара, не старше FRESH_MAX_AGE_HOURS. Исторические
        # бары сканируются для статистики, но НЕ торгуются.
        _bar_ts = pd.to_datetime(dfh.index[i])
        if _bar_ts.tzinfo is not None:
            _bar_ts = _bar_ts.tz_localize(None)
        _age_h = (pd.Timestamp.now() - _bar_ts).total_seconds() / 3600.0
        if not (is_last_bar and _age_h <= FRESH_MAX_AGE_HOURS):
            continue

        signals.append({
            "scan_time":  now_str(),
            "ticker":     ticker,
            "exchange":   "MOEX",
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

def is_duplicate(sig, existing, history=None, seen_tickers_in_run: set = None):
    """Проверяет, был ли сигнал уже отправлен.

    - Внутри одного запуска: одинаковый тикер не дублируется.
    - Один и тот же bar_time+level+ticker = ВСЕГДА дубликат (независимо от времени).
    - В истории закрытых: кулдаун 1 день.
    """
    # ── Внутри одного запуска ──────────────────────────────────
    if seen_tickers_in_run is not None and sig["ticker"] in seen_tickers_in_run:
        return True

    # ── Тот же бар + тот же уровень = дубликат всегда ──────────
    # Предотвращает повторный вход на одном и том же историческом баре
    # между разными hourly-запусками cron (даже через 14+ часов).
    if len(existing) > 0 and "bar_time" in existing.columns:
        same_bar = existing[
            (existing["ticker"] == sig["ticker"]) &
            (existing["level"].round(2) == round(sig["level"], 2)) &
            (existing["bar_time"].astype(str) == str(sig.get("bar_time", "")))
        ]
        if len(same_bar) > 0:
            return True

    # ── Среди открытых позиций: кулдаун 24 часа по тикеру+уровню ─
    if len(existing) > 0:
        same = existing[
            (existing["ticker"] == sig["ticker"]) &
            (existing["level"].round(2) == round(sig["level"], 2))
        ]
        if len(same) > 0:
            last_time = pd.to_datetime(same["scan_time"]).max()
            if (pd.Timestamp.now() - last_time).total_seconds() < 24 * 3600:
                return True

    # ── В истории закрытых ─────────────────────────────────────
    if history is not None and len(history) > 0:
        same_hist = history[
            (history["ticker"] == sig["ticker"]) &
            (history["level"].round(2) == round(sig["level"], 2))
        ]
        if len(same_hist) > 0:
            last_close = pd.to_datetime(same_hist["close_time"]).max()
            if (pd.Timestamp.now() - last_close).total_seconds() < 86400 * 1:
                return True

    return False


# ───────────────────────────────────────────────────────
# ГЛАВНЫЙ ЦИКЛ
# ───────────────────────────────────────────────────────
level_last_bar = {}   # глобальный кулдаун-словарь

_MODE_LABEL = "LIVE TRADING" if (_BROKER_AVAILABLE and _LIVE_TRADING) else "PAPER TRADING"

print("=" * 60)
print(f"  {_MODE_LABEL} SCANNER  |  {now_str()}")
print("=" * 60)
print(f"  MOEX TQBR: {', '.join(MOEX_TICKERS)}")
print(f"  FORTS:     {', '.join(list(MOEX_FUTURES) + list(MOEX_FUTURES_PERPETUAL))}")
print(f"  Параметры: LEVEL_DIST={LEVEL_DIST}, VM={VOID_MULTIPLIER}, "
      f"STOP_ATR_FRAC={STOP_ATR_FRAC}, RR={RR_TARGET}")
print(f"  ATR-фильтр: hourly < {ATR_EXHAUSTION} * daily  |  "
      f"Модель: {'OFF' if MODEL_THRESHOLD == 0 else f'ON (≥{MODEL_THRESHOLD})'}")
print(f"  Проверка последних {CHECK_LAST_N_BARS} баров")
print(f"  NTFY: {'ON' if _NTFY_AVAILABLE else 'OFF'}")
print("=" * 60)

# ── Фильтр времени (Вариант В) ───────────────────────────────
_MSK = timezone(timedelta(hours=3))
_now_msk = datetime.now(_MSK)
_DAY_NAMES = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]

if _now_msk.weekday() in TRADE_DAYS_BLOCK:
    _dn = _DAY_NAMES[_now_msk.weekday()]
    print(f"  [FILTER] {_dn} — день заблокирован (WR слабый по бэктесту). Выходим.")
    sys.exit(0)

if _now_msk.hour in TRADE_HOURS_BLOCK:
    print(f"  [FILTER] {_now_msk.hour}:00 МСК — час заблокирован (WR слабый по бэктесту). Выходим.")
    sys.exit(0)

print(f"  Время скана: {_now_msk.strftime('%H:%M МСК')} ({_DAY_NAMES[_now_msk.weekday()]})")
# ─────────────────────────────────────────────────────────────

# ── Шаг 1: Загружаем открытые позиции и проверяем SL/TP ──
print("\n[TRACKING] Проверка открытых позиций...")
positions = load_positions()
print(f"  Открытых позиций: {len(positions)}")

closed_positions = check_positions(positions)
for cp in closed_positions:
    print(f"  ⚡ {cp['ticker']} {cp['direction']}: {cp['status']} (PnL={cp['pnl_r']:+.2f}R)")
    # Отправляем Template B
    if _NTFY_AVAILABLE:
        _ntfy_close(cp, cp["exit_price"], cp["status"], cp["pnl_r"], risk_rub=100.0)
    # Локальный лог
    dir_emoji = "🟢" if cp["direction"] == "LONG" else "🔴"
    print(f"\n  {dir_emoji} ЗАКРЫТИЕ: {cp['ticker']} ({cp['direction']})")
    print(f"      Выход: {cp['exit_price']:.4f} | Статус: {cp['status']}")
    print(f"      PnL: {cp['pnl_r']:+.2f}R")

# Обновляем файлы
positions = save_closed_and_active(positions, closed_positions)

# ── Шаг 2: Сканируем новые сигналы ──
existing = positions if len(positions) > 0 else pd.DataFrame()
history_df = load_history()  # для сверки дубликатов по истории
all_new_signals = []
_seen_tickers_this_run = set()  # внутри одного запуска: тикер → не дублируем

print("\n[MOEX TQBR]")
for ticker in MOEX_TICKERS:
    print(f"  {ticker}...", end=" ", flush=True)
    df1d, dfh = fetch_moex(ticker)
    if df1d is None or dfh is None:
        print("нет данных")
        continue
    sigs = scan_ticker(ticker, df1d, dfh)
    # Поэлементная фильтрация: первый пропущенный сигнал по тикеру блокирует
    # все остальные в этом же запуске (даже если пришли из разных уровней)
    new_sigs = []
    for s in sigs:
        if is_duplicate(s, existing, history_df, _seen_tickers_this_run):
            continue
        new_sigs.append(s)
        _seen_tickers_this_run.add(ticker)  # первый сигнал → тикер занят
    hourly_count = len(dfh) if dfh is not None else 0
    print(f"{hourly_count} баров hourly | сигналов: {len(sigs)}" +
          (f" ({len(new_sigs)} новых)" if len(sigs) != len(new_sigs) else ""))
    all_new_signals.extend(new_sigs)

# ── Фьючерсы MOEX FORTS ─────────────────────────────────────
_FUTURES_LIST = list(MOEX_FUTURES) + list(MOEX_FUTURES_PERPETUAL)
if _FUTURES_LIST:
    print("\n[MOEX FORTS]")
    for ticker in _FUTURES_LIST:
        print(f"  {ticker}...", end=" ", flush=True)
        df1d, dfh = fetch_futures(ticker)
        if df1d is None or dfh is None:
            print("нет данных")
            continue
        sigs = scan_ticker(ticker, df1d, dfh)
        new_sigs = []
        for s in sigs:
            if is_duplicate(s, existing, history_df, _seen_tickers_this_run):
                continue
            new_sigs.append(s)
            _seen_tickers_this_run.add(ticker)
        hourly_count = len(dfh) if dfh is not None else 0
        print(f"{hourly_count} баров hourly | сигналов: {len(sigs)}" +
              (f" ({len(new_sigs)} новых)" if len(sigs) != len(new_sigs) else ""))
        # Для фьючерсов меняем exchange
        for s in new_sigs:
            s["exchange"] = "FORTS"
        all_new_signals.extend(new_sigs)

# ── Шаг 3: Вывод и отправка новых сигналов ──
print("\n" + "=" * 60)

if not all_new_signals:
    print("  Новых сигналов нет.")
else:
    print(f"  НАЙДЕНО СИГНАЛОВ: {len(all_new_signals)}")
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

        # Отправляем Template A через ntfy
        if _NTFY_AVAILABLE:
            atr_val = sig.get("atr_daily", 0)
            ticker = sig["ticker"]
            risk = sig["risk"]
            entry = sig["entry"]
            # Cost ratio (упрощённо: 0.06% комиссии + проскальзывание)
            from config_trading import COMMISSION_PCT, SLIPPAGE_STEPS
            ts = moex_tick_size(entry)
            cost_ratio = (entry * COMMISSION_PCT * 2 + 2 * ts * SLIPPAGE_STEPS) / risk if risk > 0 else 0
            hist_pf = _HISTORICAL_PF.get(ticker, 0.5)
            lots = max(1, int(100.0 / risk)) if risk > 0 else 1
            _ntfy_open(sig, atr_val, cost_ratio, hist_pf, lots)

    # ── исполнение через брокера Финам ──
    _live_from_env = os.getenv("LIVE_TRADING", "0").strip() == "1"
    if _BROKER_AVAILABLE and _LIVE_TRADING:
        if not _live_from_env:
            print("\n  [CRITICAL SAFETY] LIVE_TRADING=True в broker_finam, "
                  "но os.getenv('LIVE_TRADING') != '1'. ОРДЕРА ЗАБЛОКИРОВАНЫ!")
            _LIVE_TRADING = False
            _BROKER_AVAILABLE = False
        print(f"\n  Отправка ордеров в Финам...")
        ok_count = 0
        for sig in all_new_signals:
            success = _finam_execute(sig)
            status  = "OK" if success else "ОШИБКА"
            print(f"    {sig['ticker']} {sig['direction']} → {status}")
            if success: ok_count += 1
        print(f"  Исполнено: {ok_count}/{len(all_new_signals)}")
    elif _LIVE_TRADING and not _BROKER_AVAILABLE:
        print("\n  [WARN] LIVE_TRADING=1, но broker_finam.py не найден рядом со скриптом.")

    # Сохраняем новые сигналы
    new_df = pd.DataFrame(all_new_signals)
    if len(existing) > 0:
        result = pd.concat([existing, new_df], ignore_index=True)
    else:
        result = new_df
    result.to_csv(SIG_FILE, index=False)
    print(f"\n  Сохранено в: {SIG_FILE}")
    print(f"  Всего записей в файле: {len(result)}")

# ── Шаг 4: Отправляем радар портфеля (Template C) ──
print("\n[RADAR] Формирование радара портфеля...")
active_positions = positions if len(positions) > 0 else (load_positions() if len(positions) == 0 else positions)
if len(active_positions) == 0 and len(all_new_signals) > 0:
    # Если только что добавили сигналы, перезагружаем
    active_positions = load_positions()
elif len(active_positions) == 0 and len(closed_positions) == 0 and len(all_new_signals) == 0:
    active_positions = load_positions()

radar_data = []
if len(active_positions) > 0:
    # ── Инициализируем клиент Финам для радара ──
    _finam_client_radar = None
    if _BROKER_AVAILABLE and FINAM_TOKEN:
        try:
            from finam_trade_api import Client, TokenManager
            _finam_client_radar = Client(TokenManager(FINAM_TOKEN))
        except Exception:
            _finam_client_radar = None

    for idx, row in active_positions.iterrows():
        ticker = row["ticker"]
        current = fetch_current_price(ticker, client=_finam_client_radar)
        if current is None:
            continue
        time.sleep(0.15)
        entry = float(row["entry"])
        stop = float(row["stop"])
        target = float(row["target"])
        direction = row["direction"]
        is_long = direction == "LONG"

        risk = abs(entry - stop)
        if risk == 0:
            continue
        total_range = abs(target - stop)
        if total_range == 0:
            continue

        # % пути от entry до стопа/тейка (100% = у входа, 0% = у уровня)
        # Используем полный диапазон (stop-target) как знаменатель
        entry_to_stop   = abs(entry - stop)
        entry_to_target = abs(entry - target)
        if is_long:
            # long: stop < entry < target
            pct_to_sl = max(0.0, min(100.0,
                (current - stop) / (entry - stop) * 100)) if entry_to_stop > 0 else 100.0
            pct_to_tp = max(0.0, min(100.0,
                (target - current) / (target - entry) * 100)) if entry_to_target > 0 else 100.0
        else:
            # short: target < entry < stop
            pct_to_sl = max(0.0, min(100.0,
                (stop - current) / (stop - entry) * 100)) if entry_to_stop > 0 else 100.0
            pct_to_tp = max(0.0, min(100.0,
                (current - target) / (entry - target) * 100)) if entry_to_target > 0 else 100.0

        comment = _ntfy_comment(pct_to_tp, pct_to_sl, direction) if _NTFY_AVAILABLE else ""

        radar_data.append({
            "ticker": ticker,
            "direction": direction,
            "entry": entry,
            "current_price": round(current, 4),
            "target": target,
            "stop": stop,
            "pct_to_tp": round(pct_to_tp, 1),
            "pct_to_sl": round(pct_to_sl, 1),
            "status_comment": comment,
        })

if _NTFY_AVAILABLE:
    _ntfy_radar(radar_data)

# Вывод радара в консоль
if radar_data:
    print(f"\n{'='*60}")
    print(f"  РАДАР ПОРТФЕЛЯ: {len(radar_data)} активных позиций")
    print(f"{'='*60}")
    for pos in radar_data:
        d_emoji = "^" if pos["direction"] == "LONG" else "v"
        print(f"  {d_emoji} {pos['ticker']} ({pos['direction']})")
        print(f"      Вход: {pos['entry']} | Текущая: {pos['current_price']}")
        print(f"      Dist to TP: {pos['pct_to_tp']:+.1f}% | Dist to SL: {pos['pct_to_sl']:+.1f}%")
        if pos['status_comment']:
            print(f"      {pos['status_comment']}")
else:
    print("  Активных позиций нет.")

print("\n" + "=" * 60)
print(f"  Сканирование завершено: {now_str()}")
print(f"  Следующий запуск: через 1 час")
print("=" * 60)