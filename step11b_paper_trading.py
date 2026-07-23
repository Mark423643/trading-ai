"""
Step 11b — Paper Trading: P3+P2 combined на H1, Approval Mode.
P3 (ретест): пробой уровня → откат к уровню → отбой → вход.
P2 (закрепление): пробой уровня → P2_CONF_BARS баров подряд ЗА уровнем
    (без отката) → вход на баре подтверждения.
Фильтры (оба паттерна): E1 (D1 trend), E2 (H1 trend), B5 (void>=3R),
F1 (CostFilter), G4 (no squeeze). Breakeven at 2xSL. G6 НЕ применяется.
Бэктест 21.07.2026 (24 тикера): N=735, Exp=+0.316R net, PF=2.27, 7.11/нед.

Запуск: cron каждый час (10-19 МСК).
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

_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
if os.path.exists(_env_path):
    load_dotenv(_env_path, override=True)
else:
    load_dotenv(override=True)

try:
    from broker_finam import execute_signal as _finam_execute, LIVE_TRADING as _LIVE_TRADING, FINAM_TOKEN, _symbol
    _BROKER_AVAILABLE = True
except Exception:
    _BROKER_AVAILABLE = False
    _LIVE_TRADING = False
    FINAM_TOKEN = ""
    _symbol = None

try:
    from notify_ntfy import (
        send_open_signal as _ntfy_open,
        send_close_signal as _ntfy_close,
        send_portfolio_radar as _ntfy_radar,
        generate_status_comment as _ntfy_comment,
        send_signal_for_approval as _ntfy_approval,
    )
    _NTFY_AVAILABLE = True
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_vision"))
        from screenshot_chart import make_screenshot as _make_screenshot
        _VISION_SCREENSHOT_AVAILABLE = True
    except Exception as _e:
        _make_screenshot = None
        _VISION_SCREENSHOT_AVAILABLE = False
        print(f"  [WARN] screenshot_chart unavailable: {_e}")
except Exception as e:
    print(f"  [WARN] notify_ntfy not loaded: {e}")
    _NTFY_AVAILABLE = False

LOG_DIR  = "logs"
LOG_FILE = os.path.join(LOG_DIR, "trading.log")
os.makedirs(LOG_DIR, exist_ok=True)

class _Tee:
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

DATA_DIR  = "data"
MODEL_DIR = "models"
SIG_FILE  = os.path.join(DATA_DIR, "signals_live.csv")

APPROVAL_LOG = os.path.join(DATA_DIR, "approval_log.csv")

def _append_approval_log(sig, decision):
    row = {
        "scan_time":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ticker":         sig.get("ticker", ""),
        "direction":      sig.get("direction", ""),
        "entry":          sig.get("entry", ""),
        "stop":           sig.get("stop", ""),
        "target":         sig.get("target", ""),
        "risk":           sig.get("risk", ""),
        "rr":             sig.get("rr", ""),
        "level":          sig.get("level", ""),
        "atr_daily":      sig.get("atr_daily", ""),
        "trend":          sig.get("trend", ""),
        "pattern":        sig.get("pattern", "P3_RETEST"),
        "human_decision": decision,
        "result":         "",
    }
    write_header = not os.path.exists(APPROVAL_LOG)
    pd.DataFrame([row]).to_csv(APPROVAL_LOG, mode="a", index=False,
                               header=write_header, encoding="utf-8")

HIST_FILE = os.path.join(DATA_DIR, "trades_history.csv")
os.makedirs(DATA_DIR, exist_ok=True)

from config_trading import (
    ATR_EXHAUSTION, LEVEL_DIST, VOID_R_MULTIPLIER,
    STOP_ATR_FRAC,
    ATR_PERIOD, TREND_EMA,
    LOOKBACK_PIVOT, RR_TARGET, LEVEL_COOLDOWN_BARS,
    MODEL_THRESHOLD, MODEL_PROB_MAX,
    TREND_FILTER_D1, TREND_FILTER_H1,
    TREND_FILTER_SHORTS, MIN_VOL_RATIO,
    RSI_LONG_MIN, RSI_SHORT_MAX,
    TRADE_HOURS_BLOCK, TRADE_DAYS_BLOCK,
    MOEX_TICKERS,
    MOEX_FUTURES,
    MOEX_FUTURES_PERPETUAL,
    COMMISSION_PCT, SLIPPAGE_STEPS,
    MAX_COST_RATIO, COST_RATIO_ACTION, COST_RR_OVERRIDE,
    BREAKEVEN_R, CAPITAL,
)

# ── Точный расчёт лотов для уведомлений (та же формула, что при исполнении) ──
try:
    from broker_finam import FinamBroker as _FinamBroker
    _lots_broker = _FinamBroker()
except Exception:
    _lots_broker = None

def _notify_lots(ticker, risk, entry):
    """Число лотов той же calc_lots(), что и при реальном исполнении
    (RISK_PCT от депозита + лимит MAX_POSITION_VALUE_RUB). Живой баланс
    известен только в момент сделки, поэтому здесь free_cash ≈ CAPITAL."""
    if _lots_broker is not None and risk > 0:
        try:
            n = _lots_broker.calc_lots(CAPITAL, risk, ticker, entry_price=entry)
            if n > 0:
                return n
        except Exception:
            pass
    return max(1, int(100.0 / risk)) if risk > 0 else 1

CHECK_LAST_N_BARS   = 720
FRESH_MAX_AGE_HOURS = 2.0
P2_CONF_BARS        = 3   # P2: столько H1-баров подряд цена должна закрыться
                           # ЗА уровнем ("закрепление"), без отката к нему

_ML_AVAILABLE = False

# ───────────────────────────────────────────────────────
# Utilities
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

def moex_tick_size(price: float) -> float:
    if price < 10: return 0.01
    elif price < 25: return 0.02
    elif price < 100: return 0.05
    elif price < 500: return 0.10
    elif price < 1000: return 0.50
    else: return 1.00


# ───────────────────────────────────────────────────────
# Data loading
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
            if math.isnan(price) or math.isinf(price) or price <= 0:
                return None
            ts = quote.quote.timestamp
            if ts is not None:
                now_aware = datetime.now(ts.tzinfo) if ts.tzinfo else datetime.now()
                if (now_aware - ts).total_seconds() > 300:
                    return None
            return price
        except Exception:
            return None

    import asyncio
    return asyncio.run(_get())


FUTURES_URL = ("https://iss.moex.com/iss/engines/futures/markets/"
               "forts/boards/RFUD/securities/{ticker}/candles.json")

def fetch_futures(ticker):
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
        df = df.rename(columns={"begin":"Datetime","open":"Open","high":"High",
                                "low":"Low","close":"Close","volume":"Volume"})
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        return df.set_index("Datetime")[["Open","High","Low","Close","Volume"]].astype(float).dropna()

    df1d = get_candles(24, date_from_daily)
    dfh  = get_candles(60, date_from_hourly)
    return df1d, dfh


# ───────────────────────────────────────────────────────
# Position tracking
# ───────────────────────────────────────────────────────
_HISTORICAL_PF = {
    "SBER": 0.497, "GAZP": 0.660, "LKOH": 0.813, "GMKN": 0.739,
    "TATN": 1.496, "MOEX": 1.192, "MAGN": 1.507, "NVTK": 1.198,
    "ALRS": 1.211, "PHOR": 0.659, "AFKS": 2.938, "CBOM": 1.684,
    "TRNFP": 1.955,
    "VTBR": 2.440, "MGNT": 2.610, "NLMK": 2.000,
    "AFLT": 2.310, "CHMF": 1.280, "FEES": 19.96,
    "HYDR": 2.170, "IRAO": 2.300, "PIKK": 1.300,
    "RUAL": 1.590, "SNGS": 1.930,
    "MXU6": 1.500, "GDU6": 1.500,
}


def load_positions() -> pd.DataFrame:
    if os.path.exists(SIG_FILE):
        try:
            df = pd.read_csv(SIG_FILE)
            if len(df) > 0:
                df = df.sort_values("scan_time").drop_duplicates(
                    subset=["ticker", "level"], keep="last"
                ).reset_index(drop=True)
            return df
        except Exception:
            pass
    return pd.DataFrame()


def load_history() -> pd.DataFrame:
    if os.path.exists(HIST_FILE):
        try:
            return pd.read_csv(HIST_FILE)
        except Exception:
            pass
    return pd.DataFrame()


def check_positions(positions: pd.DataFrame) -> list[dict]:
    closed = []
    if len(positions) == 0:
        return closed

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
                    print(f"  [SPIKE] {t}: {_p1:.2f}->{_p2:.2f} (avg={prices[t]:.2f})")
            else:
                prices[t] = _p1
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

        # Breakeven: if price reached entry + BREAKEVEN_R * risk, move stop to entry
        be_level = entry + BREAKEVEN_R * risk if is_long else entry - BREAKEVEN_R * risk
        effective_stop = stop
        if is_long and current >= be_level:
            effective_stop = max(stop, entry)
        elif not is_long and current <= be_level:
            effective_stop = min(stop, entry)

        if is_long:
            sl_hit = current <= effective_stop
            tp_hit = current >= target
        else:
            sl_hit = current >= effective_stop
            tp_hit = current <= target

        if tp_hit:
            status = "TAKE_PROFIT"
            exit_price = target
            pnl_r = (target - entry) / risk if is_long else (entry - target) / risk
        elif sl_hit:
            if effective_stop == entry:
                status = "BREAKEVEN"
                exit_price = entry
                pnl_r = 0.0
            else:
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
    if not closed:
        return positions

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
        print(f"  [HISTORY] {len(new_history)} trades saved to {HIST_FILE}")

    closed_indices = [c["row_idx"] for c in closed]
    active = positions.drop(index=closed_indices, errors="ignore").reset_index(drop=True)
    active.to_csv(SIG_FILE, index=False)
    print(f"  [ACTIVE] {len(active)} positions remain in {SIG_FILE}")
    return active


# ───────────────────────────────────────────────────────
# P3 Retest detection
# ───────────────────────────────────────────────────────
def detect_retest(dfh, i, level_price, is_long):
    """
    P3 Retest: breakout through level -> consolidation -> return -> bounce.
    LONG:  resistance broken UP -> price returns to level -> bullish bounce
    SHORT: support broken DOWN  -> price returns to level -> bearish bounce
    """
    if i < 22:
        return False

    # 1. Find breakout: at least 1 close beyond level in bars [i-20 .. i-3]
    breakout_found = False
    for j in range(max(0, i - 20), i - 2):
        c = dfh["Close"].iloc[j]
        if is_long:
            if c > level_price * 1.002:
                breakout_found = True
                break
        else:
            if c < level_price * 0.998:
                breakout_found = True
                break

    if not breakout_found:
        return False

    # 2. Current bar close is near level (returned to it)
    cur = dfh.iloc[i]
    dist = abs(cur["Close"] - level_price) / (level_price + 1e-9)
    if dist > LEVEL_DIST:
        return False

    # 3. Current bar bounced in breakout direction
    if is_long:
        if cur["Close"] <= cur["Open"]:
            return False
        if cur["Close"] < level_price:
            return False
    else:
        if cur["Close"] >= cur["Open"]:
            return False
        if cur["Close"] > level_price:
            return False

    return True


# ───────────────────────────────────────────────────────
# Gerchik filter G4 (no squeeze)
# ───────────────────────────────────────────────────────
def is_squeeze(dfh, i):
    if i < 3:
        return False
    return (dfh["Low"].iloc[i-2] > dfh["Low"].iloc[i-3] and
            dfh["Low"].iloc[i-1] > dfh["Low"].iloc[i-2] and
            dfh["Low"].iloc[i] > dfh["Low"].iloc[i-1])


# ───────────────────────────────────────────────────────
# P2: "Пробой с закреплением" — цена закрывается ЗА уровнем
# P2_CONF_BARS раз подряд (без отката к уровню, в отличие от P3-ретеста).
# Вход на баре подтверждения. Один сигнал на уровень (перевзвод только
# на новом пивоте), проверяется только для последнего бара сканирования
# (freshness gate) — исторические подтверждения не переигрываются.
# ───────────────────────────────────────────────────────
def detect_p2_signals(ticker, dfh, lev_prices, lev_types, pivot_dates,
                       _live_price=None):
    signals = []
    H = dfh["High"].values; L = dfh["Low"].values; C = dfh["Close"].values
    AD = dfh["ATR_daily"].values; DT = dfh["daily_trend"].values
    HT = dfh["h_trend"].values; DR = dfh["day_range"].values
    NB = len(dfh)
    last_i = NB - 1

    for level_price, level_type_str, piv_date in zip(lev_prices, lev_types, pivot_dates):
        is_long = (level_type_str == "resistance")
        start_i = dfh.index.searchsorted(piv_date, side="right")
        if start_i >= NB:
            continue

        # Кулдаун P2 отдельный от P3 (см. p2_level_used в module scope)
        p2_key = (ticker, level_price)
        if p2_key in p2_level_used:
            continue

        consec = 0
        for i in range(start_i, NB):
            beyond = (C[i] > level_price) if is_long else (C[i] < level_price)
            if not beyond:
                consec = 0
                continue
            consec += 1
            if consec != P2_CONF_BARS:
                continue

            # Подтверждение "закрепления" на баре i — проверяем фильтры
            ad = AD[i]; dr = DR[i]
            if np.isnan(ad) or ad == 0:
                consec = 0; continue
            if np.isnan(dr) or dr < ATR_EXHAUSTION * ad:
                consec = 0; continue
            d_tr = DT[i]; h_tr = HT[i]
            if TREND_FILTER_D1 and not np.isnan(d_tr):
                if is_long and d_tr < 0: consec = 0; continue
                if not is_long and d_tr > 0: consec = 0; continue
            if TREND_FILTER_H1 and not np.isnan(h_tr):
                if is_long and h_tr < 0: consec = 0; continue
                if not is_long and h_tr > 0: consec = 0; continue
            # G4: без сжатия
            if is_squeeze(dfh, i):
                consec = 0; continue

            is_last_bar = (i == last_i)
            if not is_last_bar:
                # Подтверждение уже случилось в истории — уровень исчерпан
                p2_level_used.add(p2_key)
                break

            if _live_price is not None and _live_price > 0:
                entry = _live_price
            else:
                entry = C[i]

            lo_win = L[max(0, i - P2_CONF_BARS + 1):i + 1]
            hi_win = H[max(0, i - P2_CONF_BARS + 1):i + 1]
            if is_long:
                stop = min(min(lo_win), level_price) - 0.001 * level_price
                stop = min(stop, entry - STOP_ATR_FRAC * ad)
            else:
                stop = max(max(hi_win), level_price) + 0.001 * level_price
                stop = max(stop, entry + STOP_ATR_FRAC * ad)
            risk = abs(entry - stop)
            if risk == 0:
                break
            target = entry + RR_TARGET * risk if is_long else entry - RR_TARGET * risk

            # B5: пустота >= 3R до следующего уровня
            if is_long:
                cands = lev_prices[lev_prices > level_price + risk]
            else:
                cands = lev_prices[lev_prices < level_price - risk]
            if len(cands) > 0:
                nearest = cands[np.argmin(np.abs(cands - C[i]))]
                void_dist = abs(nearest - level_price)
            else:
                void_dist = 999 * ad
            if void_dist < VOID_R_MULTIPLIER * risk:
                break

            # F1: CostFilter
            _tick = moex_tick_size(entry)
            _cost_ratio = (entry * COMMISSION_PCT * 2 + 2 * _tick * SLIPPAGE_STEPS) / risk if risk > 0 else 1.0
            if _cost_ratio > MAX_COST_RATIO:
                if COST_RATIO_ACTION == "increase_rr":
                    target = entry + COST_RR_OVERRIDE * risk if is_long else entry - COST_RR_OVERRIDE * risk
                else:
                    break

            _bar_ts = pd.to_datetime(dfh.index[i])
            if _bar_ts.tzinfo is not None:
                _bar_ts = _bar_ts.tz_localize(None)
            _age_h = (pd.Timestamp.now() - _bar_ts).total_seconds() / 3600.0
            if _age_h > FRESH_MAX_AGE_HOURS:
                break

            trend_label = "BULL" if d_tr == 1 else "BEAR"
            p2_level_used.add(p2_key)

            signals.append({
                "scan_time":    now_str(),
                "ticker":       ticker,
                "exchange":     "MOEX",
                "bar_time":     str(dfh.index[i]),
                "direction":    "LONG" if is_long else "SHORT",
                "pattern":      "P2_BREAKOUT",
                "level":        round(level_price, 4),
                "entry":        round(entry, 4),
                "stop":         round(stop, 4),
                "target":       round(target, 4),
                "risk":         round(risk, 4),
                "rr":           round(abs(target - entry) / risk, 2),
                "atr_daily":    round(ad, 4),
                "trend":        trend_label,
                "h_trend":      "BULL" if h_tr == 1 else "BEAR",
                "vol_decline":  "N",
                "model_prob":   1.0,
            })
            break
    return signals


# ───────────────────────────────────────────────────────
# Scanner core — one ticker
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
    dfh["h_trend"]     = np.where(dfh["Close"] > dfh["EMA20_h"], 1, -1)

    dfh["day_hi"] = dfh.groupby("date_key")["High"].cummax()
    dfh["day_lo"] = dfh.groupby("date_key")["Low"].cummin()
    dfh["day_range"] = dfh["day_hi"] - dfh["day_lo"]

    pivots = find_pivots(df1d, LOOKBACK_PIVOT)
    if len(pivots) == 0:
        return []

    lev_prices = pivots["level"].values
    lev_types  = pivots["type"].values

    signals = []
    scan_from = max(30, len(dfh) - CHECK_LAST_N_BARS)

    _live_price = None
    if _BROKER_AVAILABLE and FINAM_TOKEN:
        _live_price = fetch_current_price(ticker)

    for i in range(scan_from, len(dfh)):
        bar     = dfh.iloc[i]
        atr_h   = bar["ATR"]
        atr_day = bar["ATR_daily"]
        d_tr    = bar["daily_trend"]
        h_tr    = bar["h_trend"]

        if np.isnan(atr_h) or np.isnan(atr_day) or atr_day == 0:
            continue

        # ATR exhaustion: day range >= X% of daily ATR
        day_range = bar["day_range"]
        if np.isnan(day_range) or day_range < ATR_EXHAUSTION * atr_day:
            continue

        # Find nearest level
        dists = np.abs(lev_prices - bar["Close"]) / (bar["Close"] + 1e-9)
        mi    = np.argmin(dists)
        if dists[mi] >= LEVEL_DIST:
            continue

        level_price    = lev_prices[mi]
        level_type_str = lev_types[mi]

        # P3: direction is INVERTED from LP
        # resistance broken UP -> now support -> LONG
        # support broken DOWN -> now resistance -> SHORT
        is_long = (level_type_str == "resistance")

        # Cooldown
        if i - level_last_bar.get(level_price, -LEVEL_COOLDOWN_BARS - 1) < LEVEL_COOLDOWN_BARS:
            continue

        # P3 retest detection
        if not detect_retest(dfh, i, level_price, is_long):
            continue

        # Filter E1: D1 trend must agree
        if TREND_FILTER_D1 and not np.isnan(d_tr):
            if is_long and d_tr < 0:
                continue
            if not is_long and d_tr > 0:
                continue

        # Filter E2: H1 trend must agree
        if TREND_FILTER_H1 and not np.isnan(h_tr):
            if is_long and h_tr < 0:
                continue
            if not is_long and h_tr > 0:
                continue

        # G4: no squeeze (3 consecutive higher lows = breakout imminent)
        if is_squeeze(dfh, i):
            continue

        # Entry price
        is_last_bar = (i == len(dfh) - 1)
        if is_last_bar and _live_price is not None and _live_price > 0:
            entry = _live_price
            stop_invalid = (is_long and entry < level_price * 0.99) or \
                           (not is_long and entry > level_price * 1.01)
            if stop_invalid:
                print(f"      [SKIP] {ticker}: live price {entry:.4f} too far from level "
                      f"{level_price:.4f}")
                continue
            print(f"      [LIVE] {ticker}: entry={bar['Close']:.4f}(close) -> {entry:.4f}(live)")
        else:
            entry = bar["Close"]

        # Stop: max(beyond level, 0.30 x daily ATR)
        if is_long:
            stop_raw = min(bar["Low"], level_price) - 0.001 * level_price
            stop = min(stop_raw, entry - STOP_ATR_FRAC * atr_day)
        else:
            stop_raw = max(bar["High"], level_price) + 0.001 * level_price
            stop = max(stop_raw, entry + STOP_ATR_FRAC * atr_day)

        risk   = abs(entry - stop)
        if risk == 0:
            continue
        target = entry + RR_TARGET * risk if is_long else entry - RR_TARGET * risk

        # Filter B5: void >= 3R to next level
        if is_long:
            cands = lev_prices[lev_prices > level_price + risk]
        else:
            cands = lev_prices[lev_prices < level_price - risk]
        if len(cands) > 0:
            nearest = cands[np.argmin(np.abs(cands - bar["Close"]))]
            void_dist = abs(nearest - level_price)
        else:
            void_dist = 999 * atr_day
        if void_dist < VOID_R_MULTIPLIER * risk:
            continue

        # Filter F1: CostFilter
        _tick = moex_tick_size(entry)
        _cost_ratio = (entry * COMMISSION_PCT * 2 + 2 * _tick * SLIPPAGE_STEPS) / risk if risk > 0 else 1.0
        if _cost_ratio > MAX_COST_RATIO:
            if COST_RATIO_ACTION == "increase_rr":
                target = entry + COST_RR_OVERRIDE * risk if is_long else entry - COST_RR_OVERRIDE * risk
            else:
                continue

        # Volume decline info (D2 — informational, not hard filter)
        vol_decline = False
        if i >= 20:
            vol5 = dfh["Volume"].iloc[i-5:i].mean()
            vol15 = dfh["Volume"].iloc[i-20:i-5].mean()
            if vol15 > 0 and vol5 / vol15 < 1.0:
                vol_decline = True

        level_last_bar[level_price] = i

        # Freshness gate
        _bar_ts = pd.to_datetime(dfh.index[i])
        if _bar_ts.tzinfo is not None:
            _bar_ts = _bar_ts.tz_localize(None)
        _age_h = (pd.Timestamp.now() - _bar_ts).total_seconds() / 3600.0
        if not (is_last_bar and _age_h <= FRESH_MAX_AGE_HOURS):
            continue

        trend_label = "BULL" if d_tr == 1 else "BEAR"

        signals.append({
            "scan_time":    now_str(),
            "ticker":       ticker,
            "exchange":     "MOEX",
            "bar_time":     str(dfh.index[i]),
            "direction":    "LONG" if is_long else "SHORT",
            "pattern":      "P3_RETEST",
            "level":        round(level_price, 4),
            "entry":        round(entry, 4),
            "stop":         round(stop, 4),
            "target":       round(target, 4),
            "risk":         round(risk, 4),
            "rr":           round(abs(target - entry) / risk, 2),
            "atr_daily":    round(atr_day, 4),
            "trend":        trend_label,
            "h_trend":      "BULL" if h_tr == 1 else "BEAR",
            "vol_decline":  "Y" if vol_decline else "N",
            "model_prob":   1.0,
        })

    # P2: "Пробой с закреплением" — независимая детекция на тех же уровнях
    signals.extend(detect_p2_signals(
        ticker, dfh, lev_prices, lev_types, pivots["date"].values, _live_price,
    ))

    return signals


# ───────────────────────────────────────────────────────
# Dedup
# ───────────────────────────────────────────────────────
def load_existing_signals():
    if os.path.exists(SIG_FILE):
        try:
            return pd.read_csv(SIG_FILE)
        except Exception:
            pass
    return pd.DataFrame()

def is_duplicate(sig, existing, history=None, seen_tickers_in_run: set = None):
    if seen_tickers_in_run is not None and sig["ticker"] in seen_tickers_in_run:
        return True

    if len(existing) > 0 and "bar_time" in existing.columns:
        same_bar = existing[
            (existing["ticker"] == sig["ticker"]) &
            (existing["level"].round(2) == round(sig["level"], 2)) &
            (existing["bar_time"].astype(str) == str(sig.get("bar_time", "")))
        ]
        if len(same_bar) > 0:
            return True

    if len(existing) > 0:
        same = existing[
            (existing["ticker"] == sig["ticker"]) &
            (existing["level"].round(2) == round(sig["level"], 2))
        ]
        if len(same) > 0:
            last_time = pd.to_datetime(same["scan_time"]).max()
            if (pd.Timestamp.now() - last_time).total_seconds() < 24 * 3600:
                return True

    if history is not None and len(history) > 0:
        same_hist = history[
            (history["ticker"] == sig["ticker"]) &
            (history["level"].round(2) == round(sig["level"], 2))
        ]
        if len(same_hist) > 0:
            last_close = pd.to_datetime(same_hist["close_time"]).max()
            if (pd.Timestamp.now() - last_close).total_seconds() < 86400:
                return True

    return False


# ───────────────────────────────────────────────────────
# MAIN LOOP
# ───────────────────────────────────────────────────────
level_last_bar = {}
p2_level_used  = set()   # P2: (ticker, level) уже использованные для сигнала

_MODE_LABEL = "LIVE TRADING" if (_BROKER_AVAILABLE and _LIVE_TRADING) else "PAPER TRADING"

print("=" * 60)
print(f"  {_MODE_LABEL} SCANNER  |  P3 RETEST H1  |  {now_str()}")
print("=" * 60)
print(f"  MOEX TQBR: {', '.join(MOEX_TICKERS)}")
print(f"  FORTS:     {', '.join(list(MOEX_FUTURES) + list(MOEX_FUTURES_PERPETUAL))}")
print(f"  Pattern: P3 Retest (breakout->retest->bounce)")
print(f"  Params:  STOP={STOP_ATR_FRAC}, RR={RR_TARGET}, VOID>={VOID_R_MULTIPLIER}R")
print(f"  Фильтры: E1(тренд D1), E2(тренд H1), B5(пустота>={VOID_R_MULTIPLIER}R), "
      f"F1(комиссия<={MAX_COST_RATIO}), G4(без сжатия)")
print(f"  Безубыток: при {BREAKEVEN_R}xSL")
print(f"  ATR день: диапазон >= {ATR_EXHAUSTION} * ATR")
print(f"  Уведомления: {'ВКЛ' if _NTFY_AVAILABLE else 'ВЫКЛ'}")
print("=" * 60)

_MSK = timezone(timedelta(hours=3))
_now_msk = datetime.now(_MSK)
_DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

if _now_msk.weekday() in TRADE_DAYS_BLOCK:
    _dn = _DAY_NAMES[_now_msk.weekday()]
    print(f"  [ФИЛЬТР] {_dn} — торговля запрещена. Выход.")
    sys.exit(0)

if _now_msk.hour in TRADE_HOURS_BLOCK:
    print(f"  [ФИЛЬТР] {_now_msk.hour}:00 МСК — торговля запрещена. Выход.")
    sys.exit(0)

print(f"  Время сканирования: {_now_msk.strftime('%H:%M МСК')} ({_DAY_NAMES[_now_msk.weekday()]})")

# Step 1: Check open positions
print("\n[ПОЗИЦИИ] Проверка открытых позиций...")
positions = load_positions()
print(f"  Open positions: {len(positions)}")

closed_positions = check_positions(positions)
for cp in closed_positions:
    print(f"  >> {cp['ticker']} {cp['direction']}: {cp['status']} (PnL={cp['pnl_r']:+.2f}R)")
    if _NTFY_AVAILABLE:
        _ntfy_close(cp, cp["exit_price"], cp["status"], cp["pnl_r"], risk_rub=100.0)

positions = save_closed_and_active(positions, closed_positions)

# Step 2: Scan for new signals
existing = positions if len(positions) > 0 else pd.DataFrame()
history_df = load_history()
all_new_signals = []
_seen_tickers_this_run = set()

print("\n[MOEX TQBR]")
for ticker in MOEX_TICKERS:
    print(f"  {ticker}...", end=" ", flush=True)
    df1d, dfh = fetch_moex(ticker)
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
    print(f"{hourly_count} bars | signals: {len(sigs)}" +
          (f" ({len(new_sigs)} new)" if len(sigs) != len(new_sigs) else ""))
    for _s in new_sigs:
        _s["_dfh"] = dfh
    all_new_signals.extend(new_sigs)

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
        print(f"{hourly_count} bars | signals: {len(sigs)}" +
              (f" ({len(new_sigs)} new)" if len(sigs) != len(new_sigs) else ""))
        for s in new_sigs:
            s["exchange"] = "FORTS"
        all_new_signals.extend(new_sigs)

# Step 3: Output and send signals
print("\n" + "=" * 60)

if not all_new_signals:
    print("  Новых сигналов нет.")
else:
    print(f"  НАЙДЕНО СИГНАЛОВ: {len(all_new_signals)}")
    print("=" * 60)
    for i, sig in enumerate(all_new_signals, 1):
        dir_emoji = "^" if sig["direction"] == "LONG" else "v"
        print(f"\n  [{i}] {sig['ticker']} ({sig['exchange']})  {dir_emoji} {sig['direction']}  [{sig['pattern']}]")
        print(f"      Bar time:  {sig['bar_time']}")
        print(f"      Level:     {sig['level']}")
        print(f"      Entry:     {sig['entry']}")
        print(f"      Stop:      {sig['stop']}  (risk: {sig['risk']:.4f})")
        print(f"      Target:    {sig['target']}  (R:R {sig['rr']}:1)")
        print(f"      Trend D1:  {sig['trend']}  |  Trend H1: {sig['h_trend']}")
        print(f"      Vol decline: {sig['vol_decline']}")

        if _NTFY_AVAILABLE:
            if os.getenv("APPROVAL_MODE", "1") == "1":
                _shot_path = None
                if _VISION_SCREENSHOT_AVAILABLE:
                    try:
                        _shot_path = _make_screenshot(
                            sig["ticker"], sig["bar_time"], sig["level"],
                            sig["entry"], sig["stop"], sig["target"],
                            sig["direction"], timeframe="H1",
                            df=sig.get("_dfh"),
                        )
                        print(f"      Скриншот: {_shot_path}")
                    except Exception as _e:
                        print(f"      Скриншот не создан: {_e}")
                _ntfy_approval(
                    sig,
                    sig.get("atr_daily", 0),
                    screenshot_path=_shot_path,
                )
                if _shot_path and os.path.exists(_shot_path):
                    import shutil
                    _pend = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "ml_vision", "data", "pending")
                    os.makedirs(_pend, exist_ok=True)
                    _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    shutil.copy(_shot_path, os.path.join(
                        _pend, f"{sig['ticker']}_{sig['direction']}_{_ts}.png"))
                _append_approval_log(sig, "PENDING")
                print(f"      Отправлено в NTFY"
                      f"{' (скриншот)' if _shot_path else ' (текст)'}")
            else:
                atr_val = sig.get("atr_daily", 0)
                risk = sig["risk"]
                entry = sig["entry"]
                ts = moex_tick_size(entry)
                cost_ratio = (entry * COMMISSION_PCT * 2 + 2 * ts * SLIPPAGE_STEPS) / risk if risk > 0 else 0
                hist_pf = _HISTORICAL_PF.get(sig["ticker"], 0.5)
                lots = _notify_lots(sig["ticker"], risk, entry)
                _ntfy_open(sig, atr_val, cost_ratio, hist_pf, lots)

    _live_from_env = os.getenv("LIVE_TRADING", "0").strip() == "1"
    if _BROKER_AVAILABLE and _LIVE_TRADING:
        if not _live_from_env:
            print("\n  [CRITICAL SAFETY] LIVE_TRADING=True in broker_finam, "
                  "but os.getenv('LIVE_TRADING') != '1'. ORDERS BLOCKED!")
            _LIVE_TRADING = False
            _BROKER_AVAILABLE = False
        print(f"\n  Sending orders to Finam...")
        ok_count = 0
        for sig in all_new_signals:
            success = _finam_execute(sig)
            status  = "OK" if success else "ERROR"
            print(f"    {sig['ticker']} {sig['direction']} -> {status}")
            if success: ok_count += 1
        print(f"  Executed: {ok_count}/{len(all_new_signals)}")
    elif _LIVE_TRADING and not _BROKER_AVAILABLE:
        print("\n  [WARN] LIVE_TRADING=1, but broker_finam.py not found.")

    new_df = pd.DataFrame(all_new_signals)
    if len(existing) > 0:
        result = pd.concat([existing, new_df], ignore_index=True)
    else:
        result = new_df
    result.to_csv(SIG_FILE, index=False)
    print(f"\n  Saved to: {SIG_FILE}")
    print(f"  Total records: {len(result)}")

# Step 4: Portfolio radar
print("\n[RADAR] Building portfolio radar...")
active_positions = positions if len(positions) > 0 else load_positions()

radar_data = []
if len(active_positions) > 0:
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

        entry_to_stop   = abs(entry - stop)
        entry_to_target = abs(entry - target)
        if is_long:
            pct_to_sl = max(0.0, min(100.0,
                (current - stop) / (entry - stop) * 100)) if entry_to_stop > 0 else 100.0
            pct_to_tp = max(0.0, min(100.0,
                (target - current) / (target - entry) * 100)) if entry_to_target > 0 else 100.0
        else:
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

if radar_data:
    print(f"\n{'='*60}")
    print(f"  PORTFOLIO RADAR: {len(radar_data)} active positions")
    print(f"{'='*60}")
    for pos in radar_data:
        d_emoji = "^" if pos["direction"] == "LONG" else "v"
        print(f"  {d_emoji} {pos['ticker']} ({pos['direction']})")
        print(f"      Entry: {pos['entry']} | Current: {pos['current_price']}")
        print(f"      Dist to TP: {pos['pct_to_tp']:+.1f}% | Dist to SL: {pos['pct_to_sl']:+.1f}%")
        if pos['status_comment']:
            print(f"      {pos['status_comment']}")
else:
    print("  No active positions.")

print("\n" + "=" * 60)
print(f"  Scan complete: {now_str()}")
print(f"  Next run: in 1 hour")
print("=" * 60)
