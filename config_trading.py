"""
config_trading.py — единый блок конфигурации торговой стратегии.

Паттерн P3 (Ретест после пробоя) на H1, Approval Mode.
Бэктест 20.07.2026: N=18, Exp=+0.980R, PF=3.03, TIME=11%, 5/7 мес+.

Импорт:
    from config_trading import (
        ATR_EXHAUSTION, LEVEL_DIST, VOID_R_MULTIPLIER,
        STOP_ATR_FRAC, RR_TARGET, ...
    )
"""

# ── Параметры P3 Retest (20.07.2026) ────────────────────
ATR_EXHAUSTION  = 0.50    # день выработан на 50%+ daily ATR
LEVEL_DIST      = 0.012   # макс. расстояние от цены до уровня (1.2%)
VOID_R_MULTIPLIER = 3.0   # пустота >= 3R до след. уровня (B5)
STOP_ATR_FRAC   = 0.30    # мин. стоп = 0.30 × daily ATR

# ── Параметры индикаторов ────────────────────────────────
ATR_PERIOD      = 14
TREND_EMA       = 20      # период EMA для трендового фильтра (D1 и H1)
LOOKBACK_PIVOT  = 10      # радиус поиска разворотных точек (дн. баров)
RR_TARGET       = 3.0     # цель 3R
LEVEL_COOLDOWN_BARS = 3   # кулдаун между сделками с одного уровня

# ── Фильтры ──────────────────────────────────────────────
TREND_FILTER_D1    = True   # E1: D1 тренд совпадает (EMA20 дневная)
TREND_FILTER_H1    = True   # E2: H1 тренд совпадает (EMA20 часовая)
TREND_FILTER_SHORTS = True  # шорты только при дневном тренде вниз
MIN_VOL_RATIO      = 1.0   # мин. объём бара / среднего (ослаблен)

# ── RSI фильтр ───────────────────────────────────────────
RSI_LONG_MIN   = 40.0
RSI_SHORT_MAX  = 60.0

# ── Управление капиталом ─────────────────────────────────
RISK_PCT    = 0.01
CAPITAL     = 10_000

# ── Модель ML (отключена) ────────────────────────────────
MODEL_THRESHOLD = 0.0
MODEL_PROB_MAX  = 1.0

# ── Фильтр по времени ───────────────────────────────────
TRADE_HOURS_BLOCK = [13, 16, 17]
TRADE_DAYS_BLOCK  = [3, 4]

# ── Комиссия и проскальзывание ───────────────────────────
COMMISSION_PCT  = 0.0003
SLIPPAGE_STEPS  = 1

# ── CostFilter ───────────────────────────────────────────
MAX_COST_RATIO    = 0.10
COST_RATIO_ACTION = "skip"
COST_RR_OVERRIDE  = 5.0

# ── Тикеры ───────────────────────────────────────────────
MOEX_TICKERS = [
    "GAZP", "SBER", "LKOH", "GMKN", "TATN",
    "NVTK", "VTBR", "MGNT", "NLMK", "CHMF",
]

# ── Фьючерсы MOEX FORTS ─────────────────────────────────
MOEX_FUTURES = [
    "MXU6",
    "GDU6",
]
MOEX_FUTURES_PERPETUAL = []

# ── Параметры удержания позиции ──────────────────────────
SWING_LOOKBACK   = 4
STOP_OFFSET_FRAC = 0.05
MAX_TRADE_BARS   = 35
