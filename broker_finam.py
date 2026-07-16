"""
broker_finam.py — интеграция с брокером Финам через finam-trade-api 4.3.2.

Установка:
    pip install finam-trade-api==4.3.2

Переменные окружения:
    FINAM_TOKEN      — токен из личного кабинета Финам (Сервисы → API)
    FINAM_CLIENT_ID  — номер торгового счёта (account_id)
    FINAM_RISK_PCT   — доля депозита на сделку, по умолчанию 0.01 (1%)
    LIVE_TRADING     — 1 чтобы слать реальные ордера, 0 (по умолчанию) — только лог

Формат символа:
    Finam REST API принимает символы в формате "MOEX:VTBR" (биржа:тикер).
    Для NASDAQ: "NASDAQ:NVDA". Уточните формат в документации вашего аккаунта.

Важно:
    Библиотека 4.3.2 — REST-клиент (aiohttp/httpx), не gRPC.
    Нет отдельного модуля stop_order — стоп-лоссы и тейк-профиты выставляются
    как обычные заявки через OrderClient.place_order() с OrderType.STOP / LIMIT.
"""

from __future__ import annotations

import asyncio
import logging
import os

import httpx
from dotenv import load_dotenv

# ── ПРИНУДИТЕЛЬНАЯ загрузка .env ──────────────────────────────
# Используем абсолютный путь к .env рядом с этим скриптом,
# чтобы не зависеть от текущей рабочей директории.
# override=True — .env переопределяет системные переменные
# (критично, если VPS cron экспортирует LIVE_TRADING=1).
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
if os.path.exists(_env_path):
    load_dotenv(_env_path, override=True)
    print(f"[broker_finam] .env загружен: {_env_path}")
else:
    load_dotenv(override=True)
    print(f"[broker_finam] .env не найден по пути {_env_path}")
# ──────────────────────────────────────────────────────────────

# ── импорт finam-trade-api 4.3.2 на уровне модуля ───────────────────────────
# Все имена импортируются здесь один раз. Если что-то не находится —
# сразу виден точный ImportError с именем отсутствующего модуля/класса.
try:
    from finam_trade_api import Client, TokenManager
    from finam_trade_api.base_client.models import FinamDecimal, Side
    from finam_trade_api.order.model import (
        Order,
        OrderType,
        StopCondition,
        TimeInForce,
    )
    _FINAM_AVAILABLE = True
    _FINAM_IMPORT_ERR = ""
except ImportError as _err:
    _FINAM_AVAILABLE = False
    _FINAM_IMPORT_ERR = str(_err)

# ── логгер ордеров ───────────────────────────────────────────────────────────
LOG_DIR   = "logs"
ORDER_LOG = os.path.join(LOG_DIR, "orders.log")
os.makedirs(LOG_DIR, exist_ok=True)

order_logger = logging.getLogger("finam_orders")
order_logger.setLevel(logging.DEBUG)
if not order_logger.handlers:
    _fh = logging.FileHandler(ORDER_LOG, encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                                        datefmt="%Y-%m-%d %H:%M:%S"))
    order_logger.addHandler(_fh)
    _sh = logging.StreamHandler()
    _sh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                                        datefmt="%Y-%m-%d %H:%M:%S"))
    order_logger.addHandler(_sh)

# ── параметры ────────────────────────────────────────────────────────────────
FINAM_TOKEN     = os.getenv("FINAM_TOKEN", "")
FINAM_CLIENT_ID = os.getenv("FINAM_CLIENT_ID", "")
RISK_PCT        = float(os.getenv("FINAM_RISK_PCT", "0.01"))
LIVE_TRADING    = os.getenv("LIVE_TRADING", "0").strip() == "1"

# ── Жёсткий лимит стоимости позиции (защита от маржинальных требований) ──
# Если общая стоимость позиции (entry × shares_total) превышает этот лимит,
# количество лотов будет пропорционально уменьшено.
# Для MOEX TQBR: шорт требует ГО ~12-15% от стоимости позиции.
# При свободных средствах ~10 000 руб. и лимите 5 000 руб.:
#   ГО = 5 000 × 15% = 750 руб. — безопасно для любого тикера.
MAX_POSITION_VALUE_RUB = 5000

# (EXCHANGE_PREFIX больше не используется — символы в формате TICKER@MIC)

# ── лотность инструментов MOEX TQBR — РЕЗЕРВНАЯ таблица ──────────────────────
# ВНИМАНИЕ: это только fallback на случай сбоя live-запроса к Finam
# (FinamBroker._get_asset_info() запрашивает реальный lot_size через
# client.assets.get_asset() перед каждой сделкой — используйте его).
# Таблица ранее содержала неверные значения (RUAL=1, CHGZ=1 вместо
# реального лота 10), что приводило к ошибке API
# "INVALID_ARGUMENT: Invalid quantity value" — исправлено по факту
# проверки через живой API 2026-07-13.
LOT_SIZES = {
    # ── Акции MOEX TQBR ──────────────────────────────────────────
    "SBER": 1,   "GAZP": 10,  "LKOH": 1,   "VTBR": 1,
    "GMKN": 10,  "CHMF": 10,  "NLMK": 10,  "MGNT": 1,
    "AFLT": 10,  "FEES": 1,   "HYDR": 1,   "IRAO": 1,
    "PIKK": 1,   "RUAL": 10,  "SNGS": 1,
    "AKRN": 1,   "BANEP": 1,  "BSPB": 1,   "CHGZ": 10,
    "CNRU": 1,   "DATA": 1,   "GLRX": 1,   "LSRG": 1,
    "MVID": 1,   "PLZL": 1,   "RNFT": 1,   "SELG": 1,
    "SPBE": 1,   "TATNP": 1,  "VKCO": 1,
    # ── Фьючерсы MOEX FORTS (мультипликатор контракта) ───────────
    "MXU6": 1,    # Мини-индекс Мосбиржи
    "GDU6": 1,    # Золото
}


# ── формат символа ────────────────────────────────────────────────────────────
def _symbol(ticker: str, exchange: str) -> str:
    """
    Формирует symbol для Finam REST API v1.
    MOEX:   "SBER@MISX"  (тикер@MIC для акций)
    NASDAQ: "NVDA@XNAS"  (тикер@MIC для NASDAQ)
    FORTS:  "SiU6@RTSX"  (тикер@MIC для фьючерсов)
    """
    mic_map = {
        "MOEX":   "MISX",
        "NASDAQ": "XNAS",
        "FORTS":  "RTSX",
    }
    mic = mic_map.get(exchange, "MISX")
    return f"{ticker}@{mic}"


def _finam_qty(n: int) -> FinamDecimal:
    """Создаёт FinamDecimal из целого числа акций/лотов."""
    return FinamDecimal(value=str(n))


def _finam_price(price: float) -> FinamDecimal:
    """Создаёт FinamDecimal из цены."""
    return FinamDecimal(value=f"{price:.6f}")

def _price_str(price: float) -> str:
    """Форматирует цену как строку без лишних нулей: 919.0→"919", 918.9643→"918.9643"."""
    return f"{round(price, 8):.8f}".rstrip('0').rstrip('.')


def _finam_price_dec(price: float) -> FinamDecimal:
    """
    Обёртка цены в FinamDecimal для полей stop_price / limit_price.
    API ожидает {"value": "919"}, а НЕ строку "919.000000".
    Пример: 919.0 → FinamDecimal(value="919"), 918.9643 → FinamDecimal(value="918.9643").
    """
    return FinamDecimal(value=_price_str(price))


def moex_tick_size(price: float) -> float:
    """Минимальный шаг цены для MOEX TQBR."""
    if price < 10: return 0.01
    elif price < 25: return 0.02
    elif price < 100: return 0.05
    elif price < 500: return 0.10
    elif price < 1000: return 0.50
    else: return 1.00


def round_to_tick_size(price: float, tick_size: float) -> float:
    """Округляет цену к ближайшему допустимому шагу."""
    if tick_size <= 0:
        return price
    return round(price / tick_size) * tick_size


def _cash_rub(account_info) -> float:
    """Извлекает рублёвый остаток из GetAccountResponse.cash (список FinamMoney)."""
    total = 0.0
    for m in account_info.cash:
        if m.currency_code.upper() in ("RUB", "RUR"):
            total += float(m.units) + m.nanos / 1_000_000_000
    # Если нет RUB — берём первую валюту (USD и т.д.)
    if total == 0.0 and account_info.cash:
        m = account_info.cash[0]
        total = float(m.units) + m.nanos / 1_000_000_000
    return total


# ── брокерский клиент ────────────────────────────────────────────────────────
class FinamBroker:
    """Обёртка над finam-trade-api 4.3.2 для автоматического исполнения сигналов."""

    def __init__(self):
        self._client: Client | None = None
        self._asset_cache: dict[str, tuple[float, int]] = {}

    async def connect(self) -> bool:
        if not _FINAM_AVAILABLE:
            order_logger.error(
                f"Ошибка импорта finam-trade-api: {_FINAM_IMPORT_ERR}. "
                "Выполните: pip install finam-trade-api==4.3.2"
            )
            return False
        try:
            self._client = Client(TokenManager(FINAM_TOKEN))
            order_logger.info("Finam REST клиент создан")
            return True
        except Exception as exc:
            order_logger.error(f"Не удалось создать Client: {exc}")
            return False

    # ── баланс ──────────────────────────────────────────────────────────────
    async def get_free_cash(self) -> float:
        try:
            info = await self._client.account.get_account_info(FINAM_CLIENT_ID)
            cash = _cash_rub(info)
            order_logger.info(f"Свободных средств: {cash:.2f}")
            return cash
        except Exception as exc:
            order_logger.error(f"Ошибка получения баланса: {exc}")
            return 0.0

    # ── расчёт числа лотов ───────────────────────────────────────────────────
    @staticmethod
    def _lots_from_ticker(ticker: str) -> int:
        """Возвращает размер лота для тикера (по умолчанию 1) — резервный fallback."""
        return LOT_SIZES.get(ticker.upper(), 1)

    # ── реальные параметры инструмента (шаг цены, лотность) из Finam API ──────
    async def _get_asset_info(self, symbol: str) -> tuple[float | None, int | None]:
        """
        Запрашивает реальный шаг цены (min_step/decimals) и лотность (lot_size)
        инструмента через client.assets.get_asset(). Кэширует по symbol.

        Возвращает (None, None) при ошибке — вызывающий код должен сам
        подставить резервные значения (moex_tick_size()/LOT_SIZES).

        Найдено эмпирически (проверка через живой аккаунт 2026-07-13):
        зашитая таблица moex_tick_size()/LOT_SIZES расходится с реальными
        значениями API в разы (напр. RUAL: реальный шаг 0.005, таблица
        давала 0.05 — в 10 раз грубее), что и было причиной массовых
        400 Bad Request на STOP-LOSS/TAKE-PROFIT и INVALID_ARGUMENT на MARKET.
        """
        if symbol in self._asset_cache:
            return self._asset_cache[symbol]
        try:
            info = await self._client.assets.get_asset(symbol, FINAM_CLIENT_ID)
            tick = int(info.min_step) / (10 ** info.decimals)
            lot = int(float(info.lot_size.value))
            self._asset_cache[symbol] = (tick, lot)
            return tick, lot
        except Exception as exc:
            order_logger.warning(
                f"{symbol}: не удалось получить min_step/lot_size из API "
                f"({type(exc).__name__}: {exc}), используем резервные значения"
            )
            return None, None

    def calc_lots(self, free_cash: float, risk_per_share: float, ticker: str,
                  entry_price: float = 0.0, lot_size: int | None = None) -> int:
        """
        Рассчитывает количество лотов/контрактов для сделки.

        1% депозита / риск на лот = макс. лотов. Строго не превышает 1.5% на контракт.
        Дополнительно ограничивает СТОИМОСТЬ позиции пределом MAX_POSITION_VALUE_RUB
        (если передан entry_price) — иначе стоимость проверяется только позже.

        lot_size: если передан (из живого API), используется вместо резервной
        таблицы LOT_SIZES — предпочтительно вызывать с реальным значением.

        Для акций: lot_size = кол-во акций в лоте, risk_per_share = RUB/акция
        Для фьючерсов: lot_size = мультипликатор (x1000, x100 и т.д.),
                       risk_per_share = риск в ценовых пунктах
        Risk_per_lot (RUB) = risk_per_share * lot_size
        """
        if risk_per_share <= 0:
            return 0
        if lot_size is None or lot_size <= 0:
            lot_size = self._lots_from_ticker(ticker)

        # ── Ограничение №1: по РИСКУ ─────────────────────────────────
        risk_per_lot = risk_per_share * lot_size
        if risk_per_lot <= 0:
            return 0
        risk_budget = free_cash * RISK_PCT
        max_lots_by_risk = int(risk_budget / risk_per_lot)  # строго 1% риск
        # Если 1 контракт не превышает 1.5% — разрешаем 1
        if max_lots_by_risk == 0 and risk_budget / risk_per_lot >= 0.5:
            max_lots_by_risk = 1

        if entry_price <= 0:
            return max(0, max_lots_by_risk)

        # ── Ограничение №2: по СТОИМОСТИ позиции в рублях ────────────
        max_shares_by_value = int(MAX_POSITION_VALUE_RUB / entry_price)
        max_shares_by_value = max(max_shares_by_value, lot_size)
        max_shares_by_value = (max_shares_by_value // lot_size) * lot_size
        max_lots_by_value = max_shares_by_value // lot_size

        max_lots = min(max_lots_by_risk, max_lots_by_value)
        if max_lots_by_risk != max_lots_by_value:
            order_logger.warning(
                f"{ticker}: Два ограничения: риск={max_lots_by_risk} лотов, "
                f"стоимость={max_lots_by_value} лотов → выбрано {max_lots} лотов"
            )
        return max(0, max_lots)

    # ── проверка наличия позиции на аккаунте ─────────────────────────────────
    async def get_position_quantity(self, ticker: str) -> int:
        """
        Запрашивает account_info и проверяет, есть ли открытая позиция
        по данному тикеру. Возвращает количество акций (0 если нет).

        GetAccountResponse.positions — список Position{symbol, quantity, ...}
        (symbol в формате "TICKER@MIC"). Прежняя реализация проверяла
        несуществующие поля info.securities / self._client.portfolio (их нет
        ни в модели ответа, ни в Client — см. finam_trade_api.account.model
        и finam_trade_api.client), из-за чего функция ВСЕГДА возвращала 0,
        даже если позиция реально была на счёте.
        """
        try:
            info = await self._client.account.get_account_info(FINAM_CLIENT_ID)
            for pos in info.positions:
                pos_ticker = pos.symbol.split("@")[0] if "@" in pos.symbol else pos.symbol
                if pos_ticker.upper() == ticker.upper():
                    qty = abs(float(pos.quantity.value))
                    order_logger.info(
                        f"[POSITION-CHECK] {ticker}: найдена позиция {qty:.0f} шт."
                    )
                    return int(qty)
            order_logger.info(f"[POSITION-CHECK] {ticker}: позиция не найдена")
            return 0
        except Exception as exc:
            order_logger.error(f"[POSITION-CHECK] {ticker}: ошибка проверки: {exc}")
            return 0

    # ── отправка ордера через Order model ─────────────────────────────────────
    async def _place_order_model(self, order: Order, label: str,
                                 treat_position_as_success: bool = False) -> str | None:
        """
        Выставляет заявку через OrderClient.place_order() с правильной
        сериализацией через model_dump(mode='json').

        treat_position_as_success:
          True  — только для РЫНОЧНОГО ВХОДА. Если API вернул ошибку, но позиция
                  найдена на счёте, значит вход всё-таки исполнился → 'UNCERTAIN'.
          False — для СТОП-ЛОССА и ТЕЙК-ПРОФИТА. Наличие позиции НЕ означает, что
                  защитный ордер выставлен (позиция — от входа). Ошибка = реальный
                  провал → None (сработает аварийное закрытие). Без этого голая
                  позиция ложно помечалась «защищённой» (баг BANEP 14.07).
        """
        try:
            state = await self._client.orders.place_order(order)
            order_logger.info(
                f"{label} | {order.side.value} {order.quantity.value} лот(ов) "
                f"{order.symbol} | order_id={state.order_id} status={state.status.value}"
            )
            return state.order_id
        except Exception as exc:
            symbol = order.symbol
            detail = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, httpx.HTTPStatusError):
                body = exc.response.text[:300]
                detail += f" | response body: {body}"
            order_logger.error(f"Ошибка {label} {symbol}: {detail}")

            # ── Извлекаем тикер из symbol (формат "TICKER@MIC") ────────────
            ticker = symbol.split("@")[0] if "@" in symbol else symbol

            # ── UNCERTAIN только для входа: позиция = подтверждение входа ──
            if treat_position_as_success:
                try:
                    pos_qty = await self.get_position_quantity(ticker)
                    if pos_qty > 0:
                        order_logger.warning(
                            f"[UNCERTAIN] {label} {symbol}: API вернул ошибку, но "
                            f"позиция {ticker} ({pos_qty} шт.) найдена на аккаунте. "
                            f"Считаем ВХОД исполнившимся."
                        )
                        return "UNCERTAIN"  # сигнал: позиция есть, нужны SL/TP
                except Exception:
                    pass

            order_logger.error(
                f"[FAILED] {label} {symbol}: ордер не исполнился."
            )
            return None

    # ── рыночный ордер на вход ───────────────────────────────────────────────
    async def _place_market_order(self, symbol: str, side: Side,
                                   lots: int, account_id: str) -> str | None:
        order = Order(
            account_id=account_id,
            symbol=symbol,
            quantity=FinamDecimal(value=str(lots)),
            side=side,
            type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,
        )
        return await self._place_order_model(order, "MARKET", treat_position_as_success=True)

    # ── стоп-лосс (стоп-маркет при срабатывании цены) ─────────────────────────
    async def _place_stop_loss(self, symbol: str, close_side: Side,
                                lots: int, account_id: str,
                                stop_price: float, is_long: bool,
                                tick_size: float | None = None) -> str | None:
        if tick_size is None or tick_size <= 0:
            tick_size = moex_tick_size(stop_price)
        stop_price_rounded = round_to_tick_size(stop_price, tick_size)
        if abs(stop_price - stop_price_rounded) > 1e-6:
            order_logger.warning(
                f"STOP-LOSS {symbol}: цена округлена {stop_price:.6f} → "
                f"{stop_price_rounded:.6f} (шаг={tick_size})"
            )
        condition = StopCondition.LAST_DOWN if is_long else StopCondition.LAST_UP
        order = Order(
            account_id=account_id,
            symbol=symbol,
            quantity=FinamDecimal(value=str(lots)),
            side=close_side,
            type=OrderType.STOP,
            stop_price=_finam_price_dec(stop_price_rounded),
            stop_condition=condition,
            time_in_force=TimeInForce.GOOD_TILL_CANCEL,
        )
        return await self._place_order_model(order, "STOP-LOSS")

    # ── тейк-профит (лимитная заявка) ─────────────────────────────────────────
    async def _place_take_profit(self, symbol: str, close_side: Side,
                                  lots: int, account_id: str,
                                  take_price: float,
                                  tick_size: float | None = None) -> str | None:
        if tick_size is None or tick_size <= 0:
            tick_size = moex_tick_size(take_price)
        take_price_rounded = round_to_tick_size(take_price, tick_size)
        if abs(take_price - take_price_rounded) > 1e-6:
            order_logger.warning(
                f"TAKE-PROFIT {symbol}: цена округлена {take_price:.6f} → "
                f"{take_price_rounded:.6f} (шаг={tick_size})"
            )
        order = Order(
            account_id=account_id,
            symbol=symbol,
            quantity=FinamDecimal(value=str(lots)),
            side=close_side,
            type=OrderType.LIMIT,
            limit_price=_finam_price_dec(take_price_rounded),
            time_in_force=TimeInForce.GOOD_TILL_CANCEL,
        )
        return await self._place_order_model(order, "TAKE-PROFIT")

    # ── стоп-лосс с повторными попытками ────────────────────────────────────
    async def _place_stop_loss_with_retry(self, symbol: str, close_side: Side,
                                           lots: int, account_id: str,
                                           stop_price: float, is_long: bool,
                                           max_retries: int = 3,
                                           tick_size: float | None = None) -> str | None:
        """Пытается выставить стоп-лосс с повторными попытками при ошибке API."""
        for attempt in range(1, max_retries + 1):
            order_logger.info(f"[RETRY {attempt}/{max_retries}] Выставление стопа {symbol}...")
            sl_id = await self._place_stop_loss(
                symbol, close_side, lots, account_id, stop_price, is_long, tick_size=tick_size
            )
            if sl_id is not None:
                order_logger.info(f"[RETRY {attempt}/{max_retries}] УСПЕХ: sl_id={sl_id}")
                return sl_id
            if attempt < max_retries:
                await asyncio.sleep(1)
        order_logger.error(f"[RETRY] Все {max_retries} попытки выставления стопа {symbol} неудачны")
        return None

    # ── главный метод ─────────────────────────────────────────────────────────
    async def execute_signal_async(self, sig: dict) -> bool:
        # ═══════════════════════════════════════════════════════════════
        # ПРИНУДИТЕЛЬНАЯ ПРОВЕРКА LIVE_TRADING (дублирующий контур
        # безопасности на случай, если модульная константа не сработала).
        # ═══════════════════════════════════════════════════════════════
        if not LIVE_TRADING:
            order_logger.critical(
                f"[SAFETY-GATE] LIVE_TRADING=0 — блокировка ВСЕХ ордеров. "
                f"Сигнал {sig.get('ticker','?')} {sig.get('direction','?')} отклонён."
            )
            return False

        ticker    = sig["ticker"]
        exchange  = sig["exchange"]
        direction = sig["direction"]   # "LONG" / "SHORT"
        entry     = float(sig["entry"])
        stop      = float(sig["stop"])
        target    = float(sig["target"])
        risk      = abs(entry - stop)

        # ── Валидация стоп-лосса и тейк-профита ─────────────────────
        if stop <= 0 or target <= 0:
            order_logger.error(
                f"[SAFETY] {ticker}: стоп ({stop}) или цель ({target}) "
                f"не могут быть <= 0. Ордер отклонён."
            )
            return False
        if risk <= 0:
            order_logger.error(
                f"[SAFETY] {ticker}: риск = {risk} (entry={entry}, stop={stop}). "
                f"Некорректные параметры. Ордер отклонён."
            )
            return False

        symbol    = _symbol(ticker, exchange)
        is_long   = direction == "LONG"
        entry_side = Side.BUY  if is_long else Side.SELL
        close_side = Side.SELL if is_long else Side.BUY

        order_logger.info(
            f"── СИГНАЛ: {ticker} ({exchange}) {direction} "
            f"| symbol={symbol} entry={entry} SL={stop} TP={target} "
            f"| risk={risk:.4f} model_prob={sig.get('model_prob', '?')}"
        )

        # ── Реальные шаг цены и лотность инструмента (из Finam API) ──────
        tick_size, live_lot_size = await self._get_asset_info(symbol)
        if tick_size is None or tick_size <= 0:
            tick_size = moex_tick_size(entry)
        if live_lot_size is None or live_lot_size <= 0:
            live_lot_size = self._lots_from_ticker(ticker)

        free_cash = await self.get_free_cash()
        if free_cash <= 0:
            order_logger.error("Нет свободных средств. Ордер отменён.")
            return False

        lots = self.calc_lots(free_cash, risk, ticker, entry_price=entry, lot_size=live_lot_size)
        if lots <= 0:
            order_logger.warning(
                f"{ticker}: 0 лотов по расчёту (риск={risk:.4f}, entry={entry:.2f}, баланс={free_cash:.2f}). Пропуск."
            )
            return False

        lot_size = live_lot_size
        shares_total = lots * lot_size
        position_value = shares_total * entry
        risk_total = shares_total * risk

        # ── ЖЁСТКИЙ ЛИМИТ СТОИМОСТИ ПОЗИЦИИ (защита от маржинальных требований) ──
        if position_value > MAX_POSITION_VALUE_RUB:
            old_lots = lots
            max_shares = int(MAX_POSITION_VALUE_RUB / entry)
            max_shares = max(max_shares, lot_size)  # минимум 1 лот
            max_shares = (max_shares // lot_size) * lot_size  # кратно лоту
            lots = max_shares // lot_size
            shares_total = lots * lot_size
            position_value = shares_total * entry
            risk_total = shares_total * risk
            order_logger.warning(
                f"{ticker}: Стоимость позиции {position_value:.2f} > лимит "
                f"{MAX_POSITION_VALUE_RUB:.0f}. Лотов урезано: {old_lots} → {lots} "
                f"(было {old_lots * lot_size} шт.)"
            )
        order_logger.info(
            f"{ticker}: {lots} лот(ов) × {lot_size} акций = {shares_total} шт., "
            f"стоимость≈{shares_total*entry:.2f}, "
            f"риск≈{risk_total:.2f} ({risk_total/free_cash*100:.2f}% депозита)"
        )

        # 1. Рыночный вход
        order_id = await self._place_market_order(symbol, entry_side, lots, FINAM_CLIENT_ID)
        if order_id is None:
            return False

        # ── UNCERTAIN: API вернул ошибку, но позиция найдена на аккаунте ──
        order_uncertain = (order_id == "UNCERTAIN")
        if order_uncertain:
            order_logger.warning(
                f"{ticker}: Вход выполнен в режиме UNCERTAIN — "
                f"продолжаем с выставлением SL/TP."
            )

        # Пауза чтобы вход исполнился до выставления защитных ордеров
        await asyncio.sleep(2)

        # 2. Стоп-лосс (ОБЯЗАТЕЛЕН) — с повторными попытками
        sl_id = await self._place_stop_loss_with_retry(
            symbol, close_side, lots, FINAM_CLIENT_ID, stop, is_long,
            max_retries=3, tick_size=tick_size
        )

        # 3. Тейк-профит
        tp_id = await self._place_take_profit(
            symbol, close_side, lots, FINAM_CLIENT_ID, target, tick_size=tick_size
        )

        # ═══════════════════════════════════════════════════════════════
        # АВАРИЙНЫЙ КОНТУР: если стоп-лосс не выставился —
        # немедленно закрываем позицию рыночным ордером.
        # ═══════════════════════════════════════════════════════════════
        if sl_id is None:
            order_logger.critical(
                f"{ticker}: СТОП-ЛОСС НЕ ВЫСТАВЛЕН! Аварийное закрытие позиции..."
            )
            close_id = await self._place_market_order(symbol, close_side, lots, FINAM_CLIENT_ID)

            # ── Если close_id is None — ордер точно не исполнился ──
            # ── Если close_id == "UNCERTAIN" — API ошибка, проверяем позицию ──
            if close_id is not None and close_id != "UNCERTAIN":
                order_logger.warning(
                    f"{ticker}: Позиция аварийно закрыта (order_id={close_id}). "
                    f"Потеряна комиссия за вход+выход, но капитал сохранён."
                )
            else:
                final_pos = await self.get_position_quantity(ticker)
                if final_pos > 0:
                    order_logger.critical(
                        f"{ticker}: КРИТИЧЕСКАЯ СИТУАЦИЯ — аварийное закрытие "
                        f"не подтверждено, но позиция ({final_pos} шт.) ВСЁ ЕЩЁ "
                        f"ОТКРЫТА. Требуется ручное вмешательство!"
                    )
                else:
                    order_logger.critical(
                        f"{ticker}: КРИТИЧЕСКАЯ ОШИБКА — не удалось закрыть позицию! "
                        f"Проверьте счёт Финам вручную немедленно!"
                    )
            return False

        if tp_id is None:
            order_logger.warning(
                f"{ticker}: SL={sl_id} выставлен, но TP не выставлен. "
                f"Позиция защищена стоп-лоссом, но без тейк-профита."
            )
        else:
            log_id = order_id if not order_uncertain else "UNCERTAIN"
            order_logger.info(
                f"{ticker}: Позиция открыта полностью. "
                f"order_id={log_id} SL={sl_id} TP={tp_id}"
            )
        return True


# ── публичный синхронный интерфейс ───────────────────────────────────────────
def execute_signal(sig: dict) -> bool:
    """Вызывается из step11b_paper_trading.py для каждого нового сигнала."""
    if not LIVE_TRADING:
        order_logger.info(
            f"[PAPER] {sig['ticker']} {sig['direction']} "
            f"entry={sig['entry']} SL={sig['stop']} TP={sig['target']} "
            f"prob={sig.get('model_prob','?')} — LIVE_TRADING=0, ордер не выставляется"
        )
        return True

    if not FINAM_TOKEN or not FINAM_CLIENT_ID:
        order_logger.error(
            "LIVE_TRADING=1 но FINAM_TOKEN или FINAM_CLIENT_ID не заданы."
        )
        return False

    if sig.get("exchange") == "NASDAQ":
        order_logger.warning(
            f"[SKIP] {sig['ticker']} (NASDAQ) — убедитесь что инструмент доступен "
            f"на вашем счёте Финам (символ: {_symbol(sig['ticker'], 'NASDAQ')}). "
            "Раскомментируйте 'pass' ниже чтобы пробовать выставить ордер."
        )
        # pass   # ← раскомментировать чтобы пробовать NASDAQ-ордера
        return False

    if sig.get("exchange") == "FORTS":
        order_logger.warning(
            f"[SKIP] {sig['ticker']} (FORTS) — фьючерсы временно отключены: "
            f"MAX_POSITION_VALUE_RUB считает от номинала сделки, а не от реального "
            f"гарантийного обеспечения (ГО), из-за чего ордера отклоняются с "
            f"'No enough coverage'. Требуется отдельная маржинальная логика."
        )
        return False

    async def _run() -> bool:
        broker = FinamBroker()
        ok = await broker.connect()
        if not ok:
            return False
        return await broker.execute_signal_async(sig)

    try:
        return asyncio.run(_run())
    except Exception as exc:
        order_logger.error(f"Критическая ошибка: {exc}")
        return False


# ── проверка соединения (python broker_finam.py) ──────────────────────────────
def check_connection() -> None:
    async def _check():
        broker = FinamBroker()
        if not await broker.connect():
            return
        cash = await broker.get_free_cash()
        print(f"Соединение OK. Свободных средств: {cash:.2f}")

    asyncio.run(_check())


if __name__ == "__main__":
    print(f"finam-trade-api доступна : {_FINAM_AVAILABLE}"
          + (f" (ошибка: {_FINAM_IMPORT_ERR})" if not _FINAM_AVAILABLE else ""))
    print(f".env путь                : {_env_path} ({'есть' if os.path.exists(_env_path) else 'НЕТ'})")
    print(f"FINAM_TOKEN              : {'***' if FINAM_TOKEN else 'НЕ ЗАДАН'}")
    print(f"FINAM_CLIENT_ID          : {FINAM_CLIENT_ID or 'НЕ ЗАДАН'}")
    print(f"LIVE_TRADING             : {LIVE_TRADING}")
    print(f"LIVE_TRADING(raw env)    : {os.getenv('LIVE_TRADING', 'не задан')}")
    print(f"RISK_PCT                 : {RISK_PCT:.1%}")
    print()
    check_connection()
