"""
notify_ntfy.py — push-уведомления через ntfy.sh для торгового бота.

Топик: mark_trading_2026
API: POST https://ntfy.sh/mark_trading_2026
Использует английские заголовки с базовыми эмодзи (UTF-8).
"""
import os
import requests
from datetime import datetime

NTFY_TOPIC = os.getenv("NTFY_TOPIC", "mark_trading_2026")
NTFY_URL = f"https://ntfy.sh/{NTFY_TOPIC}"


def _send(message: str, title: str = "", priority: int = 3, tags: list = None):
    """Базовый вызов ntfy.sh. Заголовок — только ASCII (без эмодзи в header).
    Эмодзи передаются через Tags — ntfy рендерит их автоматически."""
    # HTTP-заголовки не поддерживают UTF-8 эмодзи (latin-1),
    # поэтому title — только ASCII; эмодзи идут через tags
    safe_title = title.encode("ascii", errors="replace").decode("ascii")
    headers = {
        "Title": safe_title,
        "Priority": str(priority),
        "Tags": ",".join(tags or []),
    }
    try:
        r = requests.post(NTFY_URL, data=message.encode("utf-8"),
                          headers=headers, timeout=10)
        return r.status_code == 200
    except Exception as e:
        print(f"  [NTFY ERR] {e}")
        return False


def send_open_signal(sig: dict, atr_val: float, cost_ratio: float, hist_pf: float, lots: int):
    """Шаблон A — При открытии сделки (новый сигнал)."""
    try:
        from config_trading import RR_TARGET
    except ImportError:
        RR_TARGET = 3.0

    ticker = sig["ticker"]
    direction = sig["direction"]
    d_tag = "green_circle" if direction == "LONG" else "red_circle"
    d_ru = "ЛОНГ" if direction == "LONG" else "ШОРТ"

    msg = (
        f"🤖 СИГНАЛ: {ticker} -> {d_ru}\n"
        f"Вход: {sig['entry']:.2f} руб. | Тейк: {sig['target']:.2f} руб. | "
        f"Стоп: {sig['stop']:.2f} руб.\n"
        f"ATR: {atr_val:.4f} | Издержки: {cost_ratio:.1%}\n"
        f"PF бэктеста: {hist_pf:.3f} | Объем: {lots} лотов | "
        f"Уровень: {sig.get('level','?')}"
    )
    return _send(msg, title=f"SIGNAL {ticker} {direction}",
                 priority=4, tags=[d_tag, "chart_with_upwards_trend"])


def send_close_signal(sig: dict, exit_price: float, status: str, pnl_r: float, risk_rub: float = 100.0):
    """Шаблон B — При фиксации сделки (выход из рынка)."""
    ticker = sig["ticker"]
    direction = sig["direction"]
    d_ru = "ЛОНГ" if direction == "LONG" else "ШОРТ"

    status_tags = {"TAKE_PROFIT": "white_check_mark",
                   "STOP_LOSS": "no_entry_sign",
                   "TIMEOUT": "alarm_clock"}
    status_ru = {"TAKE_PROFIT": "ТЕЙК-ПРОФИТ",
                 "STOP_LOSS": "СТОП-ЛОСС",
                 "TIMEOUT": "ТАЙМАУТ"}
    tag = status_tags.get(status, "heavy_minus_sign")
    label = status.replace("_", " ").title()

    financial = pnl_r * risk_rub
    fin_sign = "+" if financial >= 0 else ""
    pnl_sign = "+" if pnl_r >= 0 else ""

    msg = (
        f"🏁 ЗАКРЫТИЕ: {ticker} ({d_ru})\n"
        f"Выход по: {exit_price:.4f} руб. | "
        f"Статус: {status_ru.get(status, status)}\n"
        f"Прибыль/Убыток: {pnl_sign}{pnl_r:.2f}R\n"
        f"Итог в рублях: {fin_sign}{financial:.2f} руб."
    )
    return _send(msg, title=f"{label}: {ticker}",
                 priority=4, tags=[tag, "moneybag"])


def send_portfolio_radar(active_positions: list):
    """Шаблон C — Ежечасный радар портфеля."""
    if not active_positions:
        # Не слать уведомление, если нет активных позиций
        return

    lines = [f"📊 РАДАР ПОРТФЕЛЯ: {len(active_positions)} позиций\n"]
    for pos in active_positions:
        d_tag = "green_circle" if pos["direction"] == "LONG" else "red_circle"
        d_emoji = "🟢" if pos["direction"] == "LONG" else "🔴"
        d_ru = "ЛОНГ" if pos["direction"] == "LONG" else "ШОРТ"
        comment = pos.get("status_comment", "")
        lines.append(
            f"{d_emoji} {pos['ticker']} ({d_ru}):\n"
            f"   Вход: {pos['entry']} | Текущая: {pos['current_price']}\n"
            f"   До Тейка: {pos['pct_to_tp']:+.1f}% | "
            f"До Стопа: {pos['pct_to_sl']:+.1f}%\n"
            f"   Статус: {comment}\n"
        )

    msg = "\n".join(lines)
    return _send(msg, title=f"PORTFOLIO RADAR: {len(active_positions)} positions",
                 priority=3, tags=["bar_chart"])


def generate_status_comment(pct_to_tp: float, pct_to_sl: float, direction: str) -> str:
    """Генерирует комментарий по % расстоянию от текущей цены до TP/SL.
    pct_to_tp: % пути от entry до target (100%=у входа, 0%=у target)
    pct_to_sl: % пути от entry до stop  (100%=у входа, 0%=у stop)
    """

    if pct_to_tp == 0.0:
        return "Тейк достигнут! 🎯"
    elif pct_to_sl == 0.0:
        return "Стоп-лосс сработал! 🛑"
    elif pct_to_tp < 25.0:
        return "Уже близко к тейку, ещё немного! 📈"
    elif pct_to_sl < 25.0:
        return "Близко к стопу, следим! ⚠️"
    elif pct_to_tp < pct_to_sl:
        return "Движется к тейку, всё по плану 🌊"
    else:
        return "Позиция активна, всё штатно ✅"

def send_signal_for_approval(sig: dict) -> bool:
    """Шаблон D — Запрос одобрения сигнала с кнопками ОДОБРИТЬ / ПРОПУСТИТЬ.
    Кнопка ОДОБРИТЬ открывает TradingView для визуальной проверки.
    Кнопка ПРОПУСТИТЬ закрывает уведомление без действия.
    """
    ticker     = sig["ticker"]
    direction  = sig["direction"]
    d_ru       = "ЛОНГ" if direction == "LONG" else "ШОРТ"
    d_tag      = "green_circle" if direction == "LONG" else "red_circle"
    entry      = sig.get("entry", 0)
    stop       = sig.get("stop", 0)
    target     = sig.get("target", 0)
    rr         = sig.get("rr", "?")
    level      = sig.get("level", "?")
    atr_daily  = sig.get("atr_daily", 0)
    trend      = sig.get("trend", "?")
    bar_time   = sig.get("bar_time", "?")

    tv_url = f"https://www.tradingview.com/chart/?symbol=MOEX:{ticker}"

    msg = (
        f"\u23f3 ОДОБРЕНИЕ: {ticker} \u2192 {d_ru}\n"
        f"Бар: {bar_time}\n"
        f"Вход: {entry:.2f} | Стоп: {stop:.2f} | Цель: {target:.2f}\n"
        f"R:R = {rr}:1 | Уровень: {level}\n"
        f"ATR дневной: {atr_daily:.4f} | Тренд: {trend}\n"
        f"Открой график и реши: ОДОБРИТЬ или ПРОПУСТИТЬ"
    )

    safe_title = f"APPROVAL? {ticker} {direction}".encode("ascii", errors="replace").decode("ascii")
    headers = {
        "Title": safe_title,
        "Priority": "5",
        "Tags": f"{d_tag},eyes,bell",
        # HTTP headers не поддерживают UTF-8 → ASCII метки кнопок
        "Actions": (
            f"view, APPROVE (TV), {tv_url}, clear=true; "
            f"view, SKIP, https://ntfy.sh/{NTFY_TOPIC}, clear=true"
        ),
    }
    try:
        r = requests.post(NTFY_URL, data=msg.encode("utf-8"),
                          headers=headers, timeout=10)
        return r.status_code == 200
    except Exception as e:
        print(f"  [NTFY ERR] {e}")
        return False
