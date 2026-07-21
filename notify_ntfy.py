"""
notify_ntfy.py — push-уведомления через ntfy.sh для торгового бота.

Топик: mark_trading_2026
API: POST https://ntfy.sh/mark_trading_2026
Использует английские заголовки с базовыми эмодзи (UTF-8).
"""
import os
import json
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

def send_signal_for_approval(sig: dict, atr_val: float = 0, screenshot_path: str = None) -> bool:
    """Отправляет сигнал в NTFY — всё на русском.
    Картинка: query params (поддерживают UTF-8 через URL-encoding).
    Текст: JSON API.
    """
    ticker    = sig["ticker"]
    direction = sig["direction"]
    entry     = sig.get("entry", 0)
    stop      = sig.get("stop", 0)
    target    = sig.get("target", 0)
    level     = sig.get("level", 0)
    rr        = sig.get("rr", 3.0)
    trend     = sig.get("trend", "?")

    d_ru  = "ЛОНГ" if direction == "LONG" else "ШОРТ"
    d_tag = "green_circle" if direction == "LONG" else "red_circle"

    tv_link = (
        f"https://www.tradingview.com/chart/"
        f"?symbol=MOEX:{ticker}&interval=60"
    )

    title_ru = f"⚡ СИГНАЛ: {ticker} {d_ru}"
    msg_ru = (
        f"Вход: {entry:.2f} | Стоп: {stop:.2f} | "
        f"Цель: {target:.2f}\n"
        f"Уровень: {level:.2f} | R:R {rr:.1f}:1\n"
        f"ATR: {atr_val:.4f} | Тренд: {trend}"
    )
    actions_ru = (
        f"view, ✅ ПРИНЯТЬ (TV H1), {tv_link}, clear=true; "
        f"view, ❌ ПРОПУСТИТЬ, https://ntfy.sh/{NTFY_TOPIC}, clear=true"
    )

    try:
        if screenshot_path and os.path.exists(screenshot_path):
            with open(screenshot_path, "rb") as f:
                img_data = f.read()
            params = {
                "title": title_ru,
                "message": msg_ru,
                "priority": "5",
                "tags": f"{d_tag},rotating_light",
                "filename": f"{ticker}_signal.png",
                "actions": actions_ru,
            }
            r = requests.put(NTFY_URL, data=img_data,
                             params=params, timeout=15)
        else:
            payload = {
                "topic": NTFY_TOPIC,
                "title": title_ru,
                "message": msg_ru,
                "priority": 5,
                "tags": [d_tag, "rotating_light", "chart_with_upwards_trend"],
                "actions": [
                    {"action": "view",
                     "label": "✅ ПРИНЯТЬ (TV H1)",
                     "url": tv_link, "clear": True},
                    {"action": "view",
                     "label": "❌ ПРОПУСТИТЬ",
                     "url": f"https://ntfy.sh/{NTFY_TOPIC}", "clear": True},
                ],
            }
            r = requests.post(NTFY_URL, json=payload, timeout=10)
        return r.status_code == 200
    except Exception as e:
        print(f"  [NTFY ERR] {e}")
        return False


def send_signal_screenshot(sig: dict, screenshot_path: str) -> bool:
    """Отправляет в NTFY PNG-скриншот сигнала (уровень/стоп/цель/вход
    нарисованы на графике в screenshot_chart.make_screenshot()) вместо
    текстового сообщения. Кнопки: открыть TradingView, APPROVE, SKIP.

    HTTP-заголовки поддерживают только ASCII — Title и Actions без кириллицы,
    описание сделки остаётся только на самой картинке.
    """
    ticker    = sig["ticker"]
    direction = sig["direction"]
    d_tag     = "green_circle" if direction == "LONG" else "red_circle"

    if not os.path.exists(screenshot_path):
        print(f"  [NTFY ERR] скриншот не найден: {screenshot_path}")
        return False

    tv_url = f"https://www.tradingview.com/chart/?symbol=MOEX:{ticker}&interval=60"

    safe_title = f"SIGNAL {ticker} {direction}".encode("ascii", errors="replace").decode("ascii")
    headers = {
        "Title": safe_title,
        "Priority": "5",
        "Tags": f"{d_tag},rotating_light",
        "Filename": f"{ticker}.png",
        # ASCII-only: HTTP-заголовки не поддерживают кириллицу
        "Actions": (
            f"view, TradingView, {tv_url}; "
            f"http, APPROVE, {NTFY_URL}, method=POST, "
            f"headers.X-Title=APPROVED {ticker} {direction}, clear=true; "
            f"http, SKIP, {NTFY_URL}, method=POST, "
            f"headers.X-Title=SKIPPED {ticker}, clear=true"
        ),
    }
    try:
        with open(screenshot_path, "rb") as f:
            r = requests.post(NTFY_URL, data=f.read(), headers=headers, timeout=15)
        return r.status_code == 200
    except Exception as e:
        print(f"  [NTFY ERR] {e}")
        return False
