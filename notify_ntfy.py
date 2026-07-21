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
NTFY_SERVER = "https://ntfy.sh"
NTFY_URL = f"{NTFY_SERVER}/{NTFY_TOPIC}"


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
    Картинка: HTTP-заголовки (UTF-8 байты, без latin-1 ошибок).
    Текст: JSON Publish API — POST на КОРНЕВОЙ URL сервера с "topic" в теле
    (POST на .../{topic} трактует JSON как обычный текст сообщения — мусор).
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

    # ПОДТВЕРДИТЬ/ГРАФИК — просто открывают TradingView (view, без запросов).
    # ПРОПУСТИТЬ — http-действие: POST маленького подтверждения в тот же
    # топик ("Сигнал пропущен: TICKER") и закрывает основное уведомление.
    # Title в http-действии — это HTTP-заголовок (ASCII/latin-1 only),
    # кириллица там ломает парсинг ntfy ("Unexpected char..."). Кириллицу
    # оставляем только в body — он поддерживает UTF-8.
    skip_title = "Skipped"
    skip_body = f"⏭ Сигнал пропущен: {ticker}"
    actions_ru = (
        f"view, ✅ ПОДТВЕРДИТЬ, {tv_link}, clear=true; "
        f"http, ❌ ПРОПУСТИТЬ, {NTFY_URL}, "
        f"method=POST, headers.Title={skip_title}, body={skip_body}, clear=true; "
        f"view, 📈 ГРАФИК, {tv_link}"
    )

    try:
        if screenshot_path and os.path.exists(screenshot_path):
            with open(screenshot_path, "rb") as f:
                img_data = f.read()
            headers = {
                "Title": title_ru.encode("utf-8"),
                # HTTP-заголовок не может содержать настоящий перевод строки —
                # передаём литерал \n, ntfy разворачивает его на своей стороне
                "Message": msg_ru.replace("\n", "\\n").encode("utf-8"),
                "Priority": b"5",
                "Tags": f"{d_tag},rotating_light".encode("utf-8"),
                "Filename": f"{ticker}_signal.png".encode("utf-8"),
                "Actions": actions_ru.encode("utf-8"),
            }
            r = requests.put(NTFY_URL, data=img_data,
                             headers=headers, timeout=15)
        else:
            payload = {
                "topic": NTFY_TOPIC,
                "title": title_ru,
                "message": msg_ru,
                "priority": 5,
                "tags": [d_tag, "rotating_light", "chart_with_upwards_trend"],
                "actions": [
                    {"action": "view",
                     "label": "✅ ПОДТВЕРДИТЬ",
                     "url": tv_link, "clear": True},
                    {"action": "http",
                     "label": "❌ ПРОПУСТИТЬ",
                     "url": NTFY_URL,
                     "method": "POST",
                     "headers": {"Title": skip_title},
                     "body": skip_body,
                     "clear": True},
                    {"action": "view",
                     "label": "📈 ГРАФИК",
                     "url": tv_link},
                ],
            }
            # JSON Publish API — обязательно корневой URL сервера, не /{topic}
            r = requests.post(NTFY_SERVER, json=payload, timeout=10)
        return r.status_code == 200
    except Exception as e:
        print(f"  [NTFY ERR] {e}")
        return False

