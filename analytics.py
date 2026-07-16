"""
analytics.py — Форвард-тест анализатор MOEX торгового робота.

Автономно читает файлы бумажной торговли и бэктеста,
рассчитывает текущие метрики и выводит красивую таблицу.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

DATA_DIR = "data"

# ── какие файлы читаем ──────────────────────────────────────
SIGNALS_LIVE = os.path.join(DATA_DIR, "signals_live.csv")    # бумажные сигналы (step11b)
PORTFOLIO    = os.path.join(DATA_DIR, "portfolio_final.csv")  # бэктест-сделки (step_portfolio_final)
TRADING_LOG  = os.path.join("logs", "trading.log")            # лог сканера

# ── цвета (ANSI) ────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def fmt(v, color=""):
    return f"{color}{v}{RESET}"

def color_r(r):
    """Зелёный для >0, красный для <0"""
    return GREEN if r > 0 else (RED if r < 0 else RESET)

def color_ratio(r, good_pf=True):
    if good_pf:
        return GREEN if r >= 1.5 else (YELLOW if r >= 1.0 else RED)
    return GREEN if r >= 0.3 else (YELLOW if r >= 0.1 else RED)

# ── анализ сигналов (бумажная торговля) ──────────────────────
def analyze_signals_live(path):
    """Анализирует файл signals_live.csv — неисполненные сигналы сканера."""
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if len(df) == 0:
        return None

    # Нормализация колонок (step11b формат)
    if "direction" in df.columns:
        longs  = (df["direction"].str.upper() == "LONG").sum()
        shorts = (df["direction"].str.upper() == "SHORT").sum()
    elif "is_long" in df.columns:
        longs  = df["is_long"].astype(int).sum()
        shorts = len(df) - longs
    else:
        longs = shorts = 0

    # Тикеры
    tickers_col = "ticker" if "ticker" in df.columns else None
    if tickers_col:
        unique_tickers = df[tickers_col].nunique()
        ticker_list    = df[tickers_col].value_counts().head(10)
    else:
        unique_tickers = 0
        ticker_list    = pd.Series(dtype=int)

    # Временной диапазон
    time_cols = [c for c in ["scan_time", "bar_time", "datetime"] if c in df.columns]
    ts_from, ts_to = "—", "—"
    if time_cols:
        tc = time_cols[0]
        ts_parsed = pd.to_datetime(df[tc], errors="coerce")
        ts_parsed = ts_parsed.dropna()
        if len(ts_parsed) > 0:
            ts_from = ts_parsed.min().strftime("%Y-%m-%d %H:%M")
            ts_to   = ts_parsed.max().strftime("%Y-%m-%d %H:%M")

    # Средний risk / entry
    avg_risk = df["risk"].mean() if "risk" in df.columns else None
    avg_entry = df["entry"].mean() if "entry" in df.columns else None

    return {
        "total":    len(df),
        "longs":    int(longs),
        "shorts":   int(shorts),
        "tickers":  unique_tickers,
        "top":      ticker_list,
        "from":     ts_from,
        "to":       ts_to,
        "avg_risk": avg_risk,
        "avg_entry": avg_entry,
    }


# ── анализ сделок (бэктест / форвард) ────────────────────────
def analyze_trades(path):
    """Анализирует файл portfolio_final.csv — исполненные сделки с outcome/pnl_r."""
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if len(df) == 0:
        return None

    # Нормализация колонок
    if "pnl_r" not in df.columns:
        return None

    pnl = df["pnl_r"].values
    n   = len(pnl)

    # Профит/лосс
    total_r    = pnl.sum()
    mean_r     = pnl.mean()

    # Winrate (TP = win, SL/TIMEOUT = loss)
    if "outcome" in df.columns:
        wins  = (df["outcome"] == "TP").sum()
        losses = n - wins
    else:
        wins  = (pnl > 0).sum()
        losses = (pnl <= 0).sum()

    winrate = wins / n if n > 0 else 0.0

    # Profit Factor
    gross_profit  = pnl[pnl > 0].sum() if (pnl > 0).any() else 0.0
    gross_loss    = abs(pnl[pnl < 0].sum()) if (pnl < 0).any() else 0.0
    pf = gross_profit / gross_loss if gross_loss > 0 else (99.9 if gross_profit > 0 else 0.0)

    # Просадка
    eq    = np.cumsum(pnl)
    peak  = np.maximum.accumulate(eq)
    dd    = eq - peak
    max_dd_r = dd.min()
    dd_pct   = (dd / np.maximum(peak, 1e-9)).min() * 100

    # Полоса проигрышей
    streak = 0
    max_streak = 0
    for r in pnl:
        if r < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    # Sharpe-like (R-based)
    sharpe = pnl.mean() / pnl.std() * np.sqrt(n) if pnl.std() > 0 else 0.0

    # Expectancy
    expectancy = mean_r

    # Тренд equity — простейшая линейная регрессия
    x = np.arange(n)
    if n > 1:
        slope = np.polyfit(x, eq, 1)[0]
    else:
        slope = 0.0

    # Среднее число баров в сделке
    avg_bars = df["bars_held"].mean() if "bars_held" in df.columns else None

    # Лучшие/худшие тикеры
    if "ticker" in df.columns:
        ticker_pnl = df.groupby("ticker")["pnl_r"].agg(["sum", "count", "mean"])
        ticker_pnl = ticker_pnl.sort_values("sum", ascending=False)
        # Добавляем winrate по тикеру
        if "outcome" in df.columns:
            ticker_wr = df[df["outcome"] == "TP"].groupby("ticker").size()
            ticker_n  = df.groupby("ticker").size()
            wr_series = (ticker_wr / ticker_n * 100).fillna(0)
            ticker_pnl["wr_pct"] = wr_series
        else:
            ticker_pnl["wr_pct"] = None
    else:
        ticker_pnl = pd.DataFrame()

    # Временной диапазон
    if "datetime" in df.columns:
        times = pd.to_datetime(df["datetime"], errors="coerce").dropna()
        days_span = (times.max() - times.min()).days if len(times) > 1 else 0
        trades_per_week = n / max(days_span / 7, 1)
    else:
        days_span = 0
        trades_per_week = 0.0

    return {
        "n":            n,
        "wins":         int(wins),
        "losses":       int(losses),
        "winrate":      winrate,
        "total_r":      total_r,
        "mean_r":       mean_r,
        "pf":           pf,
        "gross_profit": gross_profit,
        "gross_loss":   gross_loss,
        "max_dd_r":     max_dd_r,
        "dd_pct":       dd_pct,
        "max_streak":   max_streak,
        "sharpe":       sharpe,
        "expectancy":   expectancy,
        "slope":        slope,
        "avg_bars":     avg_bars,
        "ticker_pnl":   ticker_pnl,
        "days_span":    days_span,
        "trades_per_week": trades_per_week,
    }


# ── последние строки лога ─────────────────────────────────────
def tail_log(path, n_lines=12):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        tail = [l.strip() for l in lines[-n_lines:] if l.strip()]
        return tail if tail else None
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════
#  ВЫВОД
# ═══════════════════════════════════════════════════════════════
W = 78  # ширина

def print_header(title):
    print()
    print("═" * W)
    title_pad = (W - len(title)) // 2 - 1
    print(f"{' ' * title_pad} {BOLD}{title}{RESET}")
    print("═" * W)

def print_footer():
    print("═" * W)
    print()

def print_kv(key, val, unit="", col=None):
    if col is None:
        col = CYAN
    print(f"  {key:<22} {col}{val}{RESET}{' ' + unit if unit else ''}")

def print_separator():
    print(f"  {'─' * (W - 4)}")

# ──────────────────────────────────────────────────────────────
now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print()
print(f"  {BOLD}📊 MOEX TRADING BOT — АНАЛИТИКА ФОРВАРД-ТЕСТА{RESET}")
print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  LIVE_TRADING=0  |  PAPER MODE")
print()

# ── раздел 1: бумажные сигналы (signals_live.csv) ────────────
sig_stats = analyze_signals_live(SIGNALS_LIVE)
print_header("📡 БУМАЖНЫЕ СИГНАЛЫ (signals_live.csv)")

if sig_stats is None:
    print(f"  {YELLOW}⚠  Файл не найден или пуст.{RESET}")
    print(f"  {YELLOW}   Cron-сканер ещё не обнаружил сигналов, либо не запускался.{RESET}")
else:
    print_kv("Всего сигналов",    f"{sig_stats['total']}", col=YELLOW)
    print_kv("Лонги / Шорты",    f"{sig_stats['longs']} / {sig_stats['shorts']}")
    print_kv("Уникальных тикеров", sig_stats['tickers'])
    print_kv("Диапазон",          f"{sig_stats['from']} → {sig_stats['to']}")
    if sig_stats['avg_entry']:
        print_kv("Средняя цена входа", f"{sig_stats['avg_entry']:.2f}")
    if sig_stats['avg_risk']:
        print_kv("Средний риск",       f"{sig_stats['avg_risk']:.4f}")
    print()
    if len(sig_stats['top']) > 0:
        print(f"  {BOLD}Топ тикеров:{RESET}")
        for tkr, cnt in sig_stats['top'].items():
            bar = "█" * min(cnt, 20)
            print(f"    {tkr:<8} {cnt:>3}  {bar}")

print_footer()

# ── раздел 2: сделки (portfolio_final.csv) ───────────────────
tr_stats = analyze_trades(PORTFOLIO)
print_header("💼 СДЕЛКИ — ПОРТФЕЛЬНЫЙ БЭКТЕСТ (portfolio_final.csv)")

if tr_stats is None:
    print(f"  {RED}✖  Файл не найден или не содержит pnl_r.{RESET}")
else:
    # Сводка
    t_r_col = color_r(tr_stats['total_r'])
    pf_col  = color_ratio(tr_stats['pf'], good_pf=True)
    wr_col  = color_ratio(tr_stats['winrate'], good_pf=False)
    dd_col  = color_r(-abs(tr_stats['max_dd_r']))
    exp_col = color_r(tr_stats['expectancy'])

    print_kv("Всего сделок",         tr_stats['n'])
    print_kv("Прибыльных / Убыточных", f"{tr_stats['wins']} / {tr_stats['losses']}")
    print_kv("Winrate",              f"{tr_stats['winrate']*100:.1f}%", col=wr_col)
    print_kv("Profit Factor",        f"{tr_stats['pf']:.3f}", col=pf_col)
    print_kv("Total P&L",           f"{tr_stats['total_r']:+.2f}R", col=t_r_col)
    print_kv("Средняя сделка",       f"{tr_stats['mean_r']:+.4f}R", col=exp_col)
    print_kv("Expectancy",          f"{tr_stats['expectancy']:+.4f}R", col=exp_col)
    print_kv("Gross Profit",        f"{tr_stats['gross_profit']:.2f}R", col=GREEN)
    print_kv("Gross Loss",          f"{tr_stats['gross_loss']:.2f}R", col=RED)
    print_kv("Max Drawdown",        f"{tr_stats['max_dd_r']:.2f}R ({tr_stats['dd_pct']:.1f}%)",
              col=dd_col)
    print_kv("Sharpe (R-based)",    f"{tr_stats['sharpe']:.3f}",
              col=color_ratio(tr_stats['sharpe'], good_pf=False))
    print_kv("Max Loss Streak",     tr_stats['max_streak'])
    print_separator()
    print_kv("Период (дней)",       tr_stats['days_span'])
    print_kv("Сделок в неделю",    f"{tr_stats['trades_per_week']:.2f}")
    if tr_stats['avg_bars'] is not None:
        print_kv("Среднее баров в сделке", f"{tr_stats['avg_bars']:.1f}")
    if tr_stats['slope'] != 0:
        slope_label = "📈 рост" if tr_stats['slope'] > 0 else "📉 падение"
        print_kv("Тренд equity",    f"{slope_label} ({tr_stats['slope']:+.4f}R/сделку)")

    # Тикеры
    print()
    if len(tr_stats['ticker_pnl']) > 0:
        has_wr = "wr_pct" in tr_stats['ticker_pnl'].columns
        wr_header = "WR" if has_wr else ""
        print(f"  {BOLD}Результаты по тикерам:{RESET}")
        print(f"  {'Тикер':<8} {'Сделок':>7} {'Total R':>10} {'Средняя':>10} {wr_header:>7}")
        print(f"  {'─'*6:<8} {'─'*5:>7} {'─'*7:>10} {'─'*6:>10} {'─'*4:>7}")
        for tkr, row in tr_stats['ticker_pnl'].iterrows():
            t_col = color_r(row['sum'])
            wr_val = row.get('wr_pct', None)
            wr_str = f"{wr_val:.0f}%" if (wr_val is not None and not pd.isna(wr_val)) else ""
            cnt = int(row['count'])
            print(f"  {tkr:<8} {cnt:>7d} {t_col}{row['sum']:>+10.2f}R{RESET} "
                  f"{row['mean']:>+10.3f}R {wr_str:>7}")

print_footer()

# ── раздел 3: последние строки лога ──────────────────────────
log_tail = tail_log(TRADING_LOG)
print_header("📋 ПОСЛЕДНИЕ ЗАПИСИ ЛОГА СКАНЕРА (trading.log)")

if log_tail is None:
    print(f"  {YELLOW}⚠  trading.log не найден.{RESET}")
else:
    for line in log_tail:
        print(f"  {line}")

print_footer()

# ── раздел 4: итоговая оценка ────────────────────────────────
print_header("📈 ИТОГОВАЯ ОЦЕНКА ФОРВАРД-ТЕСТА")

if tr_stats is None:
    print(f"  {YELLOW}Нет данных для оценки.{RESET}")
else:
    score = 0
    checks = []

    if tr_stats['pf'] >= 1.5:
        checks.append((True, f"Profit Factor {tr_stats['pf']:.2f} ≥ 1.5"))
        score += 2
    elif tr_stats['pf'] >= 1.0:
        checks.append((True, f"Profit Factor {tr_stats['pf']:.2f} ≥ 1.0"))
        score += 1
    else:
        checks.append((False, f"Profit Factor {tr_stats['pf']:.2f} < 1.0 — отрицательная система"))

    if tr_stats['winrate'] >= 0.40:
        checks.append((True, f"Winrate {tr_stats['winrate']*100:.1f}% ≥ 40%"))
        score += 1
    else:
        checks.append((False, f"Winrate {tr_stats['winrate']*100:.1f}% < 40%"))

    if tr_stats['sharpe'] >= 1.0:
        checks.append((True, f"Sharpe {tr_stats['sharpe']:.2f} ≥ 1.0"))
        score += 2
    elif tr_stats['sharpe'] >= 0.5:
        checks.append((True, f"Sharpe {tr_stats['sharpe']:.2f} ≥ 0.5"))
        score += 1
    else:
        checks.append((False, f"Sharpe {tr_stats['sharpe']:.2f} < 0.5"))

    if tr_stats['max_dd_r'] > -20:
        checks.append((True, f"Max DD {tr_stats['max_dd_r']:.1f}R > -20R"))
        score += 1
    else:
        checks.append((False, f"Max DD {tr_stats['max_dd_r']:.1f}R ≤ -20R"))

    if tr_stats['max_streak'] < 10:
        checks.append((True, f"Max streak {tr_stats['max_streak']} < 10"))
        score += 1
    else:
        checks.append((False, f"Max streak {tr_stats['max_streak']} ≥ 10 — риск серий"))

    if tr_stats['expectancy'] >= 0.2:
        checks.append((True, f"Expectancy {tr_stats['expectancy']:.3f}R ≥ 0.2R"))
        score += 1
    elif tr_stats['expectancy'] > 0:
        checks.append((True, f"Expectancy {tr_stats['expectancy']:.3f}R > 0"))
        score += 0
    else:
        checks.append((False, f"Expectancy {tr_stats['expectancy']:.3f}R < 0"))

    for ok, msg in checks:
        icon = "✅" if ok else "❌"
        print(f"  {icon}  {msg}")

    print()
    if score >= 7:
        grade = f"{GREEN}ОТЛИЧНО{RESET} (система готова к LIVE)"
    elif score >= 4:
        grade = f"{YELLOW}СРЕДНЕ{RESET} (нужна донастройка параметров)"
    else:
        grade = f"{RED}СЛАБО{RESET} (система НЕ готова к реальной торговле)"

    print(f"  {BOLD}Оценка:{RESET} {grade}  ({score}/8 баллов)")

print_footer()
