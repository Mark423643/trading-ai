"""
Step 10 — Финальный бэктест без META.
Загружает результаты step8, исключает META, строит полный отчёт.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

DATA_DIR = "data"
OUT_DIR  = "charts"
os.makedirs(OUT_DIR, exist_ok=True)

EXCLUDE  = ["META"]
RISK_PCT = 0.01        # 1% капитала на сделку
CAPITAL  = 10_000

# ── цветовая схема ──────────────────────────────────────
C = {
    "bg":      "#0d1117",
    "panel":   "#161b22",
    "border":  "#30363d",
    "green":   "#2ea043",
    "green2":  "#56d364",
    "red":     "#f85149",
    "yellow":  "#d29922",
    "blue":    "#58a6ff",
    "purple":  "#bc8cff",
    "text":    "#e6edf3",
    "sub":     "#8b949e",
    "grid":    "#21262d",
}

def style_ax(ax, title=""):
    ax.set_facecolor(C["panel"])
    for spine in ax.spines.values():
        spine.set_color(C["border"])
    ax.tick_params(colors=C["sub"], labelsize=8)
    ax.xaxis.label.set_color(C["sub"])
    ax.yaxis.label.set_color(C["sub"])
    ax.grid(True, color=C["grid"], linewidth=0.5, alpha=0.8)
    if title:
        ax.set_title(title, color=C["text"], fontsize=9, fontweight="bold", pad=6)

# ───────────────────────────────────────────────────────
# 1. Загрузка и фильтрация
# ───────────────────────────────────────────────────────
raw = pd.read_csv(os.path.join(DATA_DIR, "backtest_multi.csv"))
raw["datetime"] = pd.to_datetime(raw["datetime"], utc=True)
raw = raw.sort_values("datetime").reset_index(drop=True)

df_all  = raw.copy()
df      = raw[~raw["ticker"].isin(EXCLUDE)].copy().reset_index(drop=True)
df["trade_n"]  = range(1, len(df) + 1)
df["equity_r"] = df["pnl_r"].cumsum()

removed = df_all[df_all["ticker"].isin(EXCLUDE)]

print(f"Исключены тикеры: {EXCLUDE}")
print(f"Убрано сделок: {len(removed)}  (P&L META: {removed['pnl_r'].sum():.2f}R)")
print(f"Осталось сделок: {len(df)}")

# ───────────────────────────────────────────────────────
# 2. Метрики
# ───────────────────────────────────────────────────────
def calc_metrics(d):
    if len(d) == 0:
        return {}
    wins   = d[d["outcome"] == "TP"]
    losses = d[d["outcome"] != "TP"]
    eq     = d["pnl_r"].cumsum().values
    peak   = np.maximum.accumulate(eq)
    dd     = eq - peak

    streaks, cur = [], 0
    for r in d["pnl_r"]:
        cur = cur + 1 if r < 0 else 0
        streaks.append(cur)

    gp = wins["pnl_r"].sum()
    gl = losses["pnl_r"].abs().sum()

    # P&L компаундинг
    cap = CAPITAL
    for r in d["pnl_r"]:
        cap += r * cap * RISK_PCT

    std_r = d["pnl_r"].std()
    return {
        "n_trades":       len(d),
        "n_long":         int((d["long"] == True).sum()),
        "n_short":        int((d["long"] == False).sum()),
        "n_tp":           len(wins),
        "n_sl":           int((d["outcome"] == "SL").sum()),
        "n_timeout":      int((d["outcome"] == "TIMEOUT").sum()),
        "winrate":        len(wins) / len(d),
        "avg_win_r":      wins["pnl_r"].mean()   if len(wins)   > 0 else 0,
        "avg_loss_r":     losses["pnl_r"].mean() if len(losses) > 0 else 0,
        "total_r":        d["pnl_r"].sum(),
        "profit_factor":  gp / gl if gl > 0 else 0,
        "max_dd_r":       dd.min(),
        "max_streak":     max(streaks) if streaks else 0,
        "sharpe":         d["pnl_r"].mean() / std_r * np.sqrt(len(d)) if std_r > 0 else 0,
        "equity_final":   eq[-1],
        "capital_final":  cap,
        "return_pct":     (cap / CAPITAL - 1),
        "expectancy":     d["pnl_r"].mean(),       # E[R] per trade
        "avg_bars":       d["bars_held"].mean(),
    }

m_all  = calc_metrics(df_all)
m_clean = calc_metrics(df)

# ───────────────────────────────────────────────────────
# 3. Консольный отчёт
# ───────────────────────────────────────────────────────
TICKERS_CLEAN = sorted(df["ticker"].unique())

print("\n" + "=" * 62)
print("  ФИНАЛЬНЫЙ БЭКТЕСТ  |  14 тикеров  |  hourly  |  ~2 года")
print("=" * 62)

rows = [
    ("Тикеров в листе",     f"{len(TICKERS_CLEAN)}  ({', '.join(TICKERS_CLEAN)})"),
    ("Сделок всего",        f"{m_clean['n_trades']}  (лонг: {m_clean['n_long']}, шорт: {m_clean['n_short']})"),
    ("TP / SL / Timeout",   f"{m_clean['n_tp']} / {m_clean['n_sl']} / {m_clean['n_timeout']}"),
    ("Winrate",             f"{m_clean['winrate']:.1%}"),
    ("Avg Win (R)",         f"{m_clean['avg_win_r']:+.3f}"),
    ("Avg Loss (R)",        f"{m_clean['avg_loss_r']:+.3f}"),
    ("Expectancy (R/trade)",f"{m_clean['expectancy']:+.3f}"),
    ("Total P&L (R)",       f"{m_clean['total_r']:+.2f}R"),
    ("Profit Factor",       f"{m_clean['profit_factor']:.3f}"),
    ("Max Drawdown (R)",    f"{m_clean['max_dd_r']:.2f}R"),
    ("Max убыт. подряд",    f"{m_clean['max_streak']}"),
    ("Sharpe (R-based)",    f"{m_clean['sharpe']:.3f}"),
    ("Avg баров в сделке",  f"{m_clean['avg_bars']:.1f}"),
    ("P&L ($10k, 1%/trade)",f"${m_clean['capital_final']:,.0f}  ({m_clean['return_pct']:+.1%})"),
]

for label, val in rows:
    print(f"  {label:<25} {val}")

print("\n  Сравнение с/без META:")
print(f"  {'':25} {'С META':>12} {'Без META':>12}  {'Delta':>8}")
print(f"  {'-'*57}")
compare = [
    ("Сделок",       m_all["n_trades"],      m_clean["n_trades"]),
    ("Winrate",      m_all["winrate"],        m_clean["winrate"],        "%"),
    ("Total R",      m_all["total_r"],        m_clean["total_r"]),
    ("Profit Factor",m_all["profit_factor"],  m_clean["profit_factor"]),
    ("Max DD (R)",   m_all["max_dd_r"],       m_clean["max_dd_r"]),
    ("Sharpe",       m_all["sharpe"],         m_clean["sharpe"]),
    ("Return %",     m_all["return_pct"],     m_clean["return_pct"],     "%"),
]
for row in compare:
    lbl, v1, v2 = row[0], row[1], row[2]
    fmt = row[3] if len(row) > 3 else ""
    if fmt == "%":
        print(f"  {lbl:<25} {v1:>11.1%} {v2:>12.1%}  {v2-v1:>+7.1%}")
    else:
        print(f"  {lbl:<25} {v1:>12.2f} {v2:>12.2f}  {v2-v1:>+8.2f}")

# По тикерам
print(f"\n{'='*62}")
print("  РЕЗУЛЬТАТЫ ПО ТИКЕРАМ")
print(f"{'='*62}")
print(f"  {'Тикер':<7} {'Сделок':>6} {'Winrate':>8} {'Total R':>9} {'PF':>6} {'Sharpe':>8} {'$Return':>9}")
print(f"  {'-'*58}")

ticker_rows = []
for t in sorted(df["ticker"].unique()):
    sub = df[df["ticker"] == t]
    ms  = calc_metrics(sub)
    ticker_rows.append((ms["total_r"], t, ms))

for _, t, ms in sorted(ticker_rows, reverse=True):
    sign = "+" if ms["total_r"] >= 0 else ""
    print(f"  {t:<7} {ms['n_trades']:>6} {ms['winrate']:>8.0%} "
          f"{sign}{ms['total_r']:>8.2f}R {ms['profit_factor']:>6.2f} "
          f"{ms['sharpe']:>8.3f} {ms['return_pct']:>+8.1%}")

# Сохраняем чистые данные
df.to_csv(os.path.join(DATA_DIR, "backtest_final.csv"), index=False)
print(f"\n  Сохранено: {DATA_DIR}/backtest_final.csv")

# ───────────────────────────────────────────────────────
# 4. Финальный дашборд
# ───────────────────────────────────────────────────────
print("\nСтроим финальный дашборд...")

fig = plt.figure(figsize=(20, 15))
fig.patch.set_facecolor(C["bg"])
gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.50, wspace=0.35)

# ── 4.1 Главная equity curve (вся ширина) ───────────────
ax_eq = fig.add_subplot(gs[0, :])
style_ax(ax_eq, "Equity Curve  |  14 тикеров  |  Hourly  |  ~2 года")

eq   = df["equity_r"].values
x    = df["trade_n"].values
peak = np.maximum.accumulate(eq)
dd   = eq - peak

ax_eq.fill_between(x, 0, eq, where=(eq >= 0), color=C["green"],  alpha=0.12)
ax_eq.fill_between(x, 0, eq, where=(eq < 0),  color=C["red"],    alpha=0.12)

ax_dd = ax_eq.twinx()
ax_dd.fill_between(x, 0, dd, color=C["red"], alpha=0.18)
ax_dd.set_ylabel("Drawdown (R)", color=C["red"], fontsize=8)
ax_dd.tick_params(colors=C["red"], labelsize=7)
ax_dd.spines[["top","right","left","bottom"]].set_color(C["border"])
ax_dd.set_ylim(dd.min() * 3, 0)

ax_eq.plot(x, eq, color=C["blue"], linewidth=2.2, zorder=5)
ax_eq.axhline(0, color=C["border"], linewidth=1)

tp_m = df["outcome"] == "TP"
sl_m = df["outcome"] != "TP"
ax_eq.scatter(x[tp_m], eq[tp_m], color=C["green"], s=45, zorder=6, label="TP")
ax_eq.scatter(x[sl_m], eq[sl_m], color=C["red"],   s=45, zorder=6, marker="v", label="SL")

ax_eq.annotate(f" +{eq[-1]:.1f}R",
               xy=(x[-1], eq[-1]), color=C["green2"],
               fontsize=11, fontweight="bold", va="center")

# Stat box
stat = (f"Сделок: {m_clean['n_trades']}  |  Winrate: {m_clean['winrate']:.0%}  |  "
        f"Avg Win: +{m_clean['avg_win_r']:.2f}R  |  Avg Loss: {m_clean['avg_loss_r']:.2f}R  |  "
        f"PF: {m_clean['profit_factor']:.2f}  |  Sharpe: {m_clean['sharpe']:.2f}  |  "
        f"Max DD: {m_clean['max_dd_r']:.1f}R")
ax_eq.text(0.01, 0.96, stat, transform=ax_eq.transAxes,
           color=C["sub"], fontsize=7.5, va="top",
           bbox=dict(boxstyle="round,pad=0.3", facecolor=C["panel"], alpha=0.8))

ax_eq.set_xlabel("Номер сделки")
ax_eq.set_ylabel("Накопленный P&L (R)")
ax_eq.legend(facecolor=C["panel"], edgecolor=C["border"],
             labelcolor=C["text"], fontsize=8, loc="upper left")

# ── 4.2 P&L по тикерам ──────────────────────────────────
ax_pnl = fig.add_subplot(gs[1, 0])
style_ax(ax_pnl, "Total P&L (R) по тикерам")

pnl_by_t = df.groupby("ticker")["pnl_r"].sum().sort_values()
cols_bar  = [C["green"] if v >= 0 else C["red"] for v in pnl_by_t.values]
bars = ax_pnl.barh(pnl_by_t.index, pnl_by_t.values, color=cols_bar, height=0.6, edgecolor=C["border"])
ax_pnl.axvline(0, color=C["text"], linewidth=0.8)
for bar, val in zip(bars, pnl_by_t.values):
    offset = 0.15 if val >= 0 else -0.15
    ha     = "left" if val >= 0 else "right"
    ax_pnl.text(val + offset, bar.get_y() + bar.get_height()/2,
                f"{val:+.2f}R", va="center", ha=ha, color=C["text"], fontsize=7)
ax_pnl.set_xlabel("R")

# ── 4.3 Winrate по тикерам ──────────────────────────────
ax_wr = fig.add_subplot(gs[1, 1])
style_ax(ax_wr, "Winrate по тикерам")

wr_by_t = df.groupby("ticker").apply(lambda x: (x["outcome"] == "TP").mean()).sort_values()
cols_wr  = [C["green"] if v >= 0.40 else C["yellow"] if v >= 0.25 else C["red"] for v in wr_by_t]
bars3 = ax_wr.barh(wr_by_t.index, wr_by_t.values * 100, color=cols_wr, height=0.6, edgecolor=C["border"])
ax_wr.axvline(m_clean["winrate"] * 100, color=C["blue"], linewidth=1.2,
              linestyle="--", alpha=0.8, label=f"Avg {m_clean['winrate']:.0%}")
for bar, val in zip(bars3, wr_by_t.values):
    ax_wr.text(val*100 + 0.8, bar.get_y() + bar.get_height()/2,
               f"{val:.0%}", va="center", color=C["text"], fontsize=7)
ax_wr.set_xlabel("Winrate (%)")
ax_wr.legend(facecolor=C["panel"], edgecolor=C["border"], labelcolor=C["text"], fontsize=7)

# ── 4.4 Sharpe по тикерам ───────────────────────────────
ax_sh = fig.add_subplot(gs[1, 2])
style_ax(ax_sh, "Sharpe (R-based) по тикерам")

sharpe_by_t = {}
for t, sub in df.groupby("ticker"):
    ms = calc_metrics(sub)
    sharpe_by_t[t] = ms["sharpe"]
sharpe_s = pd.Series(sharpe_by_t).sort_values()
cols_sh  = [C["green"] if v >= 0 else C["red"] for v in sharpe_s.values]
ax_sh.barh(sharpe_s.index, sharpe_s.values, color=cols_sh, height=0.6, edgecolor=C["border"])
ax_sh.axvline(0, color=C["text"], linewidth=0.8)
ax_sh.axvline(m_clean["sharpe"], color=C["blue"], linewidth=1.2, linestyle="--",
              label=f"Avg {m_clean['sharpe']:.2f}")
ax_sh.set_xlabel("Sharpe")
ax_sh.legend(facecolor=C["panel"], edgecolor=C["border"], labelcolor=C["text"], fontsize=7)

# ── 4.5 Распределение P&L ───────────────────────────────
ax_dist = fig.add_subplot(gs[2, 0])
style_ax(ax_dist, "Распределение P&L (R)")

wins_r   = df[df["outcome"] == "TP"]["pnl_r"]
losses_r = df[df["outcome"] != "TP"]["pnl_r"]
bins = np.linspace(df["pnl_r"].min() - 0.3, df["pnl_r"].max() + 0.3, 22)
ax_dist.hist(losses_r, bins=bins, color=C["red"],   alpha=0.75, label=f"Loss ({len(losses_r)})")
ax_dist.hist(wins_r,   bins=bins, color=C["green"], alpha=0.75, label=f"Win  ({len(wins_r)})")
ax_dist.axvline(0, color=C["text"], linewidth=0.8)
ax_dist.axvline(m_clean["expectancy"], color=C["yellow"], linewidth=1.5, linestyle="--",
                label=f"E={m_clean['expectancy']:+.2f}R")
ax_dist.set_xlabel("P&L (R)")
ax_dist.set_ylabel("Кол-во")
ax_dist.legend(facecolor=C["panel"], edgecolor=C["border"], labelcolor=C["text"], fontsize=7)

# ── 4.6 Equity кривые по тикерам ────────────────────────
ax_tickers = fig.add_subplot(gs[2, 1:])
style_ax(ax_tickers, "Equity curve по тикерам (R нарастающим)")

palette = [C["green"], C["blue"], C["purple"], "#79c0ff", "#56d364",
           "#aff5b4", "#d2a8ff", "#ffa657", "#ff7b72", "#6e7681",
           "#e3b341", "#a5d6ff", "#7ee787", "#f0883e"]

for i, t in enumerate(sorted(df["ticker"].unique())):
    sub    = df[df["ticker"] == t].copy()
    sub_eq = sub["pnl_r"].cumsum().values
    final  = sub_eq[-1]
    color  = C["green"] if final >= 0 else C["red"]
    ax_tickers.plot(range(1, len(sub_eq) + 1), sub_eq,
                    color=palette[i % len(palette)], linewidth=1.4,
                    label=f"{t} ({final:+.1f}R)")

ax_tickers.axhline(0, color=C["border"], linewidth=0.8)
ax_tickers.set_xlabel("Номер сделки по тикеру")
ax_tickers.set_ylabel("P&L (R)")
ax_tickers.legend(facecolor=C["panel"], edgecolor=C["border"], labelcolor=C["text"],
                  fontsize=6.5, ncol=2, loc="upper left")

# ── 4.7 Сводная метрик-карточка ─────────────────────────
ax_card = fig.add_subplot(gs[3, :])
ax_card.set_facecolor(C["panel"])
ax_card.axis("off")

card_metrics = [
    ("Сделок",            f"{m_clean['n_trades']}"),
    ("Winrate",           f"{m_clean['winrate']:.0%}"),
    ("Avg Win",           f"+{m_clean['avg_win_r']:.2f}R"),
    ("Avg Loss",          f"{m_clean['avg_loss_r']:.2f}R"),
    ("Expectancy",        f"{m_clean['expectancy']:+.3f}R"),
    ("Total P&L",         f"+{m_clean['total_r']:.1f}R"),
    ("Profit Factor",     f"{m_clean['profit_factor']:.2f}"),
    ("Max Drawdown",      f"{m_clean['max_dd_r']:.1f}R"),
    ("Max streak loss",   f"{m_clean['max_streak']}"),
    ("Sharpe",            f"{m_clean['sharpe']:.2f}"),
    ("Доходность $10k",   f"+{m_clean['return_pct']:.1%}"),
    ("Итоговый капитал",  f"${m_clean['capital_final']:,.0f}"),
]

n = len(card_metrics)
for i, (lbl, val) in enumerate(card_metrics):
    x_pos = (i % 6) / 6 + 0.01
    y_pos = 0.75 if i < 6 else 0.15
    color_val = C["green2"] if any(c in val for c in ["+", "%"]) and "-" not in val else C["red"] if "-" in val else C["text"]
    ax_card.text(x_pos,       y_pos + 0.15, lbl, transform=ax_card.transAxes,
                 color=C["sub"], fontsize=8, va="center")
    ax_card.text(x_pos,       y_pos,        val, transform=ax_card.transAxes,
                 color=color_val, fontsize=13, fontweight="bold", va="center")

ax_card.add_patch(FancyBboxPatch((0, 0), 1, 1, transform=ax_card.transAxes,
                                  boxstyle="round,pad=0.01",
                                  facecolor=C["panel"], edgecolor=C["border"], linewidth=1))

fig.suptitle(
    "Финальный бэктест стратегии Price Action + Neural Net  |  14 тикеров NASDAQ  |  Hourly  |  ~2 года  |  META исключена",
    color=C["text"], fontsize=11, fontweight="bold", y=0.99)

out_path = os.path.join(OUT_DIR, "final_dashboard.png")
plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=C["bg"])
plt.close()
print(f"Финальный дашборд сохранён: {out_path}")

# ── Сравнительный график С/БЕЗ META ─────────────────────
fig2, ax = plt.subplots(figsize=(12, 5))
fig2.patch.set_facecolor(C["bg"])
style_ax(ax, "Equity Curve: с META vs без META")

eq_all   = df_all.sort_values("datetime")["pnl_r"].cumsum().values
eq_clean = df["pnl_r"].cumsum().values

ax.plot(range(1, len(eq_all)+1),   eq_all,   color=C["red"],   linewidth=1.8,
        linestyle="--", label=f"С META  ({eq_all[-1]:+.1f}R)")
ax.plot(range(1, len(eq_clean)+1), eq_clean, color=C["green"], linewidth=2.2,
        label=f"Без META ({eq_clean[-1]:+.1f}R)")
ax.fill_between(range(1, len(eq_clean)+1), eq_all[:len(eq_clean)], eq_clean,
                where=(eq_clean > eq_all[:len(eq_clean)]),
                color=C["green"], alpha=0.15, label="Разница")
ax.axhline(0, color=C["border"], linewidth=1)
ax.set_xlabel("Номер сделки", color=C["sub"])
ax.set_ylabel("Накопленный P&L (R)", color=C["sub"])
ax.legend(facecolor=C["panel"], edgecolor=C["border"], labelcolor=C["text"], fontsize=9)
fig2.patch.set_facecolor(C["bg"])

comp_path = os.path.join(OUT_DIR, "comparison_meta.png")
plt.savefig(comp_path, dpi=160, bbox_inches="tight", facecolor=C["bg"])
plt.close()
print(f"Сравнительный график сохранён: {comp_path}")

print("\n[OK] Step 10 завершён.")
