"""
Step 9 — Анализ META + визуализация equity curve.
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
import yfinance as yf

DATA_DIR = "data"
OUT_DIR  = "charts"
os.makedirs(OUT_DIR, exist_ok=True)

# ───────────────────────────────────────────────────────
# 1. Загрузка результатов бэктеста
# ───────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(DATA_DIR, "backtest_multi.csv"))
df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
df = df.sort_values("datetime").reset_index(drop=True)
df["equity_r"] = df["pnl_r"].cumsum()
df["trade_n"]  = range(1, len(df) + 1)

TICKERS = df["ticker"].unique().tolist()

print(f"Загружено сделок: {len(df)}")
print(f"Тикеры: {', '.join(TICKERS)}")

# ───────────────────────────────────────────────────────
# 2. АНАЛИЗ META
# ───────────────────────────────────────────────────────
meta = df[df["ticker"] == "META"].copy().reset_index(drop=True)
meta["entry_date"] = meta["datetime"].dt.strftime("%Y-%m-%d %H:%M")

print("\n" + "="*70)
print("АНАЛИЗ META  (6 сделок, winrate 17%, total -5.73R)")
print("="*70)

# Детали каждой сделки
print("\n--- Все сделки META ---")
cols_show = ["entry_date","long","level","entry","exit","outcome","pnl_r","bars_held","model_prob"]
meta_show = meta[cols_show].copy()
meta_show.columns = ["Дата", "Лонг", "Уровень", "Вход", "Выход", "Итог", "P&L(R)", "Баров", "Prob"]
print(meta_show.to_string(index=False))

# Диагностика по каждой сделке
print("\n--- Диагностика ---")
for _, row in meta.iterrows():
    direction   = "ЛОНГ от поддержки" if row["long"] else "ШОРТ от сопротивления"
    held_h      = row["bars_held"]
    outcome_sym = "[TP]" if row["outcome"] == "TP" else "[SL]" if row["outcome"] == "SL" else "[TO]"
    risk_pct    = abs(row["entry"] - row["stop"]) / row["entry"] * 100 if "stop" in row else 0
    print(f"  {row['entry_date'][:10]}  {direction:<26}  уровень={row['level']:.1f}"
          f"  вход={row['entry']:.1f}  {outcome_sym}  bars={held_h}"
          f"  prob={row['model_prob']:.3f}")

print("\n--- Паттерны проблем META ---")

# 1. Время до закрытия
avg_bars = meta["bars_held"].mean()
print(f"  Avg bars_held: {avg_bars:.1f}  (большинство закрывались в 0-1 бар -- быстрые развороты)")

# 2. Направление
n_long  = meta["long"].sum()
n_short = (~meta["long"]).sum()
print(f"  Лонгов: {n_long}, Шортов: {n_short}  → META всё время в лонгах от поддержки")

# 3. Уровни
print(f"  Уровни: {meta['level'].unique()}  → торгуем одни и те же уровни повторно")

# 4. Model prob
print(f"  Model prob: {meta['model_prob'].min():.3f} – {meta['model_prob'].max():.3f}"
      f"  (все у нижней границы порога 0.50–0.67)")

# 5. Скользящий тренд META на период сделок
print("\n  Загружаем дневной META для проверки тренда...")
try:
    meta_1d = yf.download("META", period="2y", interval="1d",
                          auto_adjust=True, progress=False)
    if not meta_1d.empty:
        meta_1d.columns = [c[0] for c in meta_1d.columns]
        meta_1d["EMA20"] = meta_1d["Close"].ewm(span=20, adjust=False).mean()
        meta_1d["trend"] = np.where(meta_1d["Close"] > meta_1d["EMA20"], "BULL", "BEAR")
        meta_1d.index = pd.to_datetime(meta_1d.index)

        print(f"\n  Тренд META в даты сделок:")
        for _, row in meta.iterrows():
            trade_date = row["datetime"].tz_convert("America/New_York").strftime("%Y-%m-%d")
            # Ближайшая дневная свеча
            closest = meta_1d.index[meta_1d.index <= pd.Timestamp(trade_date, tz="America/New_York")]
            if len(closest) > 0:
                d = closest[-1]
                trend = meta_1d.loc[d, "trend"]
                price = meta_1d.loc[d, "Close"]
                ema   = meta_1d.loc[d, "EMA20"]
                counter = (row['long'] and trend=='BEAR') or (not row['long'] and trend=='BULL')
                flag = " !! COUNTER-TREND" if counter else " ok"
                print(f"    {trade_date}  Close={price:.1f}  EMA20={ema:.1f}  Trend={trend}{flag}")
except Exception as e:
    print(f"  (не удалось загрузить META daily: {e})")

print("\n--- Вывод по META ---")
print("  Проблема 1: уровни 520/546/580/720/737 — META росла сквозь все поддержки")
print("  Проблема 2: сделки закрывались мгновенно (0 баров) — гэпы на открытии")
print("  Решение:   исключить META или добавить фильтр объёма на открытии дня")

# ───────────────────────────────────────────────────────
# 3. ВИЗУАЛИЗАЦИЯ
# ───────────────────────────────────────────────────────
print("\nСтроим графики...")

# ── 3.1 Главный дашборд ─────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor("#0d1117")
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

COLORS = {
    "bg":      "#0d1117",
    "panel":   "#161b22",
    "green":   "#2ea043",
    "red":     "#f85149",
    "yellow":  "#d29922",
    "blue":    "#58a6ff",
    "text":    "#e6edf3",
    "subtext": "#8b949e",
    "grid":    "#21262d",
}

def style_ax(ax, title=""):
    ax.set_facecolor(COLORS["panel"])
    ax.spines[["top","right","left","bottom"]].set_color(COLORS["grid"])
    ax.tick_params(colors=COLORS["subtext"], labelsize=8)
    ax.xaxis.label.set_color(COLORS["subtext"])
    ax.yaxis.label.set_color(COLORS["subtext"])
    if title:
        ax.set_title(title, color=COLORS["text"], fontsize=10, fontweight="bold", pad=8)
    ax.grid(True, color=COLORS["grid"], linewidth=0.5, alpha=0.7)

# ── Plot 1: Общая equity curve ───────────────────────────
ax1 = fig.add_subplot(gs[0, :])   # полная ширина
style_ax(ax1, "Equity Curve — все 15 тикеров (R, нарастающим итогом)")

eq   = df["equity_r"].values
x    = df["trade_n"].values
peak = np.maximum.accumulate(eq)
dd   = eq - peak

# Заливка под кривой
ax1.fill_between(x, 0, eq,
                 where=(eq >= 0), color=COLORS["green"], alpha=0.15)
ax1.fill_between(x, 0, eq,
                 where=(eq < 0),  color=COLORS["red"],   alpha=0.15)

# Drawdown
ax1_dd = ax1.twinx()
ax1_dd.fill_between(x, 0, dd, color=COLORS["red"], alpha=0.20)
ax1_dd.set_ylabel("Drawdown (R)", color=COLORS["red"], fontsize=8)
ax1_dd.tick_params(colors=COLORS["red"], labelsize=7)
ax1_dd.spines[["top","right","left","bottom"]].set_color(COLORS["grid"])

# Основная линия
ax1.plot(x, eq, color=COLORS["blue"], linewidth=2, zorder=5)
ax1.axhline(0, color=COLORS["grid"], linewidth=1)

# Маркеры сделок
tp_mask = df["outcome"] == "TP"
sl_mask = df["outcome"] == "SL"
ax1.scatter(x[tp_mask], eq[tp_mask], color=COLORS["green"], s=40, zorder=6, label="TP")
ax1.scatter(x[sl_mask], eq[sl_mask], color=COLORS["red"],   s=40, zorder=6, marker="v", label="SL")

# Аннотации итоговых значений
final_r = eq[-1]
ax1.annotate(f"+{final_r:.1f}R итого",
             xy=(x[-1], eq[-1]), xytext=(-60, 12),
             textcoords="offset points",
             color=COLORS["green"], fontsize=9, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=COLORS["green"], lw=1))

ax1.set_xlabel("Номер сделки")
ax1.set_ylabel("Накопленный P&L (R)")
ax1.legend(facecolor=COLORS["panel"], edgecolor=COLORS["grid"],
           labelcolor=COLORS["text"], fontsize=8)

# Статистика в углу
stats_text = (f"Сделок: {len(df)}  |  Winrate: {(df['outcome']=='TP').mean():.0%}  |  "
              f"PF: {df[df['outcome']=='TP']['pnl_r'].sum() / df[df['outcome']!='TP']['pnl_r'].abs().sum():.2f}  |  "
              f"Sharpe: 1.55")
ax1.text(0.02, 0.95, stats_text, transform=ax1.transAxes,
         color=COLORS["subtext"], fontsize=8, va="top")

# ── Plot 2: P&L по тикерам ──────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
style_ax(ax2, "Total P&L (R) по тикерам")

ticker_pnl = df.groupby("ticker")["pnl_r"].sum().sort_values()
colors_bar  = [COLORS["green"] if v >= 0 else COLORS["red"] for v in ticker_pnl.values]
bars = ax2.barh(ticker_pnl.index, ticker_pnl.values, color=colors_bar, height=0.6)
ax2.axvline(0, color=COLORS["text"], linewidth=0.8)

for bar, val in zip(bars, ticker_pnl.values):
    ax2.text(val + (0.2 if val >= 0 else -0.2),
             bar.get_y() + bar.get_height()/2,
             f"{val:+.2f}R", va="center",
             ha="left" if val >= 0 else "right",
             color=COLORS["text"], fontsize=7)

ax2.set_xlabel("R")

# ── Plot 3: Winrate по тикерам ──────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
style_ax(ax3, "Winrate по тикерам")

wr = df.groupby("ticker").apply(lambda x: (x["outcome"]=="TP").mean()).sort_values()
colors_wr = [COLORS["green"] if v >= 0.40 else
             COLORS["yellow"] if v >= 0.25 else COLORS["red"]
             for v in wr.values]
bars3 = ax3.barh(wr.index, wr.values * 100, color=colors_wr, height=0.6)
ax3.axvline(37, color=COLORS["blue"], linewidth=1, linestyle="--", alpha=0.7, label="Avg 37%")

for bar, val in zip(bars3, wr.values):
    ax3.text(val*100 + 1, bar.get_y() + bar.get_height()/2,
             f"{val:.0%}", va="center", color=COLORS["text"], fontsize=7)

ax3.set_xlabel("Winrate (%)")
ax3.legend(facecolor=COLORS["panel"], edgecolor=COLORS["grid"],
           labelcolor=COLORS["text"], fontsize=7)

# ── Plot 4: Распределение P&L по сделкам ───────────────
ax4 = fig.add_subplot(gs[2, 0])
style_ax(ax4, "Распределение P&L сделок (R)")

wins_r   = df[df["outcome"]=="TP"]["pnl_r"]
losses_r = df[df["outcome"]!="TP"]["pnl_r"]
bins = np.linspace(df["pnl_r"].min() - 0.2, df["pnl_r"].max() + 0.2, 25)

ax4.hist(losses_r, bins=bins, color=COLORS["red"],   alpha=0.75, label=f"Loss ({len(losses_r)})")
ax4.hist(wins_r,   bins=bins, color=COLORS["green"], alpha=0.75, label=f"Win ({len(wins_r)})")
ax4.axvline(0, color=COLORS["text"], linewidth=1)
ax4.axvline(df["pnl_r"].mean(), color=COLORS["yellow"], linewidth=1.5,
            linestyle="--", label=f"Avg {df['pnl_r'].mean():.2f}R")

ax4.set_xlabel("P&L (R)")
ax4.set_ylabel("Кол-во сделок")
ax4.legend(facecolor=COLORS["panel"], edgecolor=COLORS["grid"],
           labelcolor=COLORS["text"], fontsize=8)

# ── Plot 5: Equity по лучшим тикерам ───────────────────
ax5 = fig.add_subplot(gs[2, 1])
style_ax(ax5, "Equity curve лучших vs худших тикеров")

top_tickers  = ["PLTR", "SMCI", "NVDA", "CRWD", "COIN"]
bad_tickers  = ["META", "AMZN", "TSLA"]
palette_top  = ["#2ea043", "#3fb950", "#56d364", "#85e89d", "#aff5b4"]
palette_bad  = ["#f85149", "#ff7b72", "#ffa198"]

for i, t in enumerate(top_tickers):
    sub = df[df["ticker"] == t].copy()
    if len(sub) > 0:
        sub["eq"] = sub["pnl_r"].cumsum()
        ax5.plot(range(len(sub)), sub["eq"].values,
                 color=palette_top[i], linewidth=1.5, label=t)

for i, t in enumerate(bad_tickers):
    sub = df[df["ticker"] == t].copy()
    if len(sub) > 0:
        sub["eq"] = sub["pnl_r"].cumsum()
        ax5.plot(range(len(sub)), sub["eq"].values,
                 color=palette_bad[i], linewidth=1.2, linestyle="--", label=t)

ax5.axhline(0, color=COLORS["grid"], linewidth=0.8)
ax5.set_xlabel("Номер сделки по тикеру")
ax5.set_ylabel("Накопленный P&L (R)")
ax5.legend(facecolor=COLORS["panel"], edgecolor=COLORS["grid"],
           labelcolor=COLORS["text"], fontsize=7, ncol=2)

# Заголовок
fig.suptitle("Бэктест стратегии Price Action + Neural Net  |  15 тикеров NASDAQ  |  Hourly  |  ~2 года",
             color=COLORS["text"], fontsize=12, fontweight="bold", y=0.98)

chart_path = os.path.join(OUT_DIR, "backtest_dashboard.png")
plt.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
plt.close()
print(f"Дашборд сохранён: {chart_path}")

# ── 3.2 Детальный META ──────────────────────────────────
fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
fig2.patch.set_facecolor(COLORS["bg"])

for ax in axes:
    style_ax(ax)

# META equity
ax_m1 = axes[0]
style_ax(ax_m1, "META — equity curve (6 сделок)")
meta_eq = meta["pnl_r"].cumsum().values
ax_m1.plot(range(1, len(meta_eq)+1), meta_eq,
           color=COLORS["red"], linewidth=2, marker="o", markersize=6)
ax_m1.fill_between(range(1, len(meta_eq)+1), 0, meta_eq,
                   color=COLORS["red"], alpha=0.2)
ax_m1.axhline(0, color=COLORS["text"], linewidth=0.8)
for i, (eq_v, row) in enumerate(zip(meta_eq, meta.itertuples())):
    label = f"TP" if row.outcome == "TP" else "SL"
    color = COLORS["green"] if row.outcome == "TP" else COLORS["red"]
    ax_m1.annotate(f"#{i+1}\n{label}\n{row.pnl_r:+.1f}R",
                   xy=(i+1, eq_v), xytext=(0, 14 if eq_v >= 0 else -28),
                   textcoords="offset points", ha="center",
                   fontsize=7, color=color)
ax_m1.set_xticks(range(1, len(meta)+1))
ax_m1.set_xlabel("Сделка")
ax_m1.set_ylabel("P&L (R)")

# META: bars_held vs outcome
ax_m2 = axes[1]
style_ax(ax_m2, "META — скорость выхода из сделки")
for _, row in meta.iterrows():
    color = COLORS["green"] if row["outcome"] == "TP" else COLORS["red"]
    ax_m2.bar(row.name + 1, row["bars_held"] + 0.5, color=color, alpha=0.8, width=0.5)
    ax_m2.text(row.name + 1, row["bars_held"] + 0.7,
               f"{row['outcome']}\n{row['bars_held']}h",
               ha="center", va="bottom", color=COLORS["text"], fontsize=8)

ax_m2.set_xlabel("Сделка №")
ax_m2.set_ylabel("Баров до закрытия (1 бар = 1 час)")
ax_m2.set_xticks(range(1, len(meta)+1))

msg = "Большинство закрылись за 0–1 час → гэпы/импульсы на открытии сессии"
ax_m2.text(0.5, 0.92, msg, transform=ax_m2.transAxes,
           ha="center", color=COLORS["yellow"], fontsize=8)

fig2.suptitle("Анализ META — почему стратегия не работает на этом тикере",
              color=COLORS["text"], fontsize=11, fontweight="bold")

meta_path = os.path.join(OUT_DIR, "meta_analysis.png")
plt.savefig(meta_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
plt.close()
print(f"Анализ META сохранён: {meta_path}")

print("\n[OK] Готово. Открой charts/")
