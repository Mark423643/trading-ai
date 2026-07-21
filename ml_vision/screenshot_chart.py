"""
screenshot_chart.py — рендерит график сделки в PNG для разметки Vision-модели
по паттернам Тимура.

Источники данных:
  M5  -> data/m5_cache/{ticker}_m5.pkl   (pandas, 6 мес., резэмплировано из ISS 1m)
  H1  -> data/bt_cache/{TICKER}_1h.csv
  D1  -> data/bt_cache/{TICKER}_1d.csv

Пример:
  python3 screenshot_chart.py --ticker GAZP --timeframe M5 \\
      --time "2026-07-10 14:35:00" --level 168.20 \\
      --direction LONG_ENTRY --entry 168.35 --stop 167.10 --target 172.10

Смоук-тест (без реального сигнала, для проверки пайплайна):
  python3 screenshot_chart.py --test
"""
import argparse
import os
import sys

import pandas as pd
import mplfinance as mpf

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
AI_DIR     = os.path.dirname(BASE_DIR)
M5_CACHE   = os.path.join(AI_DIR, "data", "m5_cache")
BT_CACHE   = os.path.join(AI_DIR, "data", "bt_cache")
DATA_DIR   = os.path.join(BASE_DIR, "data")

VALID_DIRECTIONS = ("LONG_ENTRY", "SHORT_ENTRY", "NO_ENTRY")


def load_ticker_df(ticker: str, timeframe: str) -> pd.DataFrame:
    tf = timeframe.upper()
    if tf == "M5":
        path = os.path.join(M5_CACHE, f"{ticker.upper()}_m5.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Нет кэша M5 для {ticker}: {path}")
        df = pd.read_pickle(path)
    elif tf == "H1":
        path = os.path.join(BT_CACHE, f"{ticker.upper()}_1h.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Нет кэша H1 для {ticker}: {path}")
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    elif tf == "D1":
        path = os.path.join(BT_CACHE, f"{ticker.upper()}_1d.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Нет кэша D1 для {ticker}: {path}")
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        raise ValueError(f"Неизвестный таймфрейм: {timeframe} (ожидается M5/H1/D1)")

    df = df.sort_index()
    return df[["Open", "High", "Low", "Close", "Volume"]].astype(float).dropna()


def slice_window(df: pd.DataFrame, signal_time, bars_before: int, bars_after: int) -> pd.DataFrame:
    if signal_time is None:
        end_idx = len(df) - 1
    else:
        ts = pd.Timestamp(signal_time)
        pos = df.index.searchsorted(ts, side="right") - 1
        end_idx = max(0, min(pos, len(df) - 1))
    start_idx = max(0, end_idx - bars_before)
    stop_idx = min(len(df), end_idx + bars_after + 1)
    return df.iloc[start_idx:stop_idx]


# ── Тёмная тема в стиле TradingView Dark ──
_TV_DARK = mpf.make_mpf_style(
    base_mpf_style="nightclouds",
    marketcolors=mpf.make_marketcolors(
        up="#26a69a", down="#ef5350",
        edge="inherit", wick="inherit",
    ),
    facecolor="#131722",
    edgecolor="#131722",
    figcolor="#131722",
    gridcolor="#2a2e39",
    gridstyle="--",
    rc={
        "axes.labelcolor": "#d1d4dc",
        "xtick.color": "#d1d4dc",
        "ytick.color": "#d1d4dc",
        "text.color": "#d1d4dc",
        "axes.edgecolor": "#2a2e39",
    },
)


def render_chart(df: pd.DataFrame, out_path: str, level=None, entry=None,
                  stop=None, target=None, title: str = "",
                  entry_bar_idx: int = -1, direction: str = None) -> None:
    """Рендерит чистый M5-график в стиле TradingView Dark: без объёма и
    прочих индикаторов, уровень/стоп/цель — линии, текущая цена подписана
    справа, точка входа — синяя стрелка (если задан entry).

    entry_bar_idx — индекс бара (по умолчанию последний, -1), к которому
    рисуется стрелка входа.
    direction — "LONG"/"SHORT"/"LONG_ENTRY"/"SHORT_ENTRY": влияет только на
    направление стрелки (снизу-вверх для лонга, сверху-вниз для шорта).
    """
    hlines, colors, styles, widths = [], [], [], []

    def add_line(value, color, style="--", width=1.2):
        if value is not None:
            hlines.append(float(value))
            colors.append(color)
            styles.append(style)
            widths.append(width)

    # Уровень — жирная сплошная линия (главный ориентир на графике)
    add_line(level,  "#ffeb3b", "-",  2.2)
    add_line(stop,   "#ef5350", "--", 1.4)
    add_line(target, "#26a69a", "--", 1.4)

    hlines_kw = None
    if hlines:
        hlines_kw = dict(hlines=hlines, colors=colors, linestyle=styles, linewidths=widths)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, axlist = mpf.plot(
        df,
        type="candle",
        style=_TV_DARK,
        volume=False,          # чистый график — без объёма и прочих индикаторов
        hlines=hlines_kw,
        title=title,
        figsize=(6.4, 4.8),    # 6.4in * 100dpi = 640px, 4.8in * 100dpi = 480px
        axisoff=False,
        tight_layout=True,
        returnfig=True,
    )
    ax = axlist[0]

    # ── Расширяем ось Y так, чтобы уровень/стоп/цель гарантированно попали
    #    в кадр, даже если они за пределами диапазона видимых свечей ──
    y_vals = [v for v in (level, stop, target) if v is not None]
    if y_vals:
        y0, y1 = ax.get_ylim()
        y0 = min(y0, min(y_vals))
        y1 = max(y1, max(y_vals))
        pad = (y1 - y0) * 0.05 or 0.01
        ax.set_ylim(y0 - pad, y1 + pad)

    # ── Точка входа — синяя стрелка ──
    if entry is not None:
        n = len(df)
        x_idx = entry_bar_idx if entry_bar_idx >= 0 else n + entry_bar_idx
        x_idx = max(0, min(n - 1, x_idx))
        is_long = str(direction).upper().startswith("LONG")
        # Диапазон цен на графике — чтобы стрелка была заметного, но не гигантского размера
        y_span = float(df["High"].max() - df["Low"].min()) or 1.0
        offset = y_span * 0.06
        if is_long:
            y_from = float(entry) - offset * 2.2
            y_to = float(entry) - offset * 0.3
        else:
            y_from = float(entry) + offset * 2.2
            y_to = float(entry) + offset * 0.3
        ax.annotate(
            "", xy=(x_idx, y_to), xytext=(x_idx, y_from),
            xycoords="data", textcoords="data",
            arrowprops=dict(arrowstyle="-|>", color="#2962ff", lw=2.2,
                             mutation_scale=16),
            zorder=10,
        )

    # ── Текущая цена — подпись справа, как в TradingView ──
    last_close = float(df["Close"].iloc[-1])
    x0, x1 = ax.get_xlim()
    ax.set_xlim(x0, x1 + (x1 - x0) * 0.06)  # место под подпись цены справа
    ax.annotate(
        f" {last_close:.2f} ",
        xy=(x1, last_close), xycoords="data",
        va="center", ha="left", fontsize=8, color="#131722",
        bbox=dict(boxstyle="square,pad=0.25", facecolor="#2962ff", edgecolor="none"),
    )

    fig.savefig(out_path, dpi=100, facecolor=_TV_DARK["figcolor"], pad_inches=0)
    import matplotlib.pyplot as plt
    plt.close(fig)


def make_screenshot(ticker: str, bar_time, level: float, entry: float,
                     stop: float, target: float, direction: str,
                     out_dir: str = None, timeframe: str = "H1",
                     bars_before: int = 99, df: "pd.DataFrame | None" = None) -> str:
    """Готовит PNG-скриншот сигнала для отправки в NTFY: 100 баров до
    bar_time включительно, уровень (жёлтый), стоп (красный), цель (зелёный),
    вход (синяя стрелка). Возвращает путь к сохранённому файлу.

    df — если передан, используется напрямую (свежие данные из step11b),
    иначе загружается из файлового кэша.
    """
    if df is None:
        df = load_ticker_df(ticker, timeframe)
    window = slice_window(df, bar_time, bars_before=bars_before, bars_after=0)
    if len(window) == 0:
        raise ValueError(f"Пустое окно данных для {ticker} @ {bar_time}")

    if out_dir is None:
        out_dir = os.path.join(DATA_DIR, "signals")
    os.makedirs(out_dir, exist_ok=True)

    ts_str = pd.Timestamp(bar_time).strftime("%Y%m%d_%H%M%S") if bar_time else \
        pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{ticker.upper()}_{ts_str}_{direction}.png"
    out_path = os.path.join(out_dir, fname)

    render_chart(
        window, out_path,
        level=level, stop=stop, target=target, entry=entry,
        direction=direction,
        title=f"{ticker.upper()} {timeframe.upper()} — {direction}",
    )
    return out_path


def build_filename(ticker: str, signal_time, direction: str) -> str:
    if signal_time is None:
        ts_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    else:
        ts_str = pd.Timestamp(signal_time).strftime("%Y%m%d_%H%M%S")
    return f"{ticker.upper()}_{ts_str}_{direction}.png"


def main():
    ap = argparse.ArgumentParser(description="Рендер графика сделки для разметки Vision-модели")
    ap.add_argument("--ticker", type=str, help="Тикер, напр. GAZP")
    ap.add_argument("--timeframe", type=str, default="M5", choices=["M5", "H1", "D1"])
    ap.add_argument("--time", type=str, default=None,
                     help="Время сигнала 'YYYY-MM-DD HH:MM:SS' (по умолчанию — последний бар)")
    ap.add_argument("--level", type=float, default=None, help="Уровень (горизонтальная линия)")
    ap.add_argument("--entry", type=float, default=None)
    ap.add_argument("--stop", type=float, default=None)
    ap.add_argument("--target", type=float, default=None)
    ap.add_argument("--direction", type=str, default="NO_ENTRY", choices=VALID_DIRECTIONS)
    ap.add_argument("--split", type=str, default="train", choices=["train", "val"])
    ap.add_argument("--bars-before", type=int, default=99, help="Баров контекста до сигнала (99+сигнальный=100 на графике)")
    ap.add_argument("--bars-after", type=int, default=0, help="Баров контекста после сигнала")
    ap.add_argument("--out", type=str, default=None, help="Явный путь для сохранения (переопределяет --split)")
    ap.add_argument("--test", action="store_true", help="Смоук-тест на кэшированных данных, без реального сигнала")
    args = ap.parse_args()

    if args.test:
        print("[TEST] Смоук-тест screenshot_chart.py...")
        ticker = "GAZP"
        try:
            df = load_ticker_df(ticker, "M5")
        except FileNotFoundError as e:
            print(f"[TEST] ОШИБКА: {e}")
            sys.exit(1)
        window = slice_window(df, None, bars_before=99, bars_after=0)
        level = float(window["High"].max())
        out_dir = os.path.join(DATA_DIR, "val", "NO_ENTRY")
        fname = build_filename(ticker, window.index[-1], "NO_ENTRY")
        out_path = os.path.join(out_dir, fname)
        render_chart(window, out_path, level=level,
                     title=f"[TEST] {ticker} M5 — смоук-тест")
        ok = os.path.exists(out_path) and os.path.getsize(out_path) > 0
        print(f"[TEST] Баров в окне: {len(window)}")
        print(f"[TEST] Файл: {out_path}")
        print(f"[TEST] Размер: {os.path.getsize(out_path) if ok else 0} байт")
        print(f"[TEST] {'OK' if ok else 'ОШИБКА — файл не создан'}")
        sys.exit(0 if ok else 1)

    if not args.ticker:
        ap.error("--ticker обязателен (или используйте --test)")

    df = load_ticker_df(args.ticker, args.timeframe)
    window = slice_window(df, args.time, args.bars_before, args.bars_after)
    if len(window) == 0:
        print("ОШИБКА: пустое окно данных — проверьте --time")
        sys.exit(1)

    if args.out:
        out_path = args.out
    else:
        fname = build_filename(args.ticker, args.time, args.direction)
        out_path = os.path.join(DATA_DIR, args.split, args.direction, fname)

    render_chart(window, out_path, level=args.level, entry=args.entry,
                 stop=args.stop, target=args.target, direction=args.direction,
                 title=f"{args.ticker.upper()} {args.timeframe} — {args.direction}")

    print(f"Сохранено: {out_path}")


if __name__ == "__main__":
    main()
