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


def render_chart(df: pd.DataFrame, out_path: str, level=None, entry=None,
                  stop=None, target=None, title: str = "") -> None:
    hlines, colors, styles, widths = [], [], [], []

    def add_line(value, color, style="--", width=1.0):
        if value is not None:
            hlines.append(float(value))
            colors.append(color)
            styles.append(style)
            widths.append(width)

    add_line(level,  "blue",   "-",  1.2)
    add_line(entry,  "black",  "--", 1.0)
    add_line(stop,   "red",    "--", 1.0)
    add_line(target, "green",  "--", 1.0)

    hlines_kw = None
    if hlines:
        hlines_kw = dict(hlines=hlines, colors=colors, linestyle=styles, linewidths=widths)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    mpf.plot(
        df,
        type="candle",
        style="yahoo",
        volume=True,
        hlines=hlines_kw,
        title=title,
        figsize=(6.4, 4.8),   # 6.4in * 100dpi = 640px, 4.8in * 100dpi = 480px
        savefig=dict(fname=out_path, dpi=100, pad_inches=0, bbox_inches=None),
        axisoff=False,
        tight_layout=True,
    )


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
    ap.add_argument("--bars-before", type=int, default=60, help="Баров контекста до сигнала")
    ap.add_argument("--bars-after", type=int, default=10, help="Баров контекста после сигнала")
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
        window = slice_window(df, None, bars_before=60, bars_after=0)
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
                 stop=args.stop, target=args.target,
                 title=f"{args.ticker.upper()} {args.timeframe} — {args.direction}")

    print(f"Сохранено: {out_path}")


if __name__ == "__main__":
    main()
