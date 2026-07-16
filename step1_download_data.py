import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

TICKER = "MSTR"
DATA_DIR = "data"

os.makedirs(DATA_DIR, exist_ok=True)

print(f"Загрузка данных для {TICKER}...")

# Дневные свечи за 2 года
print("\n[1/2] Дневные свечи (2 года)...")
daily = yf.download(TICKER, period="2y", interval="1d", auto_adjust=True, progress=False)

if daily.empty:
    print("ОШИБКА: дневные данные не загружены")
    exit(1)

daily.index = pd.to_datetime(daily.index)
daily.to_csv(os.path.join(DATA_DIR, "mstr_daily.csv"))
print(f"  Сохранено: {len(daily)} баров ({daily.index[0].date()} — {daily.index[-1].date()})")
print(f"  Файл: {DATA_DIR}/mstr_daily.csv")

# 5м свечи за 60 дней (лимит yfinance)
print("\n[2/2] 5-минутные свечи (60 дней)...")
end_date = datetime.now()
start_date = end_date - timedelta(days=59)

intraday = yf.download(
    TICKER,
    start=start_date.strftime("%Y-%m-%d"),
    end=end_date.strftime("%Y-%m-%d"),
    interval="5m",
    auto_adjust=True,
    progress=False,
)

if intraday.empty:
    print("ОШИБКА: 5м данные не загружены")
    exit(1)

intraday.index = pd.to_datetime(intraday.index)
intraday.to_csv(os.path.join(DATA_DIR, "mstr_5m.csv"))
print(f"  Сохранено: {len(intraday)} баров ({intraday.index[0]} — {intraday.index[-1]})")
print(f"  Файл: {DATA_DIR}/mstr_5m.csv")

# Превью данных
print("\n" + "="*60)
print("ДНЕВНЫЕ СВЕЧИ — первые 5 строк:")
print("="*60)
print(daily.head().to_string())

print("\n" + "="*60)
print("5М СВЕЧИ — первые 5 строк:")
print("="*60)
print(intraday.head().to_string())

print("\n[OK] Данные загружены успешно.")
