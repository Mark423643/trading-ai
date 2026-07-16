# Trading AI — CLAUDE.md

## Что это

Алготрейдинговая система для российского рынка MOEX.
Стратегия: торговля от уровней поддержки/сопротивления на H1 с ML-фильтром (MLP sklearn).
Брокер: Финам (finam-trade-api 4.3.2, REST).
Уведомления: ntfy.sh через `notify_ntfy.py`.
Деплой: VPS Linux (157.22.185.89), cron каждый час в торговые часы MOEX.

## Файлы — кто за что отвечает

```
config_trading.py          — ЕДИНСТВЕННЫЙ источник всех параметров стратегии
step1_download_data.py     — загрузка OHLCV данных с MOEX ISS
step2_find_levels.py       — поиск уровней разворота (pivot points)
step3_train_model.py       — обучение MLP-классификатора
step5_moex_dataset.py      — формирование датасета по MOEX тикерам
step6_backtest.py          — базовый бэктест
step7_hourly_backtest.py   — бэктест на часовых барах
step8_multi_backtest.py    — мультитикерный бэктест
step9_analysis.py          — анализ результатов бэктеста
step10_final_backtest.py   — финальный анализ
step_portfolio_final.py    — батч-бэктест по всем 30 тикерам (основной)
step11b_paper_trading.py   — живой/бумажный сканер (запуск по cron H1)
broker_finam.py            — интеграция с брокером Финам
notify_ntfy.py             — push-уведомления через ntfy.sh
analytics.py               — аналитика сделок
```

## VPS подключение

```bash
python .claude/vps.py                    # статус бота + последние логи
python .claude/vps.py "crontab -l"
python .claude/vps.py "tail -100 /root/trading/logs/trading.log"
python .claude/vps.py "ps aux | grep python"
```

Данные подключения в `.env`: `VPS_HOST`, `VPS_USER`, `VPS_PASSWORD`.
Торговые файлы на VPS: `/root/trading/`.

## Критически важные правила

**1. Параметры — только в config_trading.py.**
При изменении любого параметра стратегии — меняй только этот файл.

**2. LIVE_TRADING управляется только через .env.**
Никогда не хардкоди `LIVE_TRADING = True` в коде.
Дефолт в `broker_finam.py` должен оставаться `False`.

**3. Текущий режим — агрессивный (с 10.07.2026):**
- `ATR_EXHAUSTION = 0.30`
- `LEVEL_COOLDOWN_BARS = 1`
- `MODEL_THRESHOLD = 0.0` (ML отключён, только геометрия)
- History cooldown закрытых позиций = 1 день (в `step11b_paper_trading.py`)
- Не "исправляй" эти значения — они выставлены намеренно для 25+ сделок/мес.

**4. Не создавай диагностические скрипты в `.claude/`** — используй `.claude/vps.py`.

## Технический долг

- History cooldown (1 день) захардкожен в `step11b_paper_trading.py`, а не в `config_trading.py`
- Нет unit-тестов; критичные функции: `false_breakout()`, `slow_approach()`, расчёт стопа/таргета
- `_Tee` класс в `step11b_paper_trading.py` не закрывает лог при краше — кандидат на замену стандартным `logging`

## Переменные окружения (.env)

```
FINAM_TOKEN=...
FINAM_CLIENT_ID=...
FINAM_RISK_PCT=0.01
LIVE_TRADING=0

TWELVE_DATA_API_KEY=...

VPS_HOST=157.22.185.89
VPS_USER=root
VPS_PASSWORD=...
```

## Стиль кода

- Комментарии на русском
- Без type hints
- PEP8, выравнивание как в `config_trading.py`
- Не добавляй зависимости без явной просьбы
- `finam-trade-api==4.3.2` — критична точная версия, не обновляй
