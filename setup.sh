#!/bin/bash
# =============================================================
#  setup.sh — Установка торгового бота на чистый Ubuntu сервер
#  Использование: bash setup.sh
# =============================================================
set -e

# ── определяем пути ──────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
VENV_DIR="$PROJECT_DIR/venv"
PYTHON="$VENV_DIR/bin/python"
LOG_FILE="$PROJECT_DIR/logs/trading.log"

echo "============================================================"
echo "  TRADING BOT SETUP"
echo "  Проект:  $PROJECT_DIR"
echo "============================================================"

# ── [1/5] Обновление системы и Python ────────────────────────
echo ""
echo "[1/5] Обновление системы..."
sudo apt-get update -qq
sudo apt-get install -y python3 python3-pip python3-venv python3-dev build-essential

# Проверяем версию Python (нужна 3.10+)
PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "  Python $PY_VER найден."

# ── [2/5] Создание директорий проекта ────────────────────────
echo ""
echo "[2/5] Создание директорий..."
mkdir -p "$PROJECT_DIR/data/tickers"
mkdir -p "$PROJECT_DIR/models"
mkdir -p "$PROJECT_DIR/charts"
mkdir -p "$PROJECT_DIR/logs"
echo "  data/, models/, charts/, logs/, data/tickers/ — OK"

# ── [3/5] Виртуальное окружение ──────────────────────────────
echo ""
echo "[3/5] Настройка виртуального окружения..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "  Venv создан: $VENV_DIR"
else
    echo "  Venv уже существует: $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip -q

# ── [4/5] Установка зависимостей ─────────────────────────────
echo ""
echo "[4/5] Установка Python-зависимостей..."
pip install -r "$PROJECT_DIR/requirements.txt"
echo "  Все библиотеки установлены."

# ── [5/5] Настройка cron ─────────────────────────────────────
echo ""
echo "[5/5] Настройка cron job..."

# Запуск каждый час с 7:00 до 20:00 UTC, только Пн-Пт
# 7:00-15:40 UTC = торги MOEX (10:00-18:40 МСК)
# 13:30-20:00 UTC = торги NASDAQ (9:30-16:00 ET)
CRON_LINE="0 7-20 * * 1-5 cd $PROJECT_DIR && $PYTHON $PROJECT_DIR/step11b_paper_trading.py > /dev/null 2>&1"

# Проверяем, не настроен ли уже cron
if crontab -l 2>/dev/null | grep -q "step11b_paper_trading"; then
    echo "  Cron job уже настроен — пропускаем."
else
    ( crontab -l 2>/dev/null; echo "$CRON_LINE" ) | crontab -
    echo "  Cron job добавлен."
fi

echo "  Расписание: 7:00-20:00 UTC, Пн-Пт (покрывает MOEX и NASDAQ)"

# ── Итог ─────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  УСТАНОВКА ЗАВЕРШЕНА"
echo ""
echo "  Python:   $PYTHON"
echo "  Проект:   $PROJECT_DIR"
echo "  Логи:     $LOG_FILE"
echo ""
echo "  Текущий crontab:"
crontab -l 2>/dev/null | grep -E "step11b|CRON" || echo "  (пусто)"
echo ""
echo "  Ручной запуск:"
echo "    cd $PROJECT_DIR && $PYTHON step11b_paper_trading.py"
echo ""
echo "  Просмотр логов в реальном времени:"
echo "    tail -f $LOG_FILE"
echo "============================================================"
