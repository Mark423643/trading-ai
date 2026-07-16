"""
levels_timur.py — построение уровней по методу Тимура.

Заменяет механические пивоты find_pivots() из step11b_paper_trading.py.
Тимур строит уровни двумя способами (его формулировки с разметок):

  ТИП 1 — "уровень Пбара + первое откатное движение":
      пин-бар -> импульс от него -> первый откат;
      уровень = экстремум этого отката.

  ТИП 2 — "уровень отката после которого был ЛП и смена тренда":
      свинг-экстремум ложно пробит (тень за него, закрытие обратно),
      после чего сменился тренд; уровень = сам пробитый экстремум.
      ВНИМАНИЕ: формулировка допускает несколько прочтений — это
      рабочая интерпретация, надо подтвердить у Тимура.

КРИТИЧНО ПРО confirm:
    У каждого уровня есть поле confirm — дата, раньше которой уровень
    ЕЩЁ НЕ БЫЛ ИЗВЕСТЕН (откат должен сначала закончиться, тренд —
    смениться). Любой бэктест обязан фильтровать уровни по confirm <= дата
    бара. Без этого получается загляд вперёд — ровно то, на чём сгорел
    step_portfolio_final.py со своими +482R.
"""

import numpy as np

# ── параметры распознавания ──────────────────────────────────
PIN_SHADOW_BODY  = 2.0   # тень пин-бара > 2 тел
PIN_SHADOW_RANGE = 0.5   # тень пин-бара > 50% всего диапазона бара
IMPULSE_MIN_ATR  = 0.5   # импульс после пин-бара минимум 0.5 daily ATR
FWD_WINDOW       = 15    # окно поиска импульса/отката (дн. баров)
CONFIRM_BARS     = 2     # баров после экстремума отката до подтверждения
LP_SWING_LB      = 10    # радиус свинга для поиска ЛП


def _shape(o, h, l, c):
    """Геометрия бара: тело, диапазон, верхняя и нижняя тени."""
    body  = abs(c - o)
    rng   = h - l
    up_sh = h - max(o, c)
    lo_sh = min(o, c) - l
    return body, rng, up_sh, lo_sh


def is_pin_bar(o, h, l, c):
    """'bull' — длинная нижняя тень, 'bear' — длинная верхняя, иначе None."""
    body, rng, up_sh, lo_sh = _shape(o, h, l, c)
    if rng <= 0:
        return None
    if lo_sh > PIN_SHADOW_BODY * body and lo_sh > PIN_SHADOW_RANGE * rng:
        return "bull"
    if up_sh > PIN_SHADOW_BODY * body and up_sh > PIN_SHADOW_RANGE * rng:
        return "bear"
    return None


def find_pbar_levels(df1d, atr):
    """
    ТИП 1. Бычий пин-бар -> импульс вверх -> первый откат вниз;
    low отката = поддержка. Медвежий пин-бар — зеркально.
    """
    o = df1d["Open"].values
    h = df1d["High"].values
    l = df1d["Low"].values
    c = df1d["Close"].values
    idx = df1d.index
    n = len(df1d)
    out = []

    for i in range(n):
        kind = is_pin_bar(o[i], h[i], l[i], c[i])
        if kind is None:
            continue
        a = atr[i]
        if np.isnan(a) or a <= 0:
            continue
        if i + 4 >= n:
            continue

        if kind == "bull":
            seg = h[i + 1:min(i + FWD_WINDOW, n)]
            if len(seg) == 0:
                continue
            pk = int(np.argmax(seg)) + i + 1
            if h[pk] - h[i] < IMPULSE_MIN_ATR * a:
                continue                       # импульса не было — не уровень
            seg2 = l[pk + 1:min(pk + FWD_WINDOW, n)]
            if len(seg2) < 2:
                continue
            tr = int(np.argmin(seg2)) + pk + 1
            ci = tr + CONFIRM_BARS
            if ci >= n:
                continue
            out.append({"level": float(l[tr]), "type": "support",
                        "origin": "pbar", "born": idx[tr], "confirm": idx[ci]})
        else:
            seg = l[i + 1:min(i + FWD_WINDOW, n)]
            if len(seg) == 0:
                continue
            pk = int(np.argmin(seg)) + i + 1
            if l[i] - l[pk] < IMPULSE_MIN_ATR * a:
                continue
            seg2 = h[pk + 1:min(pk + FWD_WINDOW, n)]
            if len(seg2) < 2:
                continue
            tr = int(np.argmax(seg2)) + pk + 1
            ci = tr + CONFIRM_BARS
            if ci >= n:
                continue
            out.append({"level": float(h[tr]), "type": "resistance",
                        "origin": "pbar", "born": idx[tr], "confirm": idx[ci]})
    return out


def find_lp_trend_levels(df1d, atr):
    """
    ТИП 2. Свинг-экстремум ложно пробит, после чего сменился тренд.
    Смена тренда подтверждается закрытием за противоположным краем свинга.
    Уровень = пробитый экстремум.
    """
    h = df1d["High"].values
    l = df1d["Low"].values
    c = df1d["Close"].values
    idx = df1d.index
    n = len(df1d)
    out = []

    for j in range(LP_SWING_LB, n):
        a = atr[j]
        if np.isnan(a) or a <= 0:
            continue
        sw_hi = h[j - LP_SWING_LB:j].max()
        sw_lo = l[j - LP_SWING_LB:j].min()
        fw_hi = min(j + 1 + FWD_WINDOW, n)
        if fw_hi - j < 4:
            continue
        fw = c[j + 1:fw_hi]

        # ЛП сопротивления: тень выше свинга, закрытие обратно под ним
        if h[j] > sw_hi and c[j] < sw_hi:
            below = np.where(fw < sw_lo)[0]     # смена тренда: ушли под свинг
            if len(below) > 0:
                ci = j + 1 + int(below[0])
                out.append({"level": float(sw_hi), "type": "resistance",
                            "origin": "lp_trend", "born": idx[j], "confirm": idx[ci]})

        # ЛП поддержки: тень ниже свинга, закрытие обратно над ним
        if l[j] < sw_lo and c[j] > sw_lo:
            above = np.where(fw > sw_hi)[0]
            if len(above) > 0:
                ci = j + 1 + int(above[0])
                out.append({"level": float(sw_lo), "type": "support",
                            "origin": "lp_trend", "born": idx[j], "confirm": idx[ci]})
    return out


def find_levels_timur(df1d, atr):
    """Оба типа уровней вместе. Каждый несёт born и confirm."""
    return find_pbar_levels(df1d, atr) + find_lp_trend_levels(df1d, atr)
