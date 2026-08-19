"""진입 조건 판정을 차트로 재구성한다.

ThemeEntrySignal 은 판정 '결과'(전고점 가격, 눌림 저점 가격 등)만 남기고
그 지점이 몇 시 몇 분 봉이었는지는 남기지 않는다. 표만 봐서는 어느 봉이 전고점이고
어디가 눌림 구간인지 알 수 없는 이유가 이것이다.

그래서 저장된 1분봉(stock_minute_ohlcv)을 판정 시각 직전까지 되감아
patterns.py 로 똑같이 다시 계산해 시각 좌표를 복원한다. 재계산한 전고점이 저장된
값과 일치하면 verified=True 로 내려보내, 화면에서 "복원된 좌표"임을 구분할 수 있게 한다.

반환 구조:
    bars      — 당일 1분봉 (lightweight-charts 가 그대로 쓰는 time/ohlcv)
    decisions — 판정 1건 = 1행, 각 행에 geometry(전고점·눌림·돌파 좌표) 포함
    params    — 판정 임계값 (화면 범례·툴팁용)
"""

from __future__ import annotations

import calendar
import re
from datetime import date as date_cls, datetime, timedelta

from pytz import timezone as pytz_tz

from .config import (
    BREAKOUT_BUFFER_PCT,
    BREAKOUT_VOLUME_AVG_BARS,
    BREAKOUT_VOLUME_RATIO_MIN,
    ENTRY_LOOKBACK_BARS,
    ENTRY_MIN_BARS,
    PIVOT_WINDOW,
    PULLBACK_MAX_PCT,
    PULLBACK_MIN_BARS,
    PULLBACK_MIN_PCT,
    PULLBACK_VOLUME_RATIO_MAX,
)
from .patterns import analyze_breakout, analyze_pullback, find_last_swing_high

_KST = pytz_tz("Asia/Seoul")

STOCK_CODE_PATTERN = re.compile(r"^[0-9A-Za-z]{4,10}$")

# 전고점 재계산 값과 저장 값을 같다고 볼 허용 오차(원)
PRICE_EPSILON = 0.01

# 판정 시각 대비 마지막 확정 봉의 오프셋 후보(분).
# 매매 사이클 소요 시간과 분봉 수집 시점이 매번 조금씩 달라, 판정 당시 엔진이 본
# 마지막 봉이 checked_at 기준 -1 ~ +2 분 사이에서 흔들린다. 저장된 전고점을 그대로
# 재현하는 오프셋을 앞에서부터 찾아 쓰고, 못 찾으면 첫 후보(0분)로 근사한다.
CUTOFF_OFFSETS_MIN = (0, 1, -1, 2, -2)

# 화면 범례에 그대로 노출할 판정 임계값
CHART_PARAMS = {
    "lookback_bars": ENTRY_LOOKBACK_BARS,
    "min_bars": ENTRY_MIN_BARS,
    "pivot_window": PIVOT_WINDOW,
    "pullback_min_pct": PULLBACK_MIN_PCT,
    "pullback_max_pct": PULLBACK_MAX_PCT,
    "pullback_min_bars": PULLBACK_MIN_BARS,
    "pullback_volume_ratio_max": PULLBACK_VOLUME_RATIO_MAX,
    "breakout_buffer_pct": BREAKOUT_BUFFER_PCT,
    "breakout_volume_ratio_min": BREAKOUT_VOLUME_RATIO_MIN,
    "breakout_volume_avg_bars": BREAKOUT_VOLUME_AVG_BARS,
}


class EntryChartError(ValueError):
    """차트 조회 입력이 잘못된 경우."""


def build_entry_chart(user, target_date: date_cls, stock_code: str) -> dict:
    """한 종목의 하루치 1분봉과 진입 판정 좌표를 조립한다.

    Args:
        user:        조회 대상 유저 (판정 이력은 유저별로 남는다)
        target_date: 조회 날짜
        stock_code:  6자리 종목코드

    Returns:
        {"stock_code", "stock_name", "theme_name", "bars", "decisions", "params"}

    Raises:
        EntryChartError: 종목코드 형식이 올바르지 않은 경우
    """
    code = (stock_code or "").strip()
    if not STOCK_CODE_PATTERN.match(code):
        raise EntryChartError("종목코드 형식이 올바르지 않습니다.")

    signals = _load_signals(user, target_date, code)
    bars = load_minute_bars(code, target_date)
    identity = _identity(signals, target_date, code)

    return {
        "date": target_date.isoformat(),
        "stock_code": code,
        "stock_name": identity["stock_name"],
        "theme_name": identity["theme_name"],
        "bars": bars,
        "decisions": [_decision_row(bars, signal) for signal in signals],
        "params": CHART_PARAMS,
    }


def load_minute_bars(stock_code: str, target_date: date_cls) -> list[dict]:
    """당일 1분봉을 lightweight-charts 가 바로 쓰는 형태로 읽는다.

    time 은 KST 벽시계 시각을 그대로 UTC epoch 로 환산한 값이다.
    lightweight-charts 가 timestamp 를 UTC 로 렌더링하므로, 이렇게 보내야
    09:00 봉이 화면에도 09:00 으로 찍힌다 (StockChartModal 과 동일한 규약).
    """
    from myweb.models import StockMinuteOhlcv

    start = _KST.localize(datetime.combine(target_date, datetime.min.time()))
    end = _KST.localize(datetime.combine(target_date, datetime.max.time()))

    rows = StockMinuteOhlcv.objects.filter(
        stock_code=stock_code,
        bar_datetime__range=(start, end),
        volume__gt=0,
    ).order_by("bar_datetime")

    bars = []
    for row in rows:
        moment = row.bar_datetime.astimezone(_KST)
        bars.append(
            {
                "time": _chart_time(moment),
                "label": moment.strftime("%H:%M"),
                "open": row.open,
                "high": row.high,
                "low": row.low,
                "close": row.close,
                "volume": row.volume,
            }
        )
    return bars


# ──────────────────────────────────────────────────────────────
# 판정 행 조립
# ──────────────────────────────────────────────────────────────

def _decision_row(bars: list[dict], signal) -> dict:
    """판정 1건을 차트용 행으로 변환한다."""
    checked = signal.checked_at.astimezone(_KST)
    met = sum([signal.has_pullback, signal.has_breakout, signal.has_foreign_buying])

    return {
        "id": signal.id,
        "time": checked.strftime("%H:%M"),
        "chart_time": _chart_time(checked.replace(second=0, microsecond=0)),
        "price": signal.price,
        "prev_high": signal.prev_high,
        "pullback_low": signal.pullback_low,
        "pullback_pct": signal.pullback_pct,
        "volume_ratio": signal.volume_ratio,
        "foreign_net_buy": signal.foreign_net_buy,
        "has_pullback": signal.has_pullback,
        "has_breakout": signal.has_breakout,
        "has_foreign_buying": signal.has_foreign_buying,
        "conditions_met": met,
        "passed": signal.passed,
        "executed": signal.executed,
        "reason": signal.reason,
        "geometry": _geometry(bars, signal, checked),
    }


def _geometry(bars: list[dict], signal, checked: datetime) -> dict | None:
    """판정 시점의 분봉을 되감아 전고점·눌림·돌파의 시각 좌표를 복원한다.

    Returns:
        좌표 dict. 분봉이 없거나 전고점을 다시 찾지 못하면 None.
    """
    found = _best_window(bars, checked, signal.prev_high)
    if found is None:
        return None

    window, swing, verified = found
    pullback = analyze_pullback(window, swing)
    breakout = analyze_breakout(window, swing, signal.price)

    pullback_bars = window[swing.index + 1 : len(window) - 1]
    rise_bars = window[max(0, swing.index - len(pullback_bars)) : swing.index + 1]
    low_bar = min(pullback_bars, key=lambda b: b["low"]) if pullback_bars else None

    return {
        "verified": verified,
        "window_from": window[0]["time"],
        "swing_high": {"time": window[swing.index]["time"], "price": swing.price},
        "pullback_low": (
            {"time": low_bar["time"], "price": low_bar["low"]} if low_bar else None
        ),
        "pullback_zone": _zone(pullback_bars),
        "rise_zone": _zone(rise_bars),
        "breakout_threshold": round(breakout.threshold, 2),
        "decision_bar": window[-1]["time"],
        "recomputed": {
            "prev_high": swing.price,
            "pullback_low": pullback.low or None,
            "pullback_pct": round(pullback.depth_pct, 2),
            "pullback_bar_count": pullback.bar_count,
            "pullback_volume_ratio": round(pullback.volume_ratio, 2),
            "breakout_volume_ratio": round(breakout.volume_ratio, 2),
            "pullback_reason": pullback.reason,
            "breakout_reason": breakout.reason,
        },
    }


def _best_window(
    bars: list[dict], checked: datetime, stored_high: float | None
) -> tuple[list[dict], object, bool] | None:
    """저장된 전고점을 재현하는 판정 윈도우를 찾는다.

    Args:
        bars:        당일 전체 분봉
        checked:     판정 시각(KST)
        stored_high: 판정 당시 기록된 전고점. None 이면 근사만 한다.

    Returns:
        (윈도우, 스윙 고점, 저장값 일치 여부). 어떤 오프셋으로도 전고점을 못 찾으면 None.
    """
    minute = checked.replace(second=0, microsecond=0)
    fallback = None

    for offset in CUTOFF_OFFSETS_MIN:
        cutoff = _chart_time(minute + timedelta(minutes=offset))
        visible = [bar for bar in bars if bar["time"] < cutoff]
        if len(visible) < ENTRY_MIN_BARS:
            continue

        window = visible[-ENTRY_LOOKBACK_BARS:]
        swing = find_last_swing_high(window)
        if swing is None:
            continue

        if _matches(swing.price, stored_high):
            return window, swing, True
        if fallback is None:
            fallback = (window, swing, False)

    return fallback


def _zone(zone_bars: list[dict]) -> dict | None:
    """봉 구간을 {from, to} 시각 좌표로 변환한다."""
    if not zone_bars:
        return None
    return {"from": zone_bars[0]["time"], "to": zone_bars[-1]["time"]}


def _matches(recomputed: float, stored: float | None) -> bool:
    """재계산한 가격이 저장된 값과 같은지 (부동소수 오차 허용)."""
    if stored is None:
        return False
    return abs(recomputed - stored) <= PRICE_EPSILON


# ──────────────────────────────────────────────────────────────
# 소스 조회
# ──────────────────────────────────────────────────────────────

def _load_signals(user, target_date: date_cls, stock_code: str) -> list:
    from myweb.models import ThemeEntrySignal

    return list(
        ThemeEntrySignal.objects.filter(
            user=user, date=target_date, stock_code=stock_code
        ).order_by("checked_at")
    )


def _identity(signals: list, target_date: date_cls, stock_code: str) -> dict:
    """종목명·테마명. 판정 이력이 없으면 주도주 후보에서 찾는다."""
    if signals:
        latest = signals[-1]
        if latest.stock_name:
            return {"stock_name": latest.stock_name, "theme_name": latest.theme_name}

    from myweb.models import ThemeLeaderCandidate

    candidate = (
        ThemeLeaderCandidate.objects.filter(date=target_date, stock_code=stock_code)
        .order_by("-slot_time")
        .first()
    )
    if candidate:
        return {"stock_name": candidate.stock_name, "theme_name": candidate.theme_name}

    return {"stock_name": stock_code, "theme_name": ""}


def _chart_time(moment: datetime) -> int:
    """KST 벽시계 시각을 UTC epoch(초)로 환산한다."""
    return calendar.timegm(
        (moment.year, moment.month, moment.day, moment.hour, moment.minute, moment.second, 0, 0, 0)
    )
