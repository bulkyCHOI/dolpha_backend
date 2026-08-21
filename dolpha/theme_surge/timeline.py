"""09:00~15:30 급등 테마 타임라인 조립.

하루치 ThemeSnapshot / ThemeLeaderCandidate / ThemeEntrySignal 을 읽어
프론트엔드가 그대로 렌더링할 수 있는 형태로 가공한다.

    slots   : 09:00 ~ 15:30 1분 슬롯 라벨 (고정 391칸)
    themes  : 테마별 슬롯 셀 배열 (등락률·거래대금·급등 여부) + 1등 종목
    signals : 진입 조건 판정 이력 (타임라인 위 마커)
"""

from __future__ import annotations

from datetime import date as date_cls, datetime, time as time_cls, timedelta

from .config import MARKET_CLOSE, MARKET_OPEN, SLOT_MINUTES


def slot_labels() -> list[str]:
    """09:00 ~ 15:30 의 SLOT_MINUTES 슬롯 라벨을 생성한다."""
    labels: list[str] = []
    cursor = datetime.combine(date_cls(2000, 1, 1), MARKET_OPEN)
    end = datetime.combine(date_cls(2000, 1, 1), MARKET_CLOSE)

    while cursor <= end:
        labels.append(cursor.strftime("%H:%M"))
        cursor += timedelta(minutes=SLOT_MINUTES)

    return labels


def build_timeline(
    target_date: date_cls, user=None, only_surge: bool = True
) -> dict:
    """하루치 급등 테마 타임라인을 조립한다.

    Args:
        target_date: 조회 날짜
        user:        진입 시그널을 포함할 유저 (None 이면 시그널 생략)
        only_surge:  True 면 하루 중 한 번이라도 급등 판정된 테마만 포함

    Returns:
        타임라인 dict (아래 build_* 헬퍼 참조)
    """
    from myweb.models import ThemeSnapshot

    labels = slot_labels()
    snapshots = list(
        ThemeSnapshot.objects.filter(date=target_date).order_by("slot_time", "rank")
    )

    surge_theme_ids = {s.tics_id for s in snapshots if s.is_surge}
    if only_surge:
        snapshots = [s for s in snapshots if s.tics_id in surge_theme_ids]

    themes = _build_themes(snapshots, labels)
    _attach_leaders(target_date, themes)
    signals = _build_signals(target_date, user) if user is not None else []

    return {
        "date": target_date.isoformat(),
        "slots": labels,
        "themes": themes,
        "signals": signals,
        "summary": {
            "theme_count": len(themes),
            "surge_theme_count": len(surge_theme_ids),
            "signal_count": len(signals),
            "entry_count": sum(1 for s in signals if s["executed"]),
            "scanned_slots": len({s.slot_time for s in snapshots}),
        },
    }


def _build_themes(snapshots: list, labels: list[str]) -> list[dict]:
    """테마별로 슬롯 셀 배열을 만든다."""
    grouped: dict[int, dict] = {}

    for snap in snapshots:
        theme = grouped.setdefault(
            snap.tics_id,
            {
                "tics_id": snap.tics_id,
                "theme_name": snap.theme_name,
                "cells": {},
                "surge_slots": [],
                "peak_rate": 0.0,
                "peak_slot": None,
                "_peak_raw": float("-inf"),
                "max_trading_value": 0,
                "leader": None,
            },
        )

        label = _label(snap.slot_time)
        theme["cells"][label] = {
            "slot": label,
            "rank": snap.rank,
            "rate": round(snap.fluctuation_rate, 2),
            "trading_value": snap.trading_value,
            "momentum": round(snap.momentum, 2),
            "is_surge": snap.is_surge,
            "reason": snap.surge_reason,
        }

        if snap.is_surge:
            theme["surge_slots"].append(label)
        # 반올림 전 원본끼리 비교해야 동률 슬롯에서 peak_slot 이 밀리지 않는다
        if snap.fluctuation_rate > theme["_peak_raw"]:
            theme["_peak_raw"] = snap.fluctuation_rate
            theme["peak_rate"] = round(snap.fluctuation_rate, 2)
            theme["peak_slot"] = label
        theme["max_trading_value"] = max(theme["max_trading_value"], snap.trading_value)

    # 셀을 고정 슬롯 배열로 펼쳐 프론트에서 인덱스로 바로 접근할 수 있게 한다
    themes = []
    for theme in grouped.values():
        cells = theme.pop("cells")
        theme.pop("_peak_raw")
        theme["cells"] = [cells.get(label) for label in labels]
        themes.append(theme)

    themes.sort(key=lambda t: (len(t["surge_slots"]), t["peak_rate"]), reverse=True)
    return themes


def _attach_leaders(target_date: date_cls, themes: list[dict]) -> None:
    """테마별 1등 종목(당일 최고 점수 기준)을 붙인다."""
    from myweb.models import ThemeLeaderCandidate

    if not themes:
        return

    tics_ids = [t["tics_id"] for t in themes]
    best: dict[int, dict] = {}

    candidates = ThemeLeaderCandidate.objects.filter(
        date=target_date, tics_id__in=tics_ids, rank_in_theme=1
    ).order_by("-score")

    for cand in candidates:
        if cand.tics_id in best:
            continue  # score 내림차순이므로 첫 행이 당일 최고 점수
        best[cand.tics_id] = {
            "stock_code": cand.stock_code,
            "stock_name": cand.stock_name,
            "price": cand.price,
            "change_rate": round(cand.change_rate, 2),
            "trading_value": cand.trading_value,
            "score": round(cand.score, 4),
            "slot": _label(cand.slot_time),
        }

    for theme in themes:
        theme["leader"] = best.get(theme["tics_id"])


def _build_signals(target_date: date_cls, user) -> list[dict]:
    """진입 조건 판정 이력을 타임라인 마커용으로 변환한다."""
    from myweb.models import ThemeEntrySignal
    from pytz import timezone as pytz_tz

    kst = pytz_tz("Asia/Seoul")
    rows = ThemeEntrySignal.objects.filter(user=user, date=target_date).order_by("checked_at")

    return [
        {
            "time": row.checked_at.astimezone(kst).strftime("%H:%M"),
            "slot": _label(_floor_slot(row.checked_at.astimezone(kst).time())),
            "tics_id": row.tics_id,
            "theme_name": row.theme_name,
            "stock_code": row.stock_code,
            "stock_name": row.stock_name,
            "price": row.price,
            "prev_high": row.prev_high,
            "pullback_low": row.pullback_low,
            "pullback_pct": row.pullback_pct,
            "volume_ratio": row.volume_ratio,
            "foreign_net_buy": row.foreign_net_buy,
            "has_pullback": row.has_pullback,
            "has_breakout": row.has_breakout,
            "has_foreign_buying": row.has_foreign_buying,
            "passed": row.passed,
            "executed": row.executed,
            "reason": row.reason,
        }
        for row in rows
    ]


def _label(value: time_cls) -> str:
    return value.strftime("%H:%M")


def _floor_slot(value: time_cls) -> time_cls:
    """시각을 SLOT_MINUTES 슬롯으로 내림한다."""
    return value.replace(
        minute=(value.minute // SLOT_MINUTES) * SLOT_MINUTES, second=0, microsecond=0
    )
