"""급등테마주 강제청산 시각의 '익일 이월(오버나이트)' 판정 이력 조회.

`exits.py` 가 실제 체결된 매도(TradeEntry)를 조립한다면, 이 모듈은 체결이
남지 않는 판정 — 강제청산 시각에 잔량을 익일로 넘긴 오버나이트 — 을
`ThemeExitSignal` 에서 읽어 진입/청산 판정 차트가 쓸 형태로 가공한다.

두 화면(타임라인·진입 차트)이 같은 정의를 봐야 하므로 직렬화를 여기 둔다.
"""

from __future__ import annotations

import calendar
from datetime import date as date_cls, datetime

from pytz import timezone as pytz_tz

_KST = pytz_tz("Asia/Seoul")


def load_overnight_holds(
    user, day: date_cls, stock_code: str | None = None
) -> list[dict]:
    """당일 '익일 이월'로 판정된 오버나이트 보유 내역을 반환한다.

    Args:
        user:       조회 대상 유저
        day:        조회 날짜 (KST)
        stock_code: 지정하면 해당 종목만. None 이면 전 종목.

    Returns:
        오버나이트 1건 = 1행. `chart_time` 은 청산 체결과 같은 규약
        (KST 벽시계를 UTC epoch 으로 환산)을 따른다.
    """
    from myweb.models import ThemeExitSignal

    query = ThemeExitSignal.objects.filter(
        user=user, date=day, decision="overnight"
    )
    if stock_code:
        query = query.filter(stock_code=stock_code)

    return [_overnight_row(row) for row in query.order_by("-checked_at")]


def _overnight_row(row) -> dict:
    moment = row.checked_at.astimezone(_KST) if row.checked_at else None

    return {
        "id": row.id,
        "stock_code": row.stock_code,
        "stock_name": row.stock_name,
        "decision": row.decision,
        "decision_label": row.get_decision_display(),
        "days_held": row.days_held,
        "reason": row.reason,
        "force_exit_time": row.force_exit_time.strftime("%H:%M"),
        "evaluated_at": moment.strftime("%H:%M") if moment else row.force_exit_time.strftime("%H:%M"),
        "chart_time": _chart_time(moment.replace(second=0, microsecond=0)) if moment else None,
        "overnight_evaluated": row.overnight_evaluated,
        "overnight_available": row.overnight_available,
        "overnight_conditions": row.overnight_conditions or [],
        "overnight_met": row.overnight_met or {},
        "overnight_met_count": row.overnight_met_count,
        "overnight_required": row.overnight_required,
        "overnight_detail": row.overnight_detail,
    }


def _chart_time(moment: datetime) -> int:
    """KST 벽시계 시각을 UTC epoch(초)로 환산한다 (entry_chart 와 동일 규약)."""
    return calendar.timegm(
        (moment.year, moment.month, moment.day, moment.hour, moment.minute, moment.second, 0, 0, 0)
    )
