"""급등테마주 청산(매도) 체결 이력 조회.

진입은 ThemeEntrySignal 에 "판정" 이력이 매 사이클 남지만, 청산은 그런 판정
이력이 없고 실제 체결된 TradeEntry 만 남는다. 그래서 청산 쪽 화면은 모두
이 모듈이 조립한 체결 기록을 쓴다.

    자동매매 현황 패널   positions.py  → 종목 구분 없이 당일 전체
    진입/청산 판정 차트  entry_chart.py → 특정 종목만 (차트 좌표 포함)

두 화면이 같은 정의를 봐야 하므로 조회를 여기 한 곳에 둔다.
"""

from __future__ import annotations

import calendar
from datetime import date as date_cls, datetime, timedelta

from pytz import timezone as pytz_tz

_KST = pytz_tz("Asia/Seoul")

STRATEGY = "theme_surge"


def kst_day_range(day: date_cls) -> tuple[datetime, datetime]:
    """KST 기준 하루의 [시작, 끝) 경계를 timezone-aware 로 만든다.

    `filled_at__date=day` 같은 lookup 은 쓰지 않는다. 이 DB 는 MySQL 타임존
    테이블이 로드돼 있지 않아 Django 가 KST 변환에 쓰는 CONVERT_TZ 가 항상
    NULL 을 반환하고, `__date` lookup 이 조용히 0건으로 빠진다.
    """
    start = _KST.localize(datetime.combine(day, datetime.min.time()))
    return start, start + timedelta(days=1)


def load_exits(user, day: date_cls, stock_code: str | None = None) -> list[dict]:
    """당일 체결된 급등테마주 청산 내역을 최신순으로 반환한다.

    Args:
        user:       조회 대상 유저
        day:        조회 날짜 (KST)
        stock_code: 지정하면 해당 종목만. None 이면 전 종목.

    Returns:
        청산 1건 = 1행. `chart_time` 은 KST 벽시계를 UTC epoch 으로 환산한
        값으로, lightweight-charts 의 봉 시각과 같은 규약을 따른다.
    """
    from myweb.models import TradeEntry

    start, end = kst_day_range(day)

    query = TradeEntry.objects.filter(
        user=user,
        trading_config__strategy_type=STRATEGY,
        trade_type="SELL",
        status="FILLED",
        filled_at__gte=start,
        filled_at__lt=end,
    )
    if stock_code:
        query = query.filter(stock_code=stock_code)

    return [_exit_row(row) for row in query.order_by("-filled_at")]


def _exit_row(row) -> dict:
    moment = row.filled_at.astimezone(_KST) if row.filled_at else None

    return {
        "id": row.id,
        "stock_code": row.stock_code,
        "stock_name": row.stock_name,
        "exit_type": row.entry_type,
        "exit_type_label": row.get_entry_type_display(),
        "is_partial": row.entry_type == "EXIT_PARTIAL",
        "quantity": row.filled_quantity,
        "exit_price": round(float(row.filled_price)),
        "profit_loss": round(float(row.profit_loss)) if row.profit_loss is not None else None,
        "profit_loss_rate": (
            round(row.profit_loss_percent, 2) if row.profit_loss_percent is not None else None
        ),
        "reason": row.note,
        "exited_at": moment.strftime("%H:%M") if moment else None,
        "chart_time": _chart_time(moment.replace(second=0, microsecond=0)) if moment else None,
    }


def _chart_time(moment: datetime) -> int:
    """KST 벽시계 시각을 UTC epoch(초)로 환산한다 (entry_chart 와 동일 규약)."""
    return calendar.timegm(
        (moment.year, moment.month, moment.day, moment.hour, moment.minute, moment.second, 0, 0, 0)
    )
