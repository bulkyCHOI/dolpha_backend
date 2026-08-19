"""급등테마주 진입 판정.

TradingEngine 이 매 분 호출한다. 3단 조건을 모두 만족해야 진입한다.

    1) 눌림목      — 직전 스윙 고점(전고점) 이후 적정 깊이의 조정, 거래량 감소 동반
    2) 전고점 돌파 — 현재가가 전고점을 거래량을 실어 상향 돌파
    3) 외국인 매수세 — 당일 외국인 순매수가 (+) 이고 최근 구간에서 증가

판정 결과는 ThemeEntrySignal 에 남겨 타임라인에 진입 시점 마커로 표시하고,
미진입 사유를 추적할 수 있게 한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as date_cls, datetime

from .config import ENTRY_LOOKBACK_BARS, ENTRY_MIN_BARS, FOREIGN_REQUIRE_INCREASING
from .foreign_flow import ForeignFlow, get_foreign_flow
from .patterns import analyze_breakout, analyze_pullback, find_last_swing_high


@dataclass(frozen=True)
class EntryDecision:
    """진입 판정 결과."""

    passed: bool
    reason: str
    prev_high: float | None = None
    pullback_low: float | None = None
    pullback_pct: float | None = None
    volume_ratio: float | None = None
    foreign_net_buy: int | None = None
    has_pullback: bool = False
    has_breakout: bool = False
    has_foreign_buying: bool = False


def check_theme_surge_entry(
    stock_code: str,
    current_price: float,
    use_foreign_filter: bool = True,
    bars: list[dict] | None = None,
    prev_foreign_net_buy: int | None = None,
) -> EntryDecision:
    """급등테마주 신규 진입 조건을 판정한다.

    Args:
        stock_code:           6자리 종목코드
        current_price:        현재가
        use_foreign_filter:   외국인 매수세 조건 적용 여부
        bars:                 1분봉 리스트. None 이면 DB에서 당일 분봉을 읽는다.
        prev_foreign_net_buy: 직전 판정 시 기록된 외국인 순매수(추세 폴백용)

    Returns:
        EntryDecision — passed 가 True 일 때만 매수를 실행한다.
    """
    minute_bars = bars if bars is not None else load_today_minute_bars(stock_code)

    if len(minute_bars) < ENTRY_MIN_BARS:
        return EntryDecision(
            passed=False,
            reason=f"분봉 부족 ({len(minute_bars)}봉 < {ENTRY_MIN_BARS}봉)",
        )

    window = minute_bars[-ENTRY_LOOKBACK_BARS:]

    swing = find_last_swing_high(window)
    if swing is None:
        return EntryDecision(passed=False, reason="전고점(스윙 고점) 미탐지")

    pullback = analyze_pullback(window, swing)
    breakout = analyze_breakout(window, swing, current_price)

    # 눌림목 없이 그냥 오른 것은 추격 매수이므로 진입하지 않는다
    if not pullback.is_valid:
        return EntryDecision(
            passed=False,
            reason=f"눌림목 미성립 — {pullback.reason}",
            prev_high=swing.price,
            pullback_low=pullback.low or None,
            pullback_pct=pullback.depth_pct,
            volume_ratio=breakout.volume_ratio,
            has_breakout=breakout.is_valid,
        )

    if not breakout.is_valid:
        return EntryDecision(
            passed=False,
            reason=f"돌파 미성립 — {breakout.reason}",
            prev_high=swing.price,
            pullback_low=pullback.low,
            pullback_pct=pullback.depth_pct,
            volume_ratio=breakout.volume_ratio,
            has_pullback=True,
        )

    if not use_foreign_filter:
        return EntryDecision(
            passed=True,
            reason=f"{pullback.reason} → {breakout.reason} (외국인 필터 미사용)",
            prev_high=swing.price,
            pullback_low=pullback.low,
            pullback_pct=pullback.depth_pct,
            volume_ratio=breakout.volume_ratio,
            has_pullback=True,
            has_breakout=True,
        )

    flow = get_foreign_flow(stock_code, prev_foreign_net_buy)
    foreign_ok = _is_foreign_buying(flow)

    if not foreign_ok:
        return EntryDecision(
            passed=False,
            reason=f"외국인 매수세 미포착 — {flow.detail}",
            prev_high=swing.price,
            pullback_low=pullback.low,
            pullback_pct=pullback.depth_pct,
            volume_ratio=breakout.volume_ratio,
            foreign_net_buy=flow.net_buy_qty if flow.available else None,
            has_pullback=True,
            has_breakout=True,
        )

    return EntryDecision(
        passed=True,
        reason=f"{pullback.reason} → {breakout.reason} → {flow.detail}",
        prev_high=swing.price,
        pullback_low=pullback.low,
        pullback_pct=pullback.depth_pct,
        volume_ratio=breakout.volume_ratio,
        foreign_net_buy=flow.net_buy_qty,
        has_pullback=True,
        has_breakout=True,
        has_foreign_buying=True,
    )


def _is_foreign_buying(flow: ForeignFlow) -> bool:
    """외국인 수급 조건 충족 여부."""
    if not flow.is_buying:
        return False
    if FOREIGN_REQUIRE_INCREASING and not flow.is_increasing:
        return False
    return True


def load_today_minute_bars(stock_code: str, target_date: date_cls | None = None) -> list[dict]:
    """DB(stock_minute_ohlcv)에서 당일 정규장 분봉을 시간 오름차순으로 읽는다.

    TradingEngine 이 매 사이클마다 직전 '완성된' 분봉을 수집하므로 별도 API 호출이 필요 없다.
    따라서 마지막 원소는 진행 중인 봉이 아니라 직전 1분의 확정 봉이며,
    돌파 판정은 이 확정 봉의 거래량과 실시간 현재가를 함께 본다.
    """
    from pytz import timezone as pytz_tz

    from myweb.models import StockMinuteOhlcv

    kst = pytz_tz("Asia/Seoul")
    day = target_date or datetime.now(kst).date()
    start = kst.localize(datetime(day.year, day.month, day.day, 0, 0, 0))
    end = kst.localize(datetime(day.year, day.month, day.day, 23, 59, 59))

    rows = StockMinuteOhlcv.objects.filter(
        stock_code=stock_code,
        bar_datetime__range=(start, end),
        volume__gt=0,
    ).order_by("bar_datetime")

    return [
        {
            "time": row.bar_datetime.astimezone(kst).strftime("%H:%M"),
            "open": row.open,
            "high": row.high,
            "low": row.low,
            "close": row.close,
            "volume": row.volume,
        }
        for row in rows
    ]
