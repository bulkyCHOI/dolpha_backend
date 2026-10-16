"""급등테마주 진입 판정 맥락 조회 — 당일 매매 집계와 진입 시점 테마 상태.

entry_gates.py 의 순수 판정에 넣을 값을 DB 에서 모은다.

주의: filled_at 등 DateTimeField 는 `__date` lookup 을 쓰지 않는다. MySQL 타임존
테이블이 비어 있어 CONVERT_TZ 가 NULL 을 돌려주므로 항상 0건이 된다. KST 하루를
aware datetime 범위로 잘라 조회한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from .entry_gates import (
    PREFIX_CIRCUIT_BREAKER,
    DailyStats,
    check_risk_gates,
    describe_gate_block,
    load_entry_gate_settings,
)

STRATEGY_TYPE = "theme_surge"
# 라운드를 끝내는 매도 유형. 분할 익절(EXIT_PARTIAL)은 라운드를 이어간다.
_TERMINAL_SELL_TYPES = {"EXIT_FULL", "STOP_LOSS", "TRAILING_STOP"}


@dataclass(frozen=True)
class ThemeContext:
    """진입 판정 시점의 테마 상태."""

    tics_id: int
    theme_name: str
    fluctuation: float | None     # 가장 최근 스냅샷의 테마 등락률(%)
    slot_time: str | None         # 그 스냅샷 슬롯 (HH:MM) — 등락률의 신선도 확인용


def _kst_day_range(now: datetime) -> tuple[datetime, datetime]:
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return start, start + timedelta(days=1)


def load_daily_stats(user, now: datetime) -> DailyStats:
    """오늘(KST) 급등테마주 매매를 라운드 단위로 집계한다.

    손실 라운드: 라운드 종료 매도(전량청산·손절·트레일링) 시점까지 같은 설정의
    오늘 매도 손익 합계가 음수인 라운드. 전일 이월 포지션이 오늘 청산된 것도 포함한다
    (오늘 장세에서 난 손실이므로 게이트 판단에 넣는다).
    """
    from myweb.models import TradeEntry

    start, end = _kst_day_range(now)
    base = TradeEntry.objects.filter(
        user=user,
        trading_config__strategy_type=STRATEGY_TYPE,
        filled_quantity__gt=0,
        filled_at__gte=start,
        filled_at__lt=end,
    )

    rounds_started = base.filter(trade_type="BUY", entry_type="INITIAL").count()

    sells = base.filter(trade_type="SELL").order_by("filled_at").values(
        "trading_config_id", "entry_type", "profit_loss"
    )
    running: dict[int, float] = {}
    losses = 0
    total = 0.0
    for row in sells:
        pnl = float(row["profit_loss"] or 0.0)
        total += pnl
        key = row["trading_config_id"]
        running[key] = running.get(key, 0.0) + pnl
        if row["entry_type"] in _TERMINAL_SELL_TYPES:
            if running[key] < 0:
                losses += 1
            running[key] = 0.0

    return DailyStats(rounds_started=rounds_started, realized_losses=losses, realized_pnl=total)


def load_theme_context(stock_code: str, now: datetime) -> ThemeContext:
    """종목이 속한 급등 테마(가장 최근 후보 등록 기준)의 최신 등락률을 조회한다.

    후보의 테마가 이후 슬롯에서 상위 랭킹 밖으로 밀려 스냅샷이 없으면,
    남아 있는 가장 최근 스냅샷 값을 쓰고 slot_time 으로 그 시점을 알린다.
    """
    from myweb.models import ThemeLeaderCandidate, ThemeSnapshot

    today = now.date()
    candidate = (
        ThemeLeaderCandidate.objects.filter(date=today, stock_code=stock_code)
        .order_by("-slot_time")
        .only("tics_id", "theme_name")
        .first()
    )
    if candidate is None:
        return ThemeContext(tics_id=0, theme_name="", fluctuation=None, slot_time=None)

    snap = (
        ThemeSnapshot.objects.filter(
            date=today, tics_id=candidate.tics_id, slot_time__lte=now.time()
        )
        .order_by("-slot_time")
        .only("fluctuation_rate", "slot_time")
        .first()
    )
    return ThemeContext(
        tics_id=candidate.tics_id,
        theme_name=candidate.theme_name,
        fluctuation=float(snap.fluctuation_rate) if snap else None,
        slot_time=snap.slot_time.strftime("%H:%M") if snap else None,
    )


def current_gate_status(user, now: datetime) -> dict | None:
    """지금 이 유저의 신규 진입이 리스크 게이트로 막혀 있는지 (화면 상단 배너용).

    계좌 확정원금은 여기서 조회하지 않으므로(KIS 호출 회피) 손실 % 한도는 라이브 판정을
    못 한다. 대신 오늘 엔진이 남긴 가장 최근 판정의 gate_flags 에 서킷브레이커가 찍혀 있으면
    그것을 쓴다 — 실현손실은 당일 줄어들지 않으므로 한 번 걸리면 그날 내내 유효하다.

    Returns:
        describe_gate_block 형식 dict (막혀 있지 않으면 None)
    """
    from myweb.models import ThemeEntrySignal

    settings = load_entry_gate_settings(getattr(user, "trading_defaults", None))
    live = check_risk_gates(settings, now.time(), load_daily_stats(user, now), None)
    if live.blocked:
        return describe_gate_block(live.reason)

    start, end = _kst_day_range(now)
    latest_flags = (
        ThemeEntrySignal.objects.filter(user=user, checked_at__gte=start, checked_at__lt=end)
        .exclude(gate_flags={})
        .order_by("-checked_at")
        .values_list("gate_flags", flat=True)
        .first()
    )
    recorded = ((latest_flags or {}).get("risk") or {}).get("blocked_by") or ""
    if recorded.startswith(PREFIX_CIRCUIT_BREAKER):
        return describe_gate_block(recorded)
    return None
