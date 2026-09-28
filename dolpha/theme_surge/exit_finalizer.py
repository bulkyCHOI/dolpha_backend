"""장 마감 후 급등테마주 청산/오버나이트 확정.

장중 매매 사이클은 15:30 에 멈추므로, 그 시점에 아직 살아 있는 급등테마주
포지션은 '오늘 밤을 넘긴다(오버나이트)'는 사실이 확정된다. 이 모듈은 장 마감
직후(스케줄러 15:31) 실행되어, 활성 포지션마다 그 확정 상태를 ThemeExitSignal
에 기록한다 — 타임라인 마지막 슬롯(장 마감)에 '오버나이트' 마커로 표시된다.

수급 4조건 분석은 15:29 에 저장된 InvestorFlowSnapshot 으로 채운다(장 마감 후
KIS 매매동향 API 는 조회 불가). 라이브 엔진이 강제청산 시각에 이미 실시간
판정을 남겼다면 그 분석을 보존한다.
"""

from __future__ import annotations

from datetime import date as date_cls, timedelta


def count_trading_days(start: date_cls, end: date_cls) -> int:
    """start~end(양 끝 포함) 사이의 개장일 수. start > end 이면 1."""
    if start >= end:
        return 1
    from dolpha.kis.holiday import is_trading_day

    days = 0
    cursor = start
    while cursor <= end:
        if is_trading_day(cursor):
            days += 1
        cursor += timedelta(days=1)
    return max(1, days)


def _closing_profit_r(config, today: date_cls) -> float | None:
    """장 마감 시점 평가손익(R). 매수 평단 vs 당일 마지막 1분봉 종가, 손절가는 진입 시 확정값.

    산출할 수 없으면 None (호출부는 이월 조건 미충족으로 본다).
    """
    from datetime import datetime, time as time_cls

    from django.utils import timezone as tz

    from myweb.models import StockMinuteOhlcv, TradeEntry
    from dolpha.theme_surge.overnight import unrealized_r

    buys = TradeEntry.objects.filter(
        trading_config=config, trade_type="BUY", filled_quantity__gt=0
    ).values_list("filled_price", "filled_quantity")
    qty = sum(int(q) for _, q in buys)
    if qty <= 0:
        return None
    avg_price = sum(float(p) * int(q) for p, q in buys) / qty

    # __date lookup 은 MySQL 타임존 테이블 문제로 쓰지 않고 KST 하루 범위로 자른다
    start = tz.make_aware(datetime.combine(today, time_cls.min))
    last_bar = (
        StockMinuteOhlcv.objects.filter(
            stock_code=config.stock_code,
            bar_datetime__gte=start,
            bar_datetime__lt=start + timedelta(days=1),
        )
        .order_by("-bar_datetime")
        .values_list("close", flat=True)
        .first()
    )
    if not last_bar:
        return None
    return unrealized_r(avg_price, float(last_bar), config.theme_pullback_low)


# 라이브 엔진이 이미 '청산됨'으로 남긴 판정은 장 마감 확정이 건드리지 않는다.
_LIQUIDATED_DECISIONS = {"force_exit", "max_days", "stop_loss", "trailing", "staged"}


def finalize_theme_exit_signals() -> dict:
    """활성 급등테마주 포지션을 '오버나이트'로 확정 기록한다.

    Returns:
        {"active": int, "recorded": int, "skipped": int}
    """
    from django.utils import timezone as tz

    from myweb.models import (
        InvestorFlowSnapshot,
        ThemeEntrySignal,
        ThemeExitSignal,
        TradeEntry,
        TradingConfig,
    )
    from dolpha.theme_surge.exit_rules import load_exit_settings
    from dolpha.theme_surge.overnight import (
        apply_profit_cushion,
        evaluate_overnight_signal_from_snapshot,
    )

    today = tz.localdate()
    from dolpha.kis.holiday import is_trading_day

    if not is_trading_day(today):
        return {"active": 0, "recorded": 0, "skipped": 0}

    configs = list(
        TradingConfig.objects.filter(strategy_type="theme_surge", is_active=True)
        .select_related("user")
    )
    recorded = skipped = 0

    for config in configs:
        first_buy = (
            TradeEntry.objects
            .filter(user=config.user, trading_config=config,
                    trade_type="BUY", status="FILLED")
            .order_by("filled_at")
            .first()
        )
        if not first_buy or not first_buy.filled_at:
            skipped += 1
            continue

        start = tz.localtime(first_buy.filled_at).date()
        days_held = count_trading_days(start, today)

        existing = ThemeExitSignal.objects.filter(
            user=config.user, date=today, stock_code=config.stock_code
        ).first()

        # 이미 라이브 엔진이 '청산'으로 기록했다면 손대지 않는다.
        if existing and existing.decision in _LIQUIDATED_DECISIONS:
            skipped += 1
            continue

        # 강제청산이 가격조회 실패로 무산된 기록은 '오버나이트'로 덮어쓰지 않고 보존한다.
        # 정상적인 이월 판단처럼 보이면 다음 거래일 복구 청산(트레이딩 엔진의
        # _pending_force_exit_failure)이 이 기록을 찾지 못해 사고가 의도된 전략으로
        # 둔갑해버린다.
        if existing and existing.decision == "force_exit_failed":
            skipped += 1
            continue

        settings = load_exit_settings(getattr(config.user, "trading_defaults", None))
        meta = (
            ThemeEntrySignal.objects
            .filter(user=config.user, stock_code=config.stock_code, date=today)
            .order_by("-checked_at")
            .values("tics_id", "theme_name")
            .first()
        ) or {}

        # 라이브에서 이미 수급을 실시간 평가했으면 그 분석을 유지, 없으면 스냅샷으로 채운다.
        if existing and existing.overnight_evaluated:
            fields = {
                "decision": "overnight",
                "days_held": days_held,
                "reason": existing.reason or "장 마감 시점 보유 지속",
            }
        else:
            snapshot = InvestorFlowSnapshot.objects.filter(
                stock_code=config.stock_code, date=today
            ).first()
            signal = evaluate_overnight_signal_from_snapshot(
                snapshot, settings.overnight_conditions,
                settings.overnight_min_count, today,
            )
            supply_met = signal.should_hold
            signal = apply_profit_cushion(
                signal, _closing_profit_r(config, today), settings.overnight_min_profit_r
            )
            if not settings.overnight_enabled:
                reason = "장 마감 시점 보유 지속 (오버나이트 미사용 — 강제청산 누락 가능)"
            elif signal.should_hold:
                reason = f"수급 조건 충족(장 마감 스냅샷) {signal.met_count}/{signal.required}"
            elif supply_met:
                reason = "장 마감 시점 보유 지속 — 수익쿠션 부족(규칙상 청산 대상)"
            else:
                reason = (
                    f"장 마감 시점 보유 지속 — 수급 {signal.met_count}/{signal.required}"
                    " 미충족(규칙상 청산 대상)"
                )
            fields = {
                "checked_at": tz.now(),
                "tics_id": meta.get("tics_id", 0),
                "theme_name": meta.get("theme_name", ""),
                "stock_name": config.stock_name,
                "force_exit_time": settings.force_exit_time,
                "days_held": days_held,
                "decision": "overnight",
                "reason": reason,
                "overnight_evaluated": bool(settings.overnight_enabled),
                "overnight_available": signal.available,
                "overnight_conditions": list(settings.overnight_conditions),
                "overnight_met": dict(signal.met),
                "overnight_met_count": signal.met_count,
                "overnight_required": signal.required,
                "overnight_detail": signal.detail[:500],
            }

        ThemeExitSignal.objects.update_or_create(
            user=config.user, date=today, stock_code=config.stock_code,
            defaults=fields,
        )
        recorded += 1

    return {"active": len(configs), "recorded": recorded, "skipped": skipped}
