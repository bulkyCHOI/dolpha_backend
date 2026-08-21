"""1분 주기 급등 테마 스캔 오케스트레이션.

APScheduler 가 09:00~15:30 사이 매 분 run_theme_scan() 을 호출한다.

    토스 랭킹 조회
      → 직전 구간 대비 급등 판정
      → ThemeSnapshot 저장 (전 테마, 타임라인 소스)
      → 급등 테마의 구성 종목 조회 → 1등 종목 선정 → ThemeLeaderCandidate 저장
      → 급등테마주 전략을 켠 유저에게 TradingConfig 자동 등록

등록된 TradingConfig 는 기존 TradingEngine 의 분당 사이클에 그대로 편입되어,
진입은 급등테마주 전용 로직(entry.py), 청산은 Manual 기본 설정을 따른다.
"""

from __future__ import annotations

from datetime import date as date_cls, datetime, time as time_cls, timedelta

from django.db import transaction
from pytz import timezone as pytz_tz

from .config import (
    CANDIDATE_MAX_DEFAULT,
    CANDIDATE_REGISTER_UNTIL,
    LEADER_STORE_COUNT,
    MARKET_CLOSE,
    MARKET_OPEN,
    MOMENTUM_LOOKBACK_MINUTES,
    SLOT_MINUTES,
    SURGE_MAX_THEMES_PER_SLOT,
    SURGE_TOP_N,
)
from .detector import SurgeVerdict, detect_surge_themes
from .leader import LeaderCandidate, select_leaders
from .toss_client import TossThemeError, fetch_theme_ranking, fetch_theme_stocks

_KST = pytz_tz("Asia/Seoul")


def current_slot(now: datetime | None = None) -> time_cls:
    """현재 시각을 SLOT_MINUTES 슬롯 시각으로 내림한다 (09:00~15:30 범위로 clamp)."""
    moment = now or datetime.now(_KST)
    floored = moment.replace(
        minute=(moment.minute // SLOT_MINUTES) * SLOT_MINUTES, second=0, microsecond=0
    ).time()

    if floored < MARKET_OPEN:
        return MARKET_OPEN
    if floored > MARKET_CLOSE:
        return MARKET_CLOSE
    return floored


def run_theme_scan(now: datetime | None = None, force: bool = False) -> dict:
    """1회 스캔을 실행하고 요약 결과를 반환한다.

    Args:
        now:   기준 시각 (기본: 현재 KST)
        force: True면 개장일 여부와 무관하게 수집한다. 사용자가 화면에서
               직접 '지금 스캔'을 눌렀거나 테스트로 호출한 경우에만 쓴다.

    Returns:
        {"slot": "HH:MM", "themes": int, "surges": int,
         "candidates": int, "registered": int, "skipped": bool, "errors": [str]}
    """
    moment = now or datetime.now(_KST)
    today = moment.date()
    slot = current_slot(moment)
    errors: list[str] = []

    # 휴장일에 수집하면 전일 종가가 슬롯마다 복사돼 타임라인이 오염된다
    if not force and not _is_trading_day(today):
        print(f"[급등테마] {today} 는 개장일이 아님 — 스캔 생략")
        return _empty_result(slot, skipped=True)

    try:
        themes = fetch_theme_ranking(limit=SURGE_TOP_N)
    except TossThemeError as e:
        print(f"[급등테마] 랭킹 조회 실패: {e}")
        return _empty_result(slot, errors=[str(e)])

    verdicts = detect_surge_themes(themes, prev_rates=_previous_rates(today, slot))
    snapshots = _save_snapshots(today, slot, verdicts)

    surging = [v for v in verdicts if v.is_surge]
    surging.sort(key=lambda v: v.theme.fluctuation_rate, reverse=True)

    listed_codes = _listed_codes()
    candidate_count = 0
    selected: list[tuple[SurgeVerdict, LeaderCandidate]] = []

    for verdict in surging[:SURGE_MAX_THEMES_PER_SLOT]:
        try:
            stocks = fetch_theme_stocks(verdict.theme.tics_id)
        except TossThemeError as e:
            errors.append(f"{verdict.theme.name} 구성종목 조회 실패: {e}")
            continue

        leaders = select_leaders(stocks, top_n=LEADER_STORE_COUNT, listed_codes=listed_codes)
        if not leaders:
            continue

        snapshot = snapshots.get(verdict.theme.tics_id)
        if snapshot is None:
            continue

        _save_leaders(snapshot, leaders)
        candidate_count += len(leaders)
        selected.append((verdict, leaders[0]))

    registered = _register_candidates(today, slot, selected) if selected else 0

    print(
        f"[급등테마] {slot:%H:%M} 스캔 완료 — 테마 {len(verdicts)}개,"
        f" 급등 {len(surging)}개, 후보 {candidate_count}개, 신규등록 {registered}건"
    )

    return {
        "slot": slot.strftime("%H:%M"),
        "themes": len(verdicts),
        "surges": len(surging),
        "candidates": candidate_count,
        "registered": registered,
        "skipped": False,
        "errors": errors,
    }


def _empty_result(slot: time_cls, skipped: bool = False, errors: list[str] | None = None) -> dict:
    """수집을 수행하지 않았을 때의 표준 결과."""
    return {
        "slot": slot.strftime("%H:%M"),
        "themes": 0,
        "surges": 0,
        "candidates": 0,
        "registered": 0,
        "skipped": skipped,
        "errors": errors or [],
    }


def _is_trading_day(day: date_cls) -> bool:
    """국내 증시 개장일 여부. 조회 모듈이 없으면 주말만 제외한다."""
    try:
        from dolpha.kis.holiday import is_trading_day

        return is_trading_day(day)
    except Exception as e:
        print(f"[급등테마] 개장일 판정 실패 — 주말 기준 폴백: {e}")
        return day.weekday() < 5


# ──────────────────────────────────────────────────────────────
# 스냅샷 저장
# ──────────────────────────────────────────────────────────────

def _previous_rates(today: date_cls, slot: time_cls) -> dict[int, float]:
    """MOMENTUM_LOOKBACK_MINUTES 전 슬롯의 {tics_id: 등락률(%)} — 모멘텀 계산에 사용.

    슬롯 주기(SLOT_MINUTES)가 짧아져도 모멘텀 판정 구간은 고정 분 수를 유지한다.
    1분 슬롯에서 직전 1분과 비교하면 상승폭 요건이 사실상 5배로 강해져
    급등 테마가 거의 잡히지 않기 때문이다.
    """
    from myweb.models import ThemeSnapshot

    cutoff = (
        datetime.combine(today, slot) - timedelta(minutes=MOMENTUM_LOOKBACK_MINUTES)
    ).time()

    prev_slot = (
        ThemeSnapshot.objects.filter(date=today, slot_time__lte=cutoff)
        .order_by("-slot_time")
        .values_list("slot_time", flat=True)
        .first()
    )
    if prev_slot is None:
        return {}

    return dict(
        ThemeSnapshot.objects.filter(date=today, slot_time=prev_slot).values_list(
            "tics_id", "fluctuation_rate"
        )
    )


def _save_snapshots(
    today: date_cls, slot: time_cls, verdicts: list[SurgeVerdict]
) -> dict[int, object]:
    """스냅샷을 upsert 하고 {tics_id: ThemeSnapshot} 를 반환한다."""
    from myweb.models import ThemeSnapshot

    saved: dict[int, object] = {}

    for verdict in verdicts:
        theme = verdict.theme
        snapshot, _ = ThemeSnapshot.objects.update_or_create(
            date=today,
            slot_time=slot,
            tics_id=theme.tics_id,
            defaults={
                "theme_name": theme.name,
                "rank": theme.rank,
                "fluctuation_rate": theme.fluctuation_rate,
                "trading_value": theme.trading_value,
                "market_cap": theme.market_cap,
                "stock_count": theme.stock_count,
                "is_surge": verdict.is_surge,
                "surge_reason": verdict.reason[:200],
                "momentum": verdict.momentum,
            },
        )
        saved[theme.tics_id] = snapshot

    return saved


def _save_leaders(snapshot, leaders: list[LeaderCandidate]) -> None:
    """테마별 주도주 후보를 저장한다 (1위만 is_selected=True)."""
    from myweb.models import ThemeLeaderCandidate

    for leader in leaders:
        stock = leader.stock
        ThemeLeaderCandidate.objects.update_or_create(
            snapshot=snapshot,
            stock_code=stock.code,
            defaults={
                "date": snapshot.date,
                "slot_time": snapshot.slot_time,
                "tics_id": snapshot.tics_id,
                "theme_name": snapshot.theme_name,
                "stock_name": stock.name,
                "price": stock.price,
                "change_rate": stock.change_rate,
                "trading_value": stock.trading_value,
                "market_cap": stock.market_cap,
                "score": leader.score,
                "rank_in_theme": leader.rank_in_theme,
                "is_selected": leader.rank_in_theme == 1,
            },
        )


def _listed_codes() -> set[str]:
    """국내 상장 종목코드 집합 (Toss 응답에 섞인 비상장/해외 코드 제외용)."""
    from myweb.models import Company

    return set(Company.objects.values_list("code", flat=True))


# ──────────────────────────────────────────────────────────────
# 자동매매 후보 등록
# ──────────────────────────────────────────────────────────────

def _register_candidates(
    today: date_cls, slot: time_cls, selected: list[tuple[SurgeVerdict, LeaderCandidate]]
) -> int:
    """급등테마주 전략을 켠 유저에게 1등 종목을 TradingConfig 로 등록한다."""
    from myweb.models import TradingDefaults

    if slot > CANDIDATE_REGISTER_UNTIL:
        print(f"[급등테마] {slot:%H:%M} — 신규 후보 등록 마감 시각 경과, 등록 생략")
        return 0

    registered = 0
    for defaults in TradingDefaults.objects.filter(theme_surge_enabled=True).select_related("user"):
        for verdict, leader in selected:
            if not _meets_user_thresholds(defaults, verdict):
                continue
            if _register_for_user(defaults, verdict, leader):
                registered += 1

    return registered


def _meets_user_thresholds(defaults, verdict: SurgeVerdict) -> bool:
    """유저별 급등 기준(등락률·거래대금)을 만족하는 테마인지 확인한다.

    ThemeSnapshot.is_surge 는 전역 기준의 객관적 판정이고, 실제 후보 등록은
    유저가 설정한 기준을 한 번 더 통과해야 한다.
    """
    theme = verdict.theme
    if theme.fluctuation_rate < (defaults.theme_surge_min_fluctuation or 0):
        return False
    min_value = defaults.theme_surge_min_trading_value or 0
    # 거래대금 0 은 fallback 응답(미제공)이므로 조건을 적용하지 않는다
    return not (min_value > 0 and 0 < theme.trading_value < min_value)


def _exit_defaults(defaults) -> dict:
    """후보 등록 시 복사할 청산 관련 필드를 만든다.

    전용 청산(데이 트레이딩)을 쓰면 손절가·1T 폭은 진입 시점의 눌림 저점으로
    확정되므로 여기서는 비워 두고, 계좌 리스크 비율만 미리 넣는다.
    전용 청산을 끄면 기존처럼 Manual 기본값을 그대로 복사한다.
    """
    if getattr(defaults, "theme_surge_use_own_exit", True):
        return {
            "max_loss": defaults.theme_surge_max_loss,
            "stop_loss": None,          # 진입 시 눌림 저점으로 확정
            "take_profit": None,        # nT 분할 익절이 대신한다
            "pyramiding_count": 0,      # 당일 청산 전략이라 피라미딩은 쓰지 않는다
            "pyramiding_entries": [],
            "positions": [100],
        }

    return {
        "max_loss": defaults.manual_max_loss,
        "stop_loss": defaults.manual_stop_loss,
        "take_profit": defaults.manual_take_profit,
        "pyramiding_count": defaults.manual_pyramiding_count,
        "pyramiding_entries": list(defaults.manual_pyramiding_entries or []),
        "positions": list(defaults.manual_positions or [100]),
    }


def _register_for_user(defaults, verdict: SurgeVerdict, leader: LeaderCandidate) -> bool:
    """유저 1명에게 후보 1종목을 등록한다. 실제 등록 시 True."""
    from myweb.models import TradingConfig

    user = defaults.user
    stock = leader.stock
    max_candidates = defaults.theme_surge_max_candidates or CANDIDATE_MAX_DEFAULT

    try:
        with transaction.atomic():
            active = TradingConfig.objects.select_for_update().filter(
                user=user, is_active=True
            )

            # 같은 종목이 다른 전략으로 이미 돌고 있으면 중복 진입을 막는다
            if active.filter(stock_code=stock.code).exists():
                return False

            if active.filter(strategy_type="theme_surge").count() >= max_candidates:
                return False

            TradingConfig.objects.create(
                user=user,
                stock_code=stock.code,
                stock_name=stock.name,
                trading_mode="manual",
                strategy_type="theme_surge",
                **_exit_defaults(defaults),
                # 진입 시점은 1분봉 눌림목·돌파 로직이 판단하므로 가격 트리거는 쓰지 않는다
                entry_point=0,
                is_active=True,
            )
    except Exception as e:
        print(f"[급등테마] {user.username} - {stock.name} 후보 등록 실패: {e}")
        return False

    print(
        f"[급등테마] {user.username} ← {verdict.theme.name} 1등주"
        f" {stock.name}({stock.code}) 등록"
        f" (상승률 {stock.change_rate:+.2f}%, 거래대금 {stock.trading_value / 1e8:,.0f}억,"
        f" 점수 {leader.score:.3f})"
    )
    return True


# ──────────────────────────────────────────────────────────────
# 장 종료 정리
# ──────────────────────────────────────────────────────────────

def cleanup_stale_candidates() -> int:
    """장 마감 후, 끝내 진입하지 못한 급등테마주 후보 설정을 비활성화한다.

    후보는 그날 장중에만 유효하므로 다음 날 매매 대상에서 빼야 한다.
    다만 설정 행 자체는 남긴다 — 급등테마주 탭의 날짜별 이력과
    진입 판정 차트(1분봉)가 이 행을 근거로 조회되기 때문이다.
    실제 삭제는 사용자가 날짜 단위로 실행한다(purge_candidate_date).

    보유 포지션(체결된 매수 기록이 있는 설정)은 건드리지 않는다.
    청산은 Manual 기본 설정(손절/익절/트레일링스탑/분할익절)이 계속 담당한다.

    Returns:
        비활성화된 설정 수
    """
    from myweb.models import TradeEntry, TradingConfig

    stale = TradingConfig.objects.filter(strategy_type="theme_surge", is_active=True)
    deactivated = 0

    for config in stale:
        has_position = TradeEntry.objects.filter(
            user=config.user,
            stock_code=config.stock_code,
            trade_type="BUY",
            status="FILLED",
        ).exists()
        if has_position:
            continue

        config.is_active = False
        config.save(update_fields=["is_active"])
        deactivated += 1
        print(f"[급등테마] 미진입 후보 비활성화 — {config.user.username} / {config.stock_name}")

    return deactivated


def purge_candidate_date(user, target_date: date_cls) -> dict:
    """특정 날짜에 등록된 급등테마주 후보 설정과 그 근거 데이터를 삭제한다.

    삭제 대상은 세 가지다.
      1) 해당 날짜(KST)에 등록된 이 유저의 theme_surge TradingConfig
      2) 그 종목들의 해당 날짜 1분봉 (다른 날짜 분봉은 보존)
      3) 그 날짜의 진입 판정 이력(ThemeEntrySignal)

    보유 포지션이 남아 있는 설정은 청산 관리가 끊기므로 건너뛰고 그 수를 보고한다.

    Args:
        user:        삭제 대상 유저
        target_date: 등록일 (KST)

    Returns:
        {"date", "deleted_configs", "deleted_bars", "deleted_signals", "kept_positions"}
    """
    from myweb.models import (
        StockMinuteOhlcv,
        ThemeEntrySignal,
        TradeEntry,
        TradingConfig,
    )

    start = _KST.localize(datetime.combine(target_date, time_cls.min))
    end = _KST.localize(datetime.combine(target_date, time_cls.max))

    configs = TradingConfig.objects.filter(
        user=user, strategy_type="theme_surge", created_at__range=(start, end)
    )

    removable, kept = [], 0
    for config in configs:
        has_position = TradeEntry.objects.filter(
            user=user,
            stock_code=config.stock_code,
            trade_type="BUY",
            status="FILLED",
        ).exists()
        if has_position:
            kept += 1
            continue
        removable.append(config)

    codes = {config.stock_code for config in removable}
    deleted_bars = 0
    deleted_signals = 0

    with transaction.atomic():
        if codes:
            deleted_bars, _ = StockMinuteOhlcv.objects.filter(
                stock_code__in=codes, bar_datetime__range=(start, end)
            ).delete()
            deleted_signals, _ = ThemeEntrySignal.objects.filter(
                user=user, date=target_date, stock_code__in=codes
            ).delete()
        for config in removable:
            config.delete()

    print(
        f"[급등테마] {user.username} {target_date} 후보 정리 —"
        f" 설정 {len(removable)}건, 분봉 {deleted_bars}건, 판정 {deleted_signals}건 삭제"
        f" (보유 중 {kept}건 보존)"
    )

    return {
        "date": target_date.isoformat(),
        "deleted_configs": len(removable),
        "deleted_bars": deleted_bars,
        "deleted_signals": deleted_signals,
        "kept_positions": kept,
    }
