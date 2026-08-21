"""급등테마주 전용 청산 규칙 — 데이 트레이딩 기준.

Manual 기본값(손절 8% / 트레일링 트리거 8%)은 주 단위 스윙에 맞춰진 값이라
1분봉 눌림목·돌파로 진입하는 이 전략과 시간축이 맞지 않는다. 그래서 청산도
진입 신호가 만들어 낸 좌표(눌림 저점, 돌파가)를 그대로 기준으로 삼는다.

    T (1T) = 돌파가 - 눌림 저점        ← 진입 신호가 만든 측정 상승폭
    손절가 = 눌림 저점                  ← 진입 근거가 깨지는 지점
    목표가 = 평단 + n × T              ← n 은 유저가 차수별로 설정

손절폭이 곧 1T 근처이므로 nT 는 사실상 nR 이 되고, 배팅 사이즈를
"손절 시 계좌의 X%" 로 잡으면 손익비 계산이 그대로 성립한다.

이 모듈은 DB 조회와 순수 판정만 담당하고, 주문 실행은 TradingEngine 이 한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as date_cls, datetime, time as time_cls, timedelta

from .config import (
    DEFAULT_EXIT_STAGES,
    DEFAULT_FORCE_EXIT_TIME,
    DEFAULT_MAX_LOSS_PCT,
    DEFAULT_TRAILING_BAR_COUNT,
    DEFAULT_TRAILING_BAR_UNIT,
    DEFAULT_TRAILING_START_T,
    PULLBACK_MAX_PCT,
    TRAILING_BAR_UNIT_MINUTES,
)


@dataclass(frozen=True)
class ExitSettings:
    """유저의 급등테마주 청산 설정 스냅샷 (TradingDefaults 에서 읽어 정규화)."""

    use_own_exit: bool
    max_loss_pct: float
    stages: tuple[tuple[float, float], ...]   # ((T배수, 청산비율%), ...) T 오름차순
    use_trailing: bool
    trailing_start_t: float
    trailing_bar_unit: str
    trailing_bar_count: int
    force_exit_enabled: bool
    force_exit_time: time_cls


@dataclass(frozen=True)
class ExitDecision:
    """청산 판정 결과.

    sell_pct 가 100 이면 전량, 그 미만이면 부분 매도.
    stage 는 분할 익절 차수(1부터). 전량 청산 사유면 None.
    """

    should_exit: bool
    sell_pct: float = 0.0
    reason: str = ""
    stage: int | None = None


# ──────────────────────────────────────────────────────────────
# 설정 읽기
# ──────────────────────────────────────────────────────────────

def normalize_stages(raw: object) -> tuple[tuple[float, float], ...]:
    """설정에 저장된 분할 익절 차수를 (T배수, 청산비율) 튜플로 정규화한다.

    잘못된 항목(숫자 아님, 0 이하, 비율 범위 밖)은 조용히 버리지 않고 제외한 뒤
    남은 것만 T 오름차순으로 정렬한다. 전부 버려지면 빈 튜플을 반환하고,
    호출부는 분할 익절 없이 트레일링/손절만으로 운용한다.
    """
    if not isinstance(raw, (list, tuple)):
        return ()

    cleaned: list[tuple[float, float]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            t_mult = float(item.get("t"))
            sell_pct = float(item.get("sell_pct"))
        except (TypeError, ValueError):
            continue
        if t_mult <= 0 or not (0 < sell_pct <= 100):
            continue
        cleaned.append((t_mult, sell_pct))

    return tuple(sorted(cleaned, key=lambda pair: pair[0]))


def load_exit_settings(defaults) -> ExitSettings:
    """TradingDefaults 에서 급등테마주 청산 설정을 읽는다.

    설정 객체가 없거나 필드가 비어 있으면 config.py 기본값으로 대체한다.
    """
    if defaults is None:
        stages = normalize_stages(DEFAULT_EXIT_STAGES)
        return ExitSettings(
            use_own_exit=True,
            max_loss_pct=DEFAULT_MAX_LOSS_PCT,
            stages=stages,
            use_trailing=True,
            trailing_start_t=DEFAULT_TRAILING_START_T,
            trailing_bar_unit=DEFAULT_TRAILING_BAR_UNIT,
            trailing_bar_count=DEFAULT_TRAILING_BAR_COUNT,
            force_exit_enabled=True,
            force_exit_time=DEFAULT_FORCE_EXIT_TIME,
        )

    stages = normalize_stages(getattr(defaults, "theme_surge_exit_stages", None))
    unit = getattr(defaults, "theme_surge_trailing_bar_unit", None)
    if unit not in TRAILING_BAR_UNIT_MINUTES:
        unit = DEFAULT_TRAILING_BAR_UNIT

    return ExitSettings(
        use_own_exit=bool(getattr(defaults, "theme_surge_use_own_exit", True)),
        max_loss_pct=float(
            getattr(defaults, "theme_surge_max_loss", None) or DEFAULT_MAX_LOSS_PCT
        ),
        stages=stages,
        use_trailing=bool(getattr(defaults, "theme_surge_use_trailing", True)),
        trailing_start_t=float(
            getattr(defaults, "theme_surge_trailing_start_t", None)
            or DEFAULT_TRAILING_START_T
        ),
        trailing_bar_unit=unit,
        trailing_bar_count=max(
            1, int(getattr(defaults, "theme_surge_trailing_bar_count", None) or DEFAULT_TRAILING_BAR_COUNT)
        ),
        force_exit_enabled=bool(getattr(defaults, "theme_surge_force_exit_enabled", True)),
        force_exit_time=getattr(defaults, "theme_surge_force_exit_time", None)
        or DEFAULT_FORCE_EXIT_TIME,
    )


# ──────────────────────────────────────────────────────────────
# 진입 시점 좌표 확정
# ──────────────────────────────────────────────────────────────

def derive_entry_levels(
    pullback_low: float | None, breakout_price: float, entry_price: float
) -> tuple[float, float] | None:
    """진입 신호에서 손절가와 1T 폭을 뽑는다.

    Args:
        pullback_low:   진입 신호의 눌림 저점 (손절가)
        breakout_price: 돌파 판정에 쓴 가격
        entry_price:    실제 진입 가격 (fallback 계산 기준)

    Returns:
        (손절가, 1T 폭). 계산 불가 시 None.
    """
    if entry_price <= 0:
        return None

    low = pullback_low or 0.0
    if low <= 0 or low >= breakout_price:
        # 눌림 저점을 복원하지 못한 경우: 눌림목 최대 깊이를 손절폭으로 쓰는 보수적 대체
        low = entry_price * (1 - PULLBACK_MAX_PCT / 100.0)

    t_value = breakout_price - low
    if t_value <= 0:
        t_value = entry_price - low
    if t_value <= 0:
        return None

    return low, t_value


def stop_loss_pct(entry_price: float, stop_price: float) -> float | None:
    """손절가를 진입가 대비 하락률(%)로 환산한다. 포지션 사이징에 쓴다."""
    if entry_price <= 0 or stop_price <= 0 or stop_price >= entry_price:
        return None
    return (entry_price - stop_price) / entry_price * 100.0


# ──────────────────────────────────────────────────────────────
# 청산 판정
# ──────────────────────────────────────────────────────────────

def next_stage(
    settings: ExitSettings,
    completed_stages: list[int],
    avg_price: float,
    t_value: float,
    current_price: float,
) -> tuple[int, float, float] | None:
    """도달했지만 아직 실행하지 않은 분할 익절 차수 중 가장 높은 차수를 찾는다.

    급등 구간에서 1분 사이에 2차·3차를 한꺼번에 통과할 수 있으므로,
    가장 높은 도달 차수를 골라 그 아래 차수는 함께 완료 처리한다.

    Returns:
        (차수 번호, 청산 비율%, 목표가). 도달한 차수가 없으면 None.
    """
    if not settings.stages or t_value <= 0 or avg_price <= 0:
        return None

    reached: tuple[int, float, float] | None = None
    for index, (t_mult, sell_pct) in enumerate(settings.stages, start=1):
        target = avg_price + t_value * t_mult
        if current_price >= target and index not in completed_stages:
            reached = (index, sell_pct, target)

    return reached


def trailing_stop_line(
    settings: ExitSettings, bars: list[dict], exclude_last: int = 1
) -> float | None:
    """직전 N봉의 최저가를 트레일링 스탑 라인으로 계산한다.

    현재 진행 중인 봉은 아직 저가가 확정되지 않았으므로 기본적으로 제외한다.
    (제외하지 않으면 자기 자신의 저가에 걸려 즉시 청산되는 자기참조가 생긴다.)
    """
    if not bars:
        return None

    closed = bars[: len(bars) - exclude_last] if exclude_last else bars
    window = closed[-settings.trailing_bar_count :]
    if not window:
        return None

    lows = [bar["low"] for bar in window if bar.get("low")]
    return min(lows) if lows else None


def evaluate_exit(
    settings: ExitSettings,
    *,
    avg_price: float,
    current_price: float,
    stop_price: float | None,
    t_value: float | None,
    peak_price: float | None,
    completed_stages: list[int],
    trailing_started: bool,
    trailing_line: float | None,
    now: datetime,
) -> ExitDecision:
    """급등테마주 포지션의 청산 여부를 판정한다.

    우선순위:
      1. 당일 강제 청산 시각 도달        → 전량
      2. 손절 (눌림 저점 이탈)           → 전량
      3. 트레일링 스탑 (nT 초과 후 추적) → 잔량 전량
      4. 분할 익절 (nT 도달)             → 차수별 비율
    """
    if avg_price <= 0 or current_price <= 0:
        return ExitDecision(False)

    # 1. 당일 강제 청산 — 오버나이트 갭 리스크 차단
    if settings.force_exit_enabled and now.time() >= settings.force_exit_time:
        return ExitDecision(True, 100.0, f"당일 강제청산({settings.force_exit_time:%H:%M})")

    # 2. 손절 — 진입 근거(눌림목)가 깨진 지점
    if stop_price and current_price <= stop_price:
        loss_pct = (current_price - avg_price) / avg_price * 100.0
        return ExitDecision(True, 100.0, f"손절(눌림저점 {stop_price:,.0f} 이탈, {loss_pct:+.2f}%)")

    # 3. 트레일링 스탑 — nT 초과 후부터 직전 N봉 최저점 추적
    if settings.use_trailing and t_value and t_value > 0:
        trigger_price = avg_price + t_value * settings.trailing_start_t
        activated = trailing_started or (
            peak_price is not None and peak_price >= trigger_price
        )
        if activated and trailing_line and current_price <= trailing_line:
            unit_label = TRAILING_BAR_UNIT_MINUTES[settings.trailing_bar_unit][1]
            return ExitDecision(
                True,
                100.0,
                f"트레일링스탑({unit_label} {settings.trailing_bar_count}봉 최저 "
                f"{trailing_line:,.0f} 이탈)",
            )

    # 4. 분할 익절 — nT 도달
    if t_value and t_value > 0:
        stage = next_stage(settings, completed_stages, avg_price, t_value, current_price)
        if stage is not None:
            index, sell_pct, target = stage
            t_mult = settings.stages[index - 1][0]
            return ExitDecision(
                True,
                sell_pct,
                f"{index}차 익절({t_mult:g}T={target:,.0f} 도달, {sell_pct:g}%)",
                stage=index,
            )

    return ExitDecision(False)


def is_trailing_triggered(
    settings: ExitSettings,
    avg_price: float,
    t_value: float | None,
    peak_price: float | None,
) -> bool:
    """트레일링 추적을 시작할 수준(nT 초과)에 도달했는지 판정한다."""
    if not settings.use_trailing or not t_value or t_value <= 0:
        return False
    if avg_price <= 0 or peak_price is None:
        return False
    return peak_price >= avg_price + t_value * settings.trailing_start_t


# ──────────────────────────────────────────────────────────────
# 봉 조회 (트레일링 최저점용)
# ──────────────────────────────────────────────────────────────

def fetch_bars(stock_code: str, settings: ExitSettings, now: datetime) -> list[dict]:
    """트레일링 판정에 쓸 봉 목록을 설정된 단위로 조회한다.

    1분봉/5분봉은 엔진이 매 분 적재하는 `stock_minute_ohlcv` 를 집계해 쓰고,
    일봉은 기존 일별 OHLCV 를 쓴다.
    """
    if settings.trailing_bar_unit == "1d":
        return _fetch_daily_bars(stock_code, settings.trailing_bar_count + 1, now.date())

    minutes = TRAILING_BAR_UNIT_MINUTES[settings.trailing_bar_unit][0]
    return _fetch_minute_bars(stock_code, settings.trailing_bar_count + 1, minutes, now)


def _fetch_minute_bars(
    stock_code: str, need: int, group_minutes: int, now: datetime
) -> list[dict]:
    """당일 1분봉을 group_minutes 단위로 묶어 최근 need 개를 반환한다."""
    from myweb.models import StockMinuteOhlcv

    # 필요한 봉 수 × 단위 + 여유. 결측(거래 없는 분)을 감안해 넉넉히 조회한다.
    lookback_minutes = need * group_minutes * 3
    since = now - timedelta(minutes=lookback_minutes)

    rows = list(
        StockMinuteOhlcv.objects.filter(
            stock_code=stock_code, bar_datetime__gte=since
        )
        .order_by("bar_datetime")
        .values("bar_datetime", "open", "high", "low", "close", "volume")
    )
    if not rows:
        return []

    if group_minutes <= 1:
        return [_to_bar(row) for row in rows]

    grouped: dict[datetime, dict] = {}
    for row in rows:
        moment = row["bar_datetime"]
        bucket = moment.replace(
            minute=(moment.minute // group_minutes) * group_minutes,
            second=0,
            microsecond=0,
        )
        current = grouped.get(bucket)
        if current is None:
            grouped[bucket] = {
                "datetime": bucket,
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
            }
            continue
        current["high"] = max(current["high"], row["high"])
        current["low"] = min(current["low"], row["low"])
        current["close"] = row["close"]
        current["volume"] += row["volume"]

    return [grouped[key] for key in sorted(grouped)]


def _fetch_daily_bars(stock_code: str, need: int, today: date_cls) -> list[dict]:
    """최근 need 개 일봉을 반환한다 (거래정지일 제외)."""
    from myweb.models import StockOHLCV

    rows = list(
        StockOHLCV.objects.filter(code_id=stock_code, date__lte=today, open__gt=0)
        .order_by("-date")
        .values("date", "open", "high", "low", "close", "volume")[:need]
    )
    return [
        {
            "datetime": row["date"],
            "open": row["open"],
            "high": row["high"],
            "low": row["low"],
            "close": row["close"],
            "volume": row["volume"],
        }
        for row in reversed(rows)
    ]


def _to_bar(row: dict) -> dict:
    return {
        "datetime": row["bar_datetime"],
        "open": row["open"],
        "high": row["high"],
        "low": row["low"],
        "close": row["close"],
        "volume": row["volume"],
    }
