"""1분봉 패턴 분석 — 스윙 고점(전고점), 눌림목, 돌파.

DB 접근이나 API 호출 없이 분봉 리스트만 받아 계산하는 순수 함수 모음.
덕분에 단위 테스트가 쉽고, 진입 로직(entry.py)은 판정 조립에만 집중할 수 있다.

분봉 형식:
    {"open": float, "high": float, "low": float, "close": float, "volume": float}
시간 오름차순으로 정렬되어 있다고 가정한다 (마지막 원소가 가장 최근 봉).
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import (
    BREAKOUT_BUFFER_PCT,
    BREAKOUT_VOLUME_AVG_BARS,
    BREAKOUT_VOLUME_RATIO_MIN,
    PIVOT_WINDOW,
    PULLBACK_MAX_PCT,
    PULLBACK_MIN_BARS,
    PULLBACK_MIN_PCT,
    PULLBACK_VOLUME_RATIO_MAX,
)


@dataclass(frozen=True)
class SwingHigh:
    """확정된 스윙 고점(전고점)."""

    index: int      # bars 내 위치
    price: float    # 고가


@dataclass(frozen=True)
class Pullback:
    """전고점 이후의 눌림목."""

    low: float          # 눌림 저점
    depth_pct: float    # 전고점 대비 하락률(%)
    bar_count: int      # 눌림 구간 봉 수
    volume_ratio: float  # 눌림 구간 평균 거래량 / 상승 구간 평균 거래량
    is_valid: bool
    reason: str


@dataclass(frozen=True)
class Breakout:
    """전고점 돌파."""

    price: float         # 돌파 판정에 쓴 현재가
    threshold: float     # 돌파 기준선 (전고점 + 여유)
    volume_ratio: float  # 현재 봉 거래량 / 최근 평균 거래량
    is_valid: bool
    reason: str


def find_last_swing_high(
    bars: list[dict], window: int = PIVOT_WINDOW, exclude_last: int = 1
) -> SwingHigh | None:
    """가장 최근에 확정된 스윙 고점을 찾는다.

    좌우 `window` 개 봉보다 고가가 높거나 같고, 최소 한쪽보다는 확실히 높은 봉을
    스윙 고점으로 본다. 아직 좌우 봉이 채워지지 않은 최근 `exclude_last` 개 봉은
    확정되지 않았으므로 후보에서 제외한다.

    Returns:
        가장 최근 스윙 고점. 없으면 None.
    """
    if not bars:
        return None

    last_candidate = len(bars) - exclude_last - window
    for i in range(last_candidate, window - 1, -1):
        high = bars[i]["high"]
        left = [bars[j]["high"] for j in range(i - window, i)]
        right = [bars[j]["high"] for j in range(i + 1, i + 1 + window)]
        if not left or not right:
            continue
        if high >= max(left) and high >= max(right) and high > min(left + right):
            return SwingHigh(index=i, price=high)

    return None


def analyze_pullback(bars: list[dict], swing: SwingHigh) -> Pullback:
    """전고점 이후 구간이 눌림목 요건을 만족하는지 판정한다.

    요건:
      1. 깊이  — 전고점 대비 PULLBACK_MIN_PCT ~ PULLBACK_MAX_PCT 하락
      2. 기간  — 최소 PULLBACK_MIN_BARS 개 봉
      3. 거래량 — 눌림 구간 평균 거래량이 직전 상승 구간보다 감소 (매물 소화)
    """
    # 현재(마지막) 봉은 돌파 판정용이므로 눌림 구간에서 제외
    pullback_bars = bars[swing.index + 1 : len(bars) - 1]
    if len(pullback_bars) < PULLBACK_MIN_BARS:
        return Pullback(0.0, 0.0, len(pullback_bars), 0.0, False,
                        f"눌림 구간 부족 ({len(pullback_bars)}봉 < {PULLBACK_MIN_BARS}봉)")

    low = min(bar["low"] for bar in pullback_bars)
    if swing.price <= 0:
        return Pullback(low, 0.0, len(pullback_bars), 0.0, False, "전고점 가격 이상")

    depth_pct = (swing.price - low) / swing.price * 100.0

    rise_bars = bars[max(0, swing.index - len(pullback_bars)) : swing.index + 1]
    volume_ratio = _volume_ratio(pullback_bars, rise_bars)

    if depth_pct < PULLBACK_MIN_PCT:
        return Pullback(low, depth_pct, len(pullback_bars), volume_ratio, False,
                        f"눌림 부족 ({depth_pct:.2f}% < {PULLBACK_MIN_PCT}%)")
    if depth_pct > PULLBACK_MAX_PCT:
        return Pullback(low, depth_pct, len(pullback_bars), volume_ratio, False,
                        f"눌림 과다 ({depth_pct:.2f}% > {PULLBACK_MAX_PCT}%)")
    if volume_ratio > PULLBACK_VOLUME_RATIO_MAX:
        return Pullback(low, depth_pct, len(pullback_bars), volume_ratio, False,
                        f"눌림 구간 거래량 미감소 (비율 {volume_ratio:.2f} > {PULLBACK_VOLUME_RATIO_MAX})")

    return Pullback(low, depth_pct, len(pullback_bars), volume_ratio, True,
                    f"눌림 {depth_pct:.2f}%/{len(pullback_bars)}봉, 거래량비 {volume_ratio:.2f}")


def analyze_breakout(bars: list[dict], swing: SwingHigh, current_price: float) -> Breakout:
    """현재가가 전고점을 유효하게 돌파했는지 판정한다.

    요건:
      1. 가격  — 현재가 > 전고점 × (1 + BREAKOUT_BUFFER_PCT%)  (허수 돌파 배제)
      2. 거래량 — 현재 봉 거래량 ≥ 최근 평균 × BREAKOUT_VOLUME_RATIO_MIN
    """
    threshold = swing.price * (1 + BREAKOUT_BUFFER_PCT / 100.0)
    current_volume = bars[-1]["volume"] if bars else 0.0

    recent = bars[-(BREAKOUT_VOLUME_AVG_BARS + 1) : -1]
    avg_volume = sum(bar["volume"] for bar in recent) / len(recent) if recent else 0.0
    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0.0

    if current_price <= threshold:
        return Breakout(current_price, threshold, volume_ratio, False,
                        f"전고점 미돌파 ({current_price:,.0f} ≤ {threshold:,.0f})")
    if volume_ratio < BREAKOUT_VOLUME_RATIO_MIN:
        return Breakout(current_price, threshold, volume_ratio, False,
                        f"돌파 거래량 부족 (비율 {volume_ratio:.2f} < {BREAKOUT_VOLUME_RATIO_MIN})")

    return Breakout(current_price, threshold, volume_ratio, True,
                    f"전고점 {swing.price:,.0f} 돌파, 거래량비 {volume_ratio:.2f}")


def _volume_ratio(pullback_bars: list[dict], rise_bars: list[dict]) -> float:
    """눌림 구간 평균 거래량 / 상승 구간 평균 거래량. 산출 불가 시 0.0."""
    if not pullback_bars or not rise_bars:
        return 0.0
    rise_avg = sum(bar["volume"] for bar in rise_bars) / len(rise_bars)
    if rise_avg <= 0:
        return 0.0
    pullback_avg = sum(bar["volume"] for bar in pullback_bars) / len(pullback_bars)
    return pullback_avg / rise_avg
