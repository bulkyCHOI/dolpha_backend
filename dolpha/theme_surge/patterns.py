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
    MORNING_BREAKOUT_VOLUME_RATIO_MIN,
    MORNING_PEAK_MIN_RISE_PCT,
    MORNING_PULLBACK_MAX_BARS,
    MORNING_PULLBACK_MAX_PCT,
    MORNING_PULLBACK_MIN_BARS,
    MORNING_PULLBACK_MIN_PCT,
    MORNING_PULLBACK_VOLUME_RATIO_MAX,
    PIVOT_MIN_LEFT_BARS,
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

    좌측은 장 시작으로 잘려 `window` 개를 채우지 못할 수 있다. 이때 후보에서 빼면
    장 초반(09:00~09:19)에 만들어진 고점이 그날 내내 전고점이 되지 못하므로,
    좌측이 `PIVOT_MIN_LEFT_BARS` 개 이상이면 있는 만큼만 비교한다.
    반면 우측은 시간이 지나면 채워지는 값이라, 다 채워지기 전에 인정하면 아직
    확정되지 않은 고점을 전고점으로 쓰게 되므로 `window` 개를 그대로 요구한다.

    Returns:
        가장 최근 스윙 고점. 없으면 None.
    """
    if not bars:
        return None

    last_candidate = len(bars) - exclude_last - window
    for i in range(last_candidate, PIVOT_MIN_LEFT_BARS - 1, -1):
        high = bars[i]["high"]
        left = [bars[j]["high"] for j in range(max(0, i - window), i)]
        right = [bars[j]["high"] for j in range(i + 1, i + 1 + window)]
        if len(left) < PIVOT_MIN_LEFT_BARS or len(right) < window:
            continue
        if high >= max(left) and high >= max(right) and high > min(left + right):
            return SwingHigh(index=i, price=high)

    return None


def find_morning_high(bars: list[dict], exclude_last: int = 1) -> SwingHigh | None:
    """시초가 대비 +2.0% 이상 오른 당일 고점(모닝 스윙 고점)을 찾는다.

    고점 이후 최소 MORNING_PULLBACK_MIN_BARS(2)봉 동안 고가를 미경신한 고점을 확정한다.
    """
    if not bars or len(bars) < 1:
        return None

    max_candidate_idx = len(bars) - 1 - exclude_last - MORNING_PULLBACK_MIN_BARS
    if max_candidate_idx < 0:
        return None

    open_price = bars[0].get("open", 0.0)
    if open_price <= 0:
        return None

    # 0부터 max_candidate_idx 범위 내에서 최고 high를 가진 봉 탐색
    peak_idx = -1
    peak_high = -1.0
    for i in range(max_candidate_idx + 1):
        h = bars[i]["high"]
        if h > peak_high:
            peak_high = h
            peak_idx = i

    if peak_idx < 0 or peak_high <= 0:
        return None

    # peak_idx 이후부터 len(bars) - 1 - exclude_last 까지 모든 봉의 high가 peak_high 이하인지 확인 (미경신)
    eval_end = len(bars) - exclude_last
    for j in range(peak_idx + 1, eval_end):
        if bars[j]["high"] > peak_high:
            return None

    # 시초가 대비 최소 상승률 검사
    min_required_price = open_price * (1 + MORNING_PEAK_MIN_RISE_PCT / 100.0)
    if peak_high < min_required_price:
        return None

    return SwingHigh(index=peak_idx, price=peak_high)


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


def analyze_morning_pullback(bars: list[dict], swing: SwingHigh) -> Pullback:
    """모닝 세션 눌림목 요건을 판정한다.

    요건:
      1. 기간  — MORNING_PULLBACK_MIN_BARS(2) ~ MORNING_PULLBACK_MAX_BARS(12) 봉
      2. 시초가 — 눌림 저점이 당일 시초가(첫 봉 open) 이상 (시초가 사수)
      3. 깊이  — 전고점 대비 MORNING_PULLBACK_MIN_PCT(1.0%) ~ MORNING_PULLBACK_MAX_PCT(4.5%) 하락
      4. 거래량 — 눌림 구간 평균 거래량이 직전 상승 구간 대비 감소 (비율 <= MORNING_PULLBACK_VOLUME_RATIO_MAX(0.85))
    """
    pullback_bars = bars[swing.index + 1 : len(bars) - 1]
    if len(pullback_bars) < MORNING_PULLBACK_MIN_BARS:
        return Pullback(
            0.0, 0.0, len(pullback_bars), 0.0, False,
            f"모닝 눌림 구간 부족 ({len(pullback_bars)}봉 < {MORNING_PULLBACK_MIN_BARS}봉)",
        )

    low = min(bar["low"] for bar in pullback_bars)
    if swing.price <= 0:
        return Pullback(low, 0.0, len(pullback_bars), 0.0, False, "전고점 가격 이상")

    depth_pct = (swing.price - low) / swing.price * 100.0

    rise_bars = bars[max(0, swing.index - len(pullback_bars)) : swing.index + 1]
    volume_ratio = _volume_ratio(pullback_bars, rise_bars)

    if len(pullback_bars) > MORNING_PULLBACK_MAX_BARS:
        return Pullback(
            low, depth_pct, len(pullback_bars), volume_ratio, False,
            f"모닝 눌림 기간 초과 ({len(pullback_bars)}봉 > {MORNING_PULLBACK_MAX_BARS}봉)",
        )

    if bars and low < bars[0]["open"]:
        return Pullback(
            low, depth_pct, len(pullback_bars), volume_ratio, False,
            f"시초가 이탈 ({low:,.0f} < {bars[0]['open']:,.0f})",
        )

    if depth_pct < MORNING_PULLBACK_MIN_PCT:
        return Pullback(
            low, depth_pct, len(pullback_bars), volume_ratio, False,
            f"모닝 눌림 부족 ({depth_pct:.2f}% < {MORNING_PULLBACK_MIN_PCT}%)",
        )

    if depth_pct > MORNING_PULLBACK_MAX_PCT:
        return Pullback(
            low, depth_pct, len(pullback_bars), volume_ratio, False,
            f"모닝 눌림 과다 ({depth_pct:.2f}% > {MORNING_PULLBACK_MAX_PCT}%)",
        )

    if volume_ratio > MORNING_PULLBACK_VOLUME_RATIO_MAX:
        return Pullback(
            low, depth_pct, len(pullback_bars), volume_ratio, False,
            f"모닝 눌림 구간 거래량 미감소 (비율 {volume_ratio:.2f} > {MORNING_PULLBACK_VOLUME_RATIO_MAX})",
        )

    return Pullback(
        low, depth_pct, len(pullback_bars), volume_ratio, True,
        f"모닝 눌림 {depth_pct:.2f}%/{len(pullback_bars)}봉, 거래량비 {volume_ratio:.2f}",
    )


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


def analyze_morning_breakout(bars: list[dict], swing: SwingHigh, current_price: float) -> Breakout:
    """모닝 세션 전고점 돌파 여부를 판정한다.

    요건:
      1. 가격   — 현재가 > 전고점 × (1 + BREAKOUT_BUFFER_PCT%)
      2. 거래량 — (현재 거래량 >= 눌림 평균 거래량 * MORNING_BREAKOUT_VOLUME_RATIO_MIN) 또는
                 (최근 5봉 평균 대비 >= 1.5)
    """
    threshold = swing.price * (1 + BREAKOUT_BUFFER_PCT / 100.0)
    current_volume = bars[-1]["volume"] if bars else 0.0

    pullback_bars = bars[swing.index + 1 : len(bars) - 1]
    pb_avg = sum(b["volume"] for b in pullback_bars) / len(pullback_bars) if pullback_bars else 0.0

    recent = bars[-6:-1] if len(bars) >= 6 else bars[:-1]
    recent_avg = sum(b["volume"] for b in recent) / len(recent) if recent else 0.0

    volume_ratio = current_volume / pb_avg if pb_avg > 0 else (current_volume / recent_avg if recent_avg > 0 else 0.0)

    if current_price <= threshold:
        return Breakout(
            current_price, threshold, volume_ratio, False,
            f"전고점 미돌파 ({current_price:,.0f} <= {threshold:,.0f})",
        )

    is_vol_ok = (pb_avg > 0 and current_volume >= pb_avg * MORNING_BREAKOUT_VOLUME_RATIO_MIN) or (
        recent_avg > 0 and current_volume >= recent_avg * 1.5
    )
    if not is_vol_ok:
        return Breakout(
            current_price, threshold, volume_ratio, False,
            f"모닝 돌파 거래량 부족 (비율 {volume_ratio:.2f} < {MORNING_BREAKOUT_VOLUME_RATIO_MIN})",
        )

    return Breakout(
        current_price, threshold, volume_ratio, True,
        f"모닝 전고점 {swing.price:,.0f} 돌파, 거래량비 {volume_ratio:.2f}",
    )


def _volume_ratio(pullback_bars: list[dict], rise_bars: list[dict]) -> float:
    """눌림 구간 평균 거래량 / 상승 구간 평균 거래량. 산출 불가 시 0.0."""
    if not pullback_bars or not rise_bars:
        return 0.0
    rise_avg = sum(bar["volume"] for bar in rise_bars) / len(rise_bars)
    if rise_avg <= 0:
        return 0.0
    pullback_avg = sum(bar["volume"] for bar in pullback_bars) / len(pullback_bars)
    return pullback_avg / rise_avg
