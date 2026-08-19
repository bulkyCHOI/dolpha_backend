"""급등 테마 판정.

토스 랭킹 스냅샷을 받아 '지금 급등 중인 테마'만 골라낸다.
단순히 등락률 상위를 취하는 대신, 아래 4가지를 함께 본다.

  1. 등락률   — 테마 전체가 의미 있게 올랐는가
  2. 거래대금 — 실제 수급이 붙었는가 (등락률만 높은 소형 테마 배제)
  3. 구성종목 — 종목 수가 너무 적어 1~2 종목 변동에 좌우되지 않는가
  4. 모멘텀   — 직전 5분 슬롯 대비 상승폭이 커지고 있는가 (이미 식은 테마 배제)

모멘텀은 직전 슬롯 스냅샷이 있을 때만 적용한다. 장 시작 첫 슬롯처럼 비교 대상이
없으면 1~3만으로 판정한다.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import (
    SURGE_MIN_FLUCTUATION_PCT,
    SURGE_MIN_MOMENTUM_PCT,
    SURGE_MIN_STOCK_COUNT,
    SURGE_MIN_TRADING_VALUE,
)
from .toss_client import ThemeRank


@dataclass(frozen=True)
class SurgeVerdict:
    """테마 1개에 대한 급등 판정 결과."""

    theme: ThemeRank
    is_surge: bool
    momentum: float          # 직전 슬롯 대비 등락률 변화(%p). 비교 대상 없으면 0.0
    reason: str              # 판정 근거 (충족 시 요약, 미충족 시 탈락 사유)


def detect_surge_themes(
    themes: list[ThemeRank],
    prev_rates: dict[int, float] | None = None,
    min_fluctuation: float = SURGE_MIN_FLUCTUATION_PCT,
    min_trading_value: int = SURGE_MIN_TRADING_VALUE,
) -> list[SurgeVerdict]:
    """테마 랭킹에서 급등 테마를 판정한다.

    Args:
        themes:            토스 랭킹 스냅샷
        prev_rates:        직전 슬롯의 {tics_id: 등락률(%)} — 모멘텀 계산용
        min_fluctuation:   등락률 하한(%)
        min_trading_value: 거래대금 하한(원). 0 이면 거래대금 조건 미적용

    Returns:
        입력 순서를 유지한 SurgeVerdict 리스트 (급등/비급등 모두 포함).
        스냅샷 저장 시 전체를 기록하고, 후보 선정은 is_surge 인 것만 사용한다.
    """
    previous = prev_rates or {}
    verdicts: list[SurgeVerdict] = []

    for theme in themes:
        prev_rate = previous.get(theme.tics_id)
        momentum = theme.fluctuation_rate - prev_rate if prev_rate is not None else 0.0
        is_surge, reason = _judge(theme, momentum, prev_rate, min_fluctuation, min_trading_value)
        verdicts.append(
            SurgeVerdict(theme=theme, is_surge=is_surge, momentum=momentum, reason=reason)
        )

    return verdicts


def _judge(
    theme: ThemeRank,
    momentum: float,
    prev_rate: float | None,
    min_fluctuation: float,
    min_trading_value: int,
) -> tuple[bool, str]:
    """단일 테마 급등 여부와 사유를 반환한다."""
    if theme.fluctuation_rate < min_fluctuation:
        return False, f"등락률 미달 ({theme.fluctuation_rate:+.2f}% < {min_fluctuation:.1f}%)"

    if theme.stock_count < SURGE_MIN_STOCK_COUNT:
        return False, f"구성종목 부족 ({theme.stock_count}개 < {SURGE_MIN_STOCK_COUNT}개)"

    # fallback 응답은 거래대금을 제공하지 않으므로(0) 조건을 건너뛴다
    if min_trading_value > 0 and 0 < theme.trading_value < min_trading_value:
        return False, (
            f"거래대금 미달 ({theme.trading_value / 1e8:,.0f}억 "
            f"< {min_trading_value / 1e8:,.0f}억)"
        )

    if prev_rate is not None and momentum < SURGE_MIN_MOMENTUM_PCT:
        return False, f"모멘텀 둔화 ({momentum:+.2f}%p < {SURGE_MIN_MOMENTUM_PCT:+.2f}%p)"

    momentum_text = f", 모멘텀 {momentum:+.2f}%p" if prev_rate is not None else ""
    return True, (
        f"등락률 {theme.fluctuation_rate:+.2f}%, "
        f"거래대금 {theme.trading_value / 1e8:,.0f}억{momentum_text}"
    )
