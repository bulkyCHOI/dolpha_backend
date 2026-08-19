"""급등 테마의 1등 종목(주도주) 선정.

거래대금과 상승률을 테마 내에서 각각 0~1 로 정규화(min-max)한 뒤 가중 합산해
복합 점수를 만든다. 절대값을 그대로 쓰면 단위가 달라(원 vs %) 한쪽이 지배하므로
반드시 테마 내부 상대 비교로 환산한다.

    score = w_value × norm(거래대금) + w_change × norm(상승률)

선정 전에 다음 종목은 후보에서 제외한다.

  - 국내 상장 종목 마스터(Company)에 없는 코드 (ETF/ETN/해외 종목 등)
  - 상승률·거래대금·시가총액이 하한 미달 → 유동성 부족
  - 상승률이 상한 초과 → 상한가 근접 구간 추격 매수 방지
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import (
    LEADER_MAX_CHANGE_RATE_PCT,
    LEADER_MIN_CHANGE_RATE_PCT,
    LEADER_MIN_MARKET_CAP,
    LEADER_MIN_TRADING_VALUE,
    LEADER_STORE_COUNT,
    LEADER_WEIGHT_CHANGE_RATE,
    LEADER_WEIGHT_TRADING_VALUE,
)
from .toss_client import ThemeStock


@dataclass(frozen=True)
class LeaderCandidate:
    """점수가 매겨진 주도주 후보."""

    stock: ThemeStock
    score: float
    rank_in_theme: int


def select_leaders(
    stocks: list[ThemeStock],
    top_n: int = LEADER_STORE_COUNT,
    listed_codes: set[str] | None = None,
) -> list[LeaderCandidate]:
    """테마 구성 종목 중 주도주 상위 N개를 점수순으로 반환한다.

    Args:
        stocks:       테마 구성 종목
        top_n:        반환할 상위 후보 수
        listed_codes: 국내 상장 종목코드 집합. 주어지면 여기 없는 코드를 제외한다.

    Returns:
        점수 내림차순 LeaderCandidate 리스트 (rank_in_theme 은 1부터).
        조건을 만족하는 종목이 없으면 빈 리스트.
    """
    eligible = [s for s in stocks if _is_eligible(s, listed_codes)]
    if not eligible:
        return []

    value_scores = _min_max_normalize([float(s.trading_value) for s in eligible])
    change_scores = _min_max_normalize([s.change_rate for s in eligible])

    scored = [
        (
            stock,
            LEADER_WEIGHT_TRADING_VALUE * value_score
            + LEADER_WEIGHT_CHANGE_RATE * change_score,
        )
        for stock, value_score, change_score in zip(eligible, value_scores, change_scores)
    ]
    # 동점 시 거래대금이 큰 쪽을 우선 (실체 있는 수급 우선)
    scored.sort(key=lambda pair: (pair[1], pair[0].trading_value), reverse=True)

    return [
        LeaderCandidate(stock=stock, score=score, rank_in_theme=idx)
        for idx, (stock, score) in enumerate(scored[:top_n], start=1)
    ]


def _is_eligible(stock: ThemeStock, listed_codes: set[str] | None) -> bool:
    """후보 자격 필터."""
    if listed_codes is not None and stock.code not in listed_codes:
        return False
    if stock.change_rate < LEADER_MIN_CHANGE_RATE_PCT:
        return False
    if stock.change_rate > LEADER_MAX_CHANGE_RATE_PCT:
        return False
    if stock.trading_value < LEADER_MIN_TRADING_VALUE:
        return False
    if stock.market_cap < LEADER_MIN_MARKET_CAP:
        return False
    return stock.price > 0


def _min_max_normalize(values: list[float]) -> list[float]:
    """리스트를 0~1 로 정규화한다. 전부 같은 값이면 모두 1.0."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [1.0] * len(values)
    span = hi - lo
    return [(v - lo) / span for v in values]
