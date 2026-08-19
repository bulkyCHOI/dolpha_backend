"""외국인 매수세 판정.

KIS '외국인/기관 매매종목 가집계'(시간대별)를 1순위로 사용해 당일 외국인 순매수와
그 증가 추세를 본다. 이 엔드포인트는 KIS REAL 키가 있어야 하므로, 실패하면
'주식현재가 투자자'(당일 투자자별 순매수)로 폴백해 시점 값만 얻고 추세는
직전 사이클에 기록해 둔 값과 비교해 판단한다.

조회 자체가 불가능한 경우 available=False 로 반환하며, 상위 진입 로직은
외국인 필터를 미충족으로 처리한다(임의 통과시키지 않는다).
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import (
    FOREIGN_MIN_NET_BUY_QTY,
    FOREIGN_RECENT_SLOTS,
)

# KIS inquire-investor output2 의 투자자 행 순서: 0=개인, 1=외국인, 2=기관계
_FOREIGN_ROW_INDEX = 1


@dataclass(frozen=True)
class ForeignFlow:
    """외국인 수급 판정 결과."""

    available: bool          # 조회 성공 여부
    net_buy_qty: int         # 당일 외국인 순매수(주)
    is_increasing: bool      # 최근 구간에서 순매수가 늘고 있는가
    source: str              # "foreign-total" | "investor-today" | "none"
    detail: str              # 사람이 읽을 수 있는 요약/오류 사유

    @property
    def is_buying(self) -> bool:
        """매수세로 인정할 수 있는 상태인지."""
        return self.available and self.net_buy_qty > FOREIGN_MIN_NET_BUY_QTY


def get_foreign_flow(stock_code: str, prev_net_buy: int | None = None) -> ForeignFlow:
    """종목의 외국인 매수세를 조회한다.

    Args:
        stock_code:   6자리 종목코드
        prev_net_buy: 직전 사이클에 기록해 둔 외국인 순매수(주).
                      시간대별 조회가 실패했을 때 증가 추세 판단에 사용한다.
    """
    flow = _from_foreign_total(stock_code)
    if flow.available:
        return flow

    return _from_investor_today(stock_code, prev_net_buy, fallback_reason=flow.detail)


def _from_foreign_total(stock_code: str) -> ForeignFlow:
    """시간대별 가집계로 순매수와 추세를 계산한다."""
    try:
        from dolpha.kis.investor_flow import GetForeignInstitutionTotal

        rows = (GetForeignInstitutionTotal(stock_code) or {}).get("output2", [])
    except Exception as e:
        return ForeignFlow(False, 0, False, "none", f"가집계 조회 실패: {e}")

    series = _extract_series(rows)
    if not series:
        return ForeignFlow(False, 0, False, "none", "가집계 응답에 유효한 시간대 데이터 없음")

    latest = series[-1]
    # 비교 기준: 최근 N 구간 이전 값 (구간이 부족하면 가장 오래된 값)
    baseline_idx = max(0, len(series) - 1 - FOREIGN_RECENT_SLOTS)
    baseline = series[baseline_idx]
    is_increasing = latest > baseline

    return ForeignFlow(
        available=True,
        net_buy_qty=latest,
        is_increasing=is_increasing,
        source="foreign-total",
        detail=f"외국인 순매수 {latest:+,}주 (최근 {len(series) - baseline_idx}구간 {latest - baseline:+,}주)",
    )


def _from_investor_today(
    stock_code: str, prev_net_buy: int | None, fallback_reason: str
) -> ForeignFlow:
    """투자자별 순매수(시점 값)로 폴백한다."""
    try:
        from dolpha.kis.investor_flow import GetInvestorToday

        rows = (GetInvestorToday(stock_code) or {}).get("output2", [])
    except Exception as e:
        return ForeignFlow(
            False, 0, False, "none", f"{fallback_reason} / 투자자별 조회도 실패: {e}"
        )

    if len(rows) <= _FOREIGN_ROW_INDEX:
        return ForeignFlow(False, 0, False, "none", f"{fallback_reason} / 외국인 행 없음")

    net_buy = _to_int(rows[_FOREIGN_ROW_INDEX].get("ntby_rsqn"))
    # 직전 기록이 없으면 추세를 알 수 없으므로 증가로 단정하지 않는다
    is_increasing = prev_net_buy is not None and net_buy > prev_net_buy

    return ForeignFlow(
        available=True,
        net_buy_qty=net_buy,
        is_increasing=is_increasing,
        source="investor-today",
        detail=f"외국인 순매수 {net_buy:+,}주 (직전 대비 판정)",
    )


def _extract_series(rows: list[dict]) -> list[int]:
    """가집계 응답을 시간 오름차순 외국인 순매수 시계열로 변환한다.

    KIS 는 최신 행을 먼저 주는 경우가 많아, 시간 필드가 있으면 그 기준으로 정렬한다.
    """
    parsed: list[tuple[str, int]] = []

    for row in rows:
        hour = str(row.get("stck_cntg_hour") or row.get("hts_hour") or "")
        raw = row.get("frgn_ntby_qty")
        if raw is None:
            continue
        parsed.append((hour, _to_int(raw)))

    if not parsed:
        return []

    if all(hour for hour, _ in parsed):
        parsed.sort(key=lambda pair: pair[0])
    else:
        # 시간 필드를 신뢰할 수 없으면 KIS 기본(최신 우선) 가정하고 뒤집는다
        parsed.reverse()

    return [qty for _, qty in parsed]


def _to_int(value) -> int:
    try:
        return int(float(str(value).replace(",", "") or 0))
    except (TypeError, ValueError):
        return 0
