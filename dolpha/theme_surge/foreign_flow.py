"""외국인 매수세 판정.

KIS '주식현재가 회원사'(inquire-member) 응답의 외국계 합계(glob_*)를 사용한다.
이 값은 장중에도 실시간으로 갱신되며, 당일 외국계 순매수 수량을 바로 준다.
추세는 직전 사이클에 기록해 둔 값(prev_net_buy)과 비교해 판단한다.

조회 자체가 불가능한 경우 available=False 로 반환하며, 상위 진입 로직은
외국인 필터를 미충족으로 처리한다(임의 통과시키지 않는다).
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import FOREIGN_MIN_NET_BUY_QTY


@dataclass(frozen=True)
class ForeignFlow:
    """외국인 수급 판정 결과."""

    available: bool          # 조회 성공 여부
    net_buy_qty: int         # 당일 외국계 순매수(주)
    is_increasing: bool      # 직전 사이클 대비 순매수가 늘고 있는가
    source: str              # "member-firm" | "none"
    detail: str              # 사람이 읽을 수 있는 요약/오류 사유

    @property
    def is_buying(self) -> bool:
        """매수세로 인정할 수 있는 상태인지."""
        return self.available and self.net_buy_qty > FOREIGN_MIN_NET_BUY_QTY


def get_foreign_flow(stock_code: str, prev_net_buy: int | None = None) -> ForeignFlow:
    """종목의 외국인 매수세를 조회한다.

    Args:
        stock_code:   6자리 종목코드
        prev_net_buy: 직전 사이클에 기록해 둔 외국계 순매수(주). 증가 추세 판단에 사용한다.
    """
    try:
        from dolpha.kis.investor_flow import GetMemberFirmTrading

        foreign = (GetMemberFirmTrading(stock_code) or {}).get("foreign") or {}
    except Exception as e:
        return ForeignFlow(False, 0, False, "none", f"회원사 조회 실패: {e}")

    if "ntby_qty" not in foreign:
        return ForeignFlow(False, 0, False, "none", "회원사 응답에 외국계 합계 없음")

    net_buy = int(foreign["ntby_qty"])
    # 직전 기록이 없으면 추세를 알 수 없으므로 증가로 단정하지 않는다
    is_increasing = prev_net_buy is not None and net_buy > prev_net_buy

    trend = f"직전 {net_buy - prev_net_buy:+,}주" if prev_net_buy is not None else "직전 기록 없음"
    return ForeignFlow(
        available=True,
        net_buy_qty=net_buy,
        is_increasing=is_increasing,
        source="member-firm",
        detail=f"외국계 순매수 {net_buy:+,}주 ({trend})",
    )
