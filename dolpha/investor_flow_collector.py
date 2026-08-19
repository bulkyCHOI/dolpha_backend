"""
자동매매 설정 목록 종목의 매매동향 장마감 전 스냅샷 수집.

KIS 매매동향 API(investor_flow.py)는 장중(09:00~15:30 KST)에만 데이터를 제공하므로,
장 마감 직전에 TradingConfig(is_active=True)에 등록된 종목들의 매매동향을 미리 수집해
InvestorFlowSnapshot에 저장한다. 장 마감 후에는 이 스냅샷을 조회해서 볼 수 있다.
"""

import logging
import time

from django.utils import timezone

from .kis.investor_flow import (
    GetInvestorToday,
    GetForeignInstitutionTotal,
    GetProgramTradeToday,
    GetMemberFirmTrading,
)

logger = logging.getLogger(__name__)

_REQUEST_INTERVAL_SEC = 0.3  # KIS API 호출 간 최소 간격 (레이트리밋 회피)


def collect_and_save_investor_flow_snapshots() -> dict:
    """자동매매 활성 종목의 매매동향을 조회해 오늘 날짜로 스냅샷을 저장한다.

    Returns:
        {"total": int, "saved": int, "failed": list[str]}
    """
    from myweb.models import TradingConfig, InvestorFlowSnapshot

    today = timezone.localdate()

    targets = list(
        TradingConfig.objects.filter(is_active=True)
        .values("stock_code", "stock_name")
        .distinct()
    )

    saved = 0
    failed = []

    for target in targets:
        stock_code = target["stock_code"]
        stock_name = target["stock_name"]
        try:
            investor_today = GetInvestorToday(stock_code)
            time.sleep(_REQUEST_INTERVAL_SEC)
            foreign_total = GetForeignInstitutionTotal(stock_code)
            time.sleep(_REQUEST_INTERVAL_SEC)
            program_trade = GetProgramTradeToday(stock_code)
            time.sleep(_REQUEST_INTERVAL_SEC)
            member_firm = GetMemberFirmTrading(stock_code)
            time.sleep(_REQUEST_INTERVAL_SEC)

            InvestorFlowSnapshot.objects.update_or_create(
                stock_code=stock_code,
                date=today,
                defaults={
                    "stock_name": stock_name,
                    "investor_today": investor_today,
                    "foreign_total": foreign_total,
                    "program_trade": program_trade,
                    "member_firm": member_firm,
                },
            )
            saved += 1
        except Exception as e:
            logger.warning("매매동향 스냅샷 수집 실패 [%s]: %s", stock_code, e)
            failed.append(stock_code)

    return {"total": len(targets), "saved": saved, "failed": failed}
