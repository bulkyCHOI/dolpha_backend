"""
매매동향 API — 외국인·프로그램·회원사 매매동향 엔드포인트

엔드포인트:
  GET /stock/{code}/investor-today    — 일자별 투자자 순매수 (개인/외국인/기관)
  GET /stock/{code}/program-trade     — 당일 프로그램매매 추이
  GET /stock/{code}/member-firm       — 회원사(증권사) 매도/매수 상위 및 외국계 합계
  GET /stock/{code}/investor-flow-snapshot — 장 마감 직전 저장된 스냅샷 조회 (장 마감 후용)
  GET /investor-flow-snapshot/list         — 자동매매 대상 전 종목 스냅샷 목록 (장 마감 후용)

참고: program-trade 는 장중(평일 09:00~15:30 KST)에만 시간대별 데이터를 제공합니다.
      자동매매 대상 종목은 15:29에 스냅샷으로 미리 저장되어 장 마감 후에도 조회 가능합니다.
"""

import time
import logging
from datetime import datetime, timezone, timedelta

from ninja import Router
from django.http import JsonResponse

from .kis.investor_flow import (
    GetInvestorToday,
    GetProgramTradeToday,
    GetMemberFirmTrading,
)

logger = logging.getLogger(__name__)

investor_flow_router = Router()

# 5분 TTL 인메모리 캐시 (프로세스 재시작 시 초기화)
_cache: dict[str, tuple[float, object]] = {}
_TTL = 300  # seconds

KST = timezone(timedelta(hours=9))
_MARKET_OPEN = (9, 0)    # 09:00
_MARKET_CLOSE = (15, 30) # 15:30


def _is_market_open() -> bool:
    """한국 증시 장 운영 시간 여부 (평일 09:00~15:30 KST). 공휴일은 고려하지 않음."""
    now = datetime.now(KST)
    if now.weekday() >= 5:  # 토요일=5, 일요일=6
        return False
    t = (now.hour, now.minute)
    return _MARKET_OPEN <= t <= _MARKET_CLOSE


def _cached(key: str, fetcher):
    now = time.time()
    if key in _cache:
        ts, val = _cache[key]
        if now - ts < _TTL:
            return val
    val = fetcher()
    _cache[key] = (now, val)
    return val


def _handle(request, stock_code: str, fetcher_fn, cache_key_prefix: str):
    if not stock_code or len(stock_code) != 6 or not stock_code.isdigit():
        return JsonResponse({"success": False, "error": "유효한 6자리 종목코드가 필요합니다."}, status=400)

    market_open = _is_market_open()
    key = f"{cache_key_prefix}:{stock_code}"
    try:
        data = _cached(key, lambda: fetcher_fn(stock_code))
        return JsonResponse({"success": True, "data": data, "is_market_closed": not market_open})
    except RuntimeError as e:
        logger.warning("investor_flow API 오류 [%s] %s: %s", cache_key_prefix, stock_code, e)
        return JsonResponse({"success": False, "error": str(e)}, status=502)
    except Exception as e:
        logger.exception("investor_flow API 예외 [%s] %s", cache_key_prefix, stock_code)
        return JsonResponse({"success": False, "error": "서버 오류가 발생했습니다."}, status=500)


@investor_flow_router.get("/stock/{stock_code}/investor-today")
def get_investor_today(request, stock_code: str):
    """일자별 투자자 순매수 (개인/외국인/기관) — FHKST01010900"""
    return _handle(request, stock_code, GetInvestorToday, "investor-today")


@investor_flow_router.get("/stock/{stock_code}/program-trade")
def get_program_trade(request, stock_code: str):
    """당일 프로그램매매 추이(체결) — FHPPG04650100"""
    return _handle(request, stock_code, GetProgramTradeToday, "program-trade")


@investor_flow_router.get("/stock/{stock_code}/member-firm")
def get_member_firm(request, stock_code: str):
    """회원사(증권사) 매도/매수 상위 및 외국계 합계 — FHKST01010600"""
    return _handle(request, stock_code, GetMemberFirmTrading, "member-firm")


def _snapshot_to_dict(snapshot) -> dict:
    return {
        "stock_code": snapshot.stock_code,
        "stock_name": snapshot.stock_name,
        "date": snapshot.date.isoformat(),
        "investor_today": snapshot.investor_today,
        "program_trade": snapshot.program_trade,
        "member_firm": snapshot.member_firm,
        "captured_at": snapshot.captured_at.isoformat(),
    }


@investor_flow_router.get("/stock/{stock_code}/investor-flow-snapshot")
def get_investor_flow_snapshot(request, stock_code: str, date: str = ""):
    """장 마감 직전(15:29)에 저장해 둔 매매동향 스냅샷 조회 (장 마감 후 조회용).

    date 미지정 시 가장 최근 저장된 스냅샷을 반환한다.
    """
    from myweb.models import InvestorFlowSnapshot

    if not stock_code or len(stock_code) != 6 or not stock_code.isdigit():
        return JsonResponse({"success": False, "error": "유효한 6자리 종목코드가 필요합니다."}, status=400)

    qs = InvestorFlowSnapshot.objects.filter(stock_code=stock_code)
    if date:
        qs = qs.filter(date=date)

    snapshot = qs.order_by("-date").first()
    if snapshot is None:
        return JsonResponse({"success": False, "error": "저장된 매매동향 스냅샷이 없습니다."}, status=404)

    return JsonResponse({"success": True, "data": _snapshot_to_dict(snapshot)})


@investor_flow_router.get("/investor-flow-snapshot/list")
def list_investor_flow_snapshots(request, date: str = ""):
    """자동매매 대상 전 종목의 매매동향 스냅샷 목록 조회 (date 미지정 시 최신 날짜)."""
    from myweb.models import InvestorFlowSnapshot

    qs = InvestorFlowSnapshot.objects.all()
    if date:
        qs = qs.filter(date=date)
    else:
        latest = InvestorFlowSnapshot.objects.order_by("-date").values_list("date", flat=True).first()
        if latest is None:
            return JsonResponse({"success": True, "data": []})
        qs = qs.filter(date=latest)

    data = [_snapshot_to_dict(s) for s in qs.order_by("stock_code")]
    return JsonResponse({"success": True, "data": data})
