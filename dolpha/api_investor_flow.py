"""
매매동향 API — 외국인·프로그램·회원사 매매동향 엔드포인트

엔드포인트:
  GET /stock/{code}/investor-today    — 당일 투자자별 순매수 (개인/외국인/기관)
  GET /stock/{code}/foreign-total     — 당일 외국인/기관 가집계 시간대별
  GET /stock/{code}/program-trade     — 당일 프로그램매매 추이
  GET /stock/{code}/member-firm       — 당일 회원사(전 증권사)별 매매동향

참고: KIS API의 해당 엔드포인트들은 장중(평일 09:00~15:30 KST)에만 데이터를 제공합니다.
"""

import time
import logging
from datetime import datetime, timezone, timedelta

from ninja import Router
from django.http import JsonResponse

from .kis.investor_flow import (
    GetInvestorToday,
    GetForeignInstitutionTotal,
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
    """당일 투자자별 순매수 (개인/외국인/기관) — FHKST01010900"""
    return _handle(request, stock_code, GetInvestorToday, "investor-today")


@investor_flow_router.get("/stock/{stock_code}/foreign-total")
def get_foreign_total(request, stock_code: str):
    """당일 외국인/기관 매매 가집계 시간대별 — HHKST03900300"""
    return _handle(request, stock_code, GetForeignInstitutionTotal, "foreign-total")


@investor_flow_router.get("/stock/{stock_code}/program-trade")
def get_program_trade(request, stock_code: str):
    """당일 프로그램매매 추이(체결) — FHPPG04650100"""
    return _handle(request, stock_code, GetProgramTradeToday, "program-trade")


@investor_flow_router.get("/stock/{stock_code}/member-firm")
def get_member_firm(request, stock_code: str):
    """전 증권사(회원사)별 당일 매매동향 — FHKST01010600"""
    return _handle(request, stock_code, GetMemberFirmTrading, "member-firm")
