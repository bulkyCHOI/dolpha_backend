# standard library
from typing import List

# third-party
import requests
from ninja import NinjaAPI, Router, Schema

# Django
from django.http import JsonResponse

# local Django
from .api_auth import auth_router
from .api_data import data_router
from .api_mypage_ninja import mypage_router
from .api_query import query_router
from .api_trading_status import trading_status_router
from .api_account_settings import account_settings_router
from .api_search import search_router
from .api_trading_reviews import trading_reviews_router
from .api_data_status import data_status_router
from .api_trade import trade_router

api = NinjaAPI()


@api.get("/hello")
def hello(request):
    return {"message": "Hello, Django Ninja!"}


# 데이터 수집/저장/계산 관련 API 라우터 추가
api.add_router("/", data_router)

# 데이터 조회 관련 API 라우터 추가
api.add_router("/", query_router)


# Google OAuth 인증 관련 API 라우터 추가

api.add_router("/auth", auth_router)

# 마이페이지 관련 API 라우터 추가

api.add_router("/mypage", mypage_router)

# 거래 상태 조회 API 라우터 추가

api.add_router("/mypage", trading_status_router)

# 계좌 설정 API 라우터 추가

api.add_router("/mypage", account_settings_router)

# 종목 검색 API 라우터 추가

api.add_router("/search", search_router)

# 매매복기 API 라우터 추가

api.add_router("/", trading_reviews_router)

# 데이터 수집 현황 API 라우터 추가
api.add_router("/data-status", data_status_router)

# 수동 매매 API 라우터 추가
api.add_router("/trade", trade_router)


NAVER_STOCK_URL = "https://polling.finance.naver.com/api/realtime/domestic/stock/{code}"


def _fetch_naver_price(stock_code: str) -> dict:
    """네이버 파이낸스에서 단일 종목 현재가 조회. 성공 시 표준 가격 dict 반환, 실패 시 None."""
    try:
        res = requests.get(NAVER_STOCK_URL.format(code=stock_code), timeout=5)
        if res.status_code != 200:
            return None
        data = res.json().get("datas", [])
        if not data:
            return None
        d = data[0]
        return {
            "success": True,
            "price": int(d["closePriceRaw"]),
            "change": int(d["compareToPreviousClosePriceRaw"]),
            "changePercent": float(d["fluctuationsRatioRaw"]),
            "source": "naver",
            "market_state": d.get("marketStatus", "UNKNOWN"),
        }
    except Exception:
        return None


# 단일 종목 현재가 조회
@api.get("/stock-price/{stock_code}")
def get_stock_price(request, stock_code: str):
    result = _fetch_naver_price(stock_code)
    if result is None:
        return JsonResponse({"success": False, "error": f"현재가 조회 실패: {stock_code}"}, status=404)
    return result


class BatchPriceRequest(Schema):
    stock_codes: List[str]


# 다수 종목 현재가 일괄 조회
@api.post("/stock-prices/batch")
def get_stock_prices_batch(request, body: BatchPriceRequest):
    stock_codes = body.stock_codes

    if not stock_codes:
        return JsonResponse({"success": False, "error": "stock_codes가 비어있습니다."}, status=400)
    if len(stock_codes) > 50:
        return JsonResponse({"success": False, "error": "한 번에 최대 50개까지 조회 가능합니다."}, status=400)

    results = {}
    for code in stock_codes:
        result = _fetch_naver_price(code)
        results[code] = result if result else {"success": False, "error": "현재가 조회 실패"}

    return JsonResponse({"success": True, "prices": results})
