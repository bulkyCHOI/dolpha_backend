# standard library
import json

# third-party
import requests
from ninja import NinjaAPI, Router

# Django
from django.http import JsonResponse

# local Django
from .api_auth import auth_router
from .api_data import data_router
from .api_mypage_ninja import mypage_router
from .api_query import query_router
from .api_trading_status import trading_status_router

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


# 주가 조회 API 엔드포인트
@api.get("/stock-price/{stock_code}")
def get_stock_price(request, stock_code: str):
    """
    한국 주식의 현재가를 조회하는 API
    여러 데이터 소스를 시도하여 가장 정확한 데이터를 제공합니다.
    """

    def try_investing_com(stock_code):
        """Investing.com API 시도"""
        try:
            # Investing.com의 한국 주식 데이터
            url = f"https://api.investing.com/api/financialdata/{stock_code}/historical/chart/"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "application/json",
            }
            response = requests.get(url, headers=headers, timeout=8)
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    latest = data[-1]  # 최신 데이터
                    return {
                        "price": latest.get("price_close"),
                        "change": latest.get("price_close") - latest.get("price_open"),
                        "source": "investing.com",
                    }
        except:
            pass
        return None

    def try_alpha_vantage(stock_code):
        """Alpha Vantage API 시도 (무료 버전)"""
        try:
            # Alpha Vantage 무료 API (제한적이지만 정확)
            api_key = "demo"  # 실제 운영시에는 API 키 필요
            symbol = f"{stock_code}.KRX"  # 한국거래소 심볼
            url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"

            response = requests.get(url, timeout=8)
            if response.status_code == 200:
                data = response.json()
                quote = data.get("Global Quote", {})

                current_price = quote.get("05. price")
                change = quote.get("09. change")

                if current_price:
                    return {
                        "price": float(current_price),
                        "change": float(change) if change else 0,
                        "source": "alphavantage",
                    }
        except:
            pass
        return None

    def try_yahoo_finance(stock_code):
        """Yahoo Finance API 시도 (기존 방식 개선)"""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }

            # KOSPI 시도
            for suffix in [".KS", ".KQ"]:
                symbol = f"{stock_code}{suffix}"

                # 최신 Yahoo Finance API 사용
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=1d"

                response = requests.get(url, headers=headers, timeout=8)

                if response.status_code == 200:
                    data = response.json()

                    if data.get("chart") and data["chart"].get("result"):
                        result = data["chart"]["result"][0]
                        meta = result.get("meta", {})

                        # 여러 가격 필드 시도
                        current_price = (
                            meta.get("regularMarketPrice")
                            or meta.get("previousClose")
                            or meta.get("chartPreviousClose")
                        )

                        previous_close = meta.get("previousClose", 0)

                        if current_price:
                            change = (
                                current_price - previous_close if previous_close else 0
                            )
                            return {
                                "price": current_price,
                                "change": change,
                                "source": f"yahoo_{suffix}",
                                "market_state": meta.get("marketState"),
                            }

        except Exception as e:
            pass
        return None

    def try_naver_finance(stock_code):
        """네이버 금융 크롤링 시도 (개선된 패턴 매칭)"""
        try:
            url = f"https://finance.naver.com/item/main.nhn?code={stock_code}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "ko-KR,ko;q=0.8,en-US;q=0.5,en;q=0.3",
            }

            response = requests.get(url, headers=headers, timeout=8)

            if response.status_code == 200:
                import re

                content = response.text

                # 다양한 현재가 패턴 시도 (우선순위 순)
                price_patterns = [
                    r'<span class="blind">현재가</span>\s*([0-9,]+)',  # 기존 패턴
                    r'<strong id="_nowVal"[^>]*>([0-9,]+)</strong>',  # nowVal ID
                    r'class="no_today">([0-9,]+)</span>',  # no_today 클래스
                    r'<em class="blind">현재가</em>[^>]*>([0-9,]+)',  # em 태그
                    r"현재가.*?([0-9,]+)",  # 일반적인 패턴
                    r"<dd[^>]*>([0-9,]+)</dd>",  # dd 태그
                    r'price_now">([0-9,]+)</span>',  # price_now 클래스
                ]

                current_price = None
                for pattern in price_patterns:
                    price_match = re.search(pattern, content)
                    if price_match:
                        price_str = price_match.group(1).replace(",", "")
                        try:
                            current_price = float(price_str)
                            break
                        except ValueError:
                            continue

                if current_price:
                    # 전일대비 패턴 찾기
                    change_patterns = [
                        r'<span class="blind">전일대비</span>.*?([+-]?[0-9,]+)',
                        r"전일대비.*?([+-]?[0-9,]+)",
                        r"change_rate[^>]*>([+-]?[0-9,]+)",
                    ]

                    change = 0
                    for pattern in change_patterns:
                        change_match = re.search(pattern, content)
                        if change_match:
                            try:
                                change_str = change_match.group(1).replace(",", "")
                                change = float(change_str)
                                break
                            except ValueError:
                                continue

                    return {
                        "price": current_price,
                        "change": change,
                        "source": "naver_finance",
                    }

        except Exception as e:
            pass
        return None

    # 여러 데이터 소스 시도 (우선순위 순)
    data_sources = [
        try_naver_finance,  # 1순위: 네이버 금융 (가장 정확)
        try_yahoo_finance,  # 2순위: Yahoo Finance
        try_alpha_vantage,  # 3순위: Alpha Vantage
        try_investing_com,  # 4순위: Investing.com
    ]

    for source_func in data_sources:
        try:
            result = source_func(stock_code)
            if result and result.get("price"):
                price = result["price"]
                change = result.get("change", 0)
                change_percent = (
                    (change / (price - change)) * 100 if (price - change) > 0 else 0
                )

                return {
                    "success": True,
                    "price": price,
                    "change": round(change, 2),
                    "changePercent": round(change_percent, 2),
                    "source": result.get("source", "unknown"),
                    "market_state": result.get("market_state", "unknown"),
                }
        except Exception as e:
            continue

    # 모든 데이터 소스에서 실패한 경우
    return JsonResponse(
        {
            "success": False,
            "error": f"주가 정보를 찾을 수 없습니다: {stock_code}. 모든 데이터 소스에서 조회 실패.",
        },
        status=404,
    )
