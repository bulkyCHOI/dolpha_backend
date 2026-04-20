"""
dolpha.kis — KIS(한국투자증권) API 통합 패키지

환경변수로 인증 정보를 관리합니다 (YAML 파일 의존성 제거).

필수 환경변수:
  KIS_APP_KEY      앱 키
  KIS_APP_SECRET   앱 시크릿
  KIS_ACCOUNT_NO   계좌번호 (8자리)
  KIS_ACCOUNT_CD   계좌 상품코드 (기본 01)
  KIS_URL          API 기본 URL (기본 https://openapi.koreainvestment.com:9443)
  KIS_TOKEN_PATH   토큰 저장 파일 경로 (기본 ./kis_token.json)
  KIS_MODE         REAL | VIRTUAL (기본 REAL)
"""
from .auth import GetToken, MakeToken, GetHeaders
from .ohlcv import GetOhlcvKR
from .trade import (
    GetHashKey,
    GetBalance,
    GetMyStockList,
    GetCurrentPrice,
    MakeBuyMarketOrder,
    MakeSellMarketOrder,
)

__all__ = [
    "GetToken", "MakeToken", "GetHeaders",
    "GetOhlcvKR",
    "GetHashKey",
    "GetBalance", "GetMyStockList", "GetCurrentPrice",
    "MakeBuyMarketOrder", "MakeSellMarketOrder",
]
