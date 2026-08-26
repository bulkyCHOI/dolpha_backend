"""
dolpha.kis — KIS(한국투자증권) API 통합 패키지

인증 정보는 두 경로로 관리됩니다.
  1) 사용자별 계좌 (myweb.models.KisAccount, 마이페이지에서 등록)
     — 자동매매·잔고조회 등 "누구의 돈인가"가 중요한 호출에 사용.
     dolpha.kis.credentials.credential_for_account()로 KisCredential을 만들어 넘긴다.
  2) 서버 환경변수 계좌 (하위 호환 / 시세 수집 전용)
     KIS_REAL_APP_KEY / KIS_REAL_APP_SECRET / KIS_REAL_ACCOUNT_NO
     KIS_VIRTUAL_APP_KEY / KIS_VIRTUAL_APP_SECRET / KIS_VIRTUAL_ACCOUNT_NO
     KIS_ACCOUNT_CD   계좌 상품코드 (기본 01)
     KIS_TOKEN_PATH   토큰 저장 파일 경로 (기본 ./kis_token_{계좌}.json)
     KIS_MODE         REAL | VIRTUAL (기본 REAL) — account 인자 생략 시 기본값
"""
from .auth import GetToken, MakeToken, GetHeaders
from .holiday import is_trading_day, refresh_holiday_cache, next_trading_day
from .ohlcv import GetOhlcvKR
from .minute import GetMinuteOhlcvKR, _iter_minute_pages
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
    "is_trading_day", "refresh_holiday_cache", "next_trading_day",
    "GetOhlcvKR",
    "GetMinuteOhlcvKR", "_iter_minute_pages",
    "GetHashKey",
    "GetBalance", "GetMyStockList", "GetCurrentPrice",
    "MakeBuyMarketOrder", "MakeSellMarketOrder",
]
