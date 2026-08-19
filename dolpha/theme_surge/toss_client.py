"""토스증권 '지금 뜨는 산업' 크롤러.

토스증권 웹(tossinvest.com)이 사용하는 공개 조회 API를 그대로 호출한다.
인증 토큰이 필요 없는 시세 조회 엔드포인트이며, 응답은 아래 두 종류다.

  1) 테마 랭킹  POST /api/v2/dashboard/wts/overview/tics/ranking
       body: {"nation": "KR", "duration": "1d", "sortBy": "FLUCTUATION_RATE"}
       → 테마별 등락률·거래대금·시가총액·구성종목수·대표종목

  2) 테마 구성종목  POST /api/v2/dashboard/wts/overview/tics/{ticsId}/stocks
       body: {"nation": "KR", "page": 1}
       → 종목별 등락률·거래대금·시가총액·거래량 (페이지당 10개)

응답 스키마가 바뀌어도 전략 전체가 멈추지 않도록, 이 모듈에서 내부 dataclass 로
정규화한 뒤 상위 계층에 전달한다.
"""

from __future__ import annotations

import time as time_module
from dataclasses import dataclass

import requests

from .config import (
    TOSS_API_BASE,
    TOSS_DURATION,
    TOSS_FALLBACK_RANKING_PATH,
    TOSS_MAX_PAGES,
    TOSS_MAX_RETRY,
    TOSS_NATION,
    TOSS_PAGE_SIZE,
    TOSS_RANKING_PATH,
    TOSS_REQUEST_INTERVAL_SEC,
    TOSS_RETRY_BACKOFF_SEC,
    TOSS_SORT_BY,
    TOSS_STOCKS_PATH,
    TOSS_TIMEOUT_SEC,
)

_HEADERS = {
    "accept": "application/json",
    "content-type": "application/json",
    "user-agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    ),
    "referer": "https://www.tossinvest.com/",
}


class TossThemeError(RuntimeError):
    """토스증권 API 호출/파싱 실패."""


@dataclass(frozen=True)
class ThemeRank:
    """테마 랭킹 1행."""

    tics_id: int
    name: str
    rank: int
    fluctuation_rate: float   # % 단위 (예: 7.44)
    trading_value: int        # 원
    market_cap: int           # 원
    stock_count: int
    leading_stock_code: str   # 6자리 (토스 productCode 에서 접두 'A' 제거)
    leading_stock_name: str


@dataclass(frozen=True)
class ThemeStock:
    """테마 구성 종목 1행."""

    code: str                 # 6자리 종목코드
    name: str
    price: float
    change_rate: float        # % 단위
    trading_value: int        # 원
    volume: int
    market_cap: int


# ──────────────────────────────────────────────────────────────
# 내부 헬퍼
# ──────────────────────────────────────────────────────────────

def _normalize_code(product_code: str | None) -> str:
    """토스 productCode('A005380') → KRX 종목코드('005380').

    국내 종목이 아니거나 형식이 다르면 빈 문자열을 반환한다.
    """
    if not product_code:
        return ""
    code = product_code.strip().upper()
    if code.startswith("A"):
        code = code[1:]
    return code if len(code) == 6 else ""


def _to_int(value) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


def _to_float(value) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _post(path: str, payload: dict) -> dict:
    """토스 API POST 호출 (재시도 포함). 실패 시 TossThemeError."""
    url = f"{TOSS_API_BASE}{path}"
    last_error: Exception | None = None

    for attempt in range(TOSS_MAX_RETRY):
        try:
            res = requests.post(
                url, json=payload, headers=_HEADERS, timeout=TOSS_TIMEOUT_SEC
            )
            if res.status_code != 200:
                raise TossThemeError(
                    f"HTTP {res.status_code}: {res.text[:200]}"
                )
            body = res.json()
            if "result" not in body:
                raise TossThemeError(f"예상치 못한 응답 형식: {str(body)[:200]}")
            return body["result"]
        except (requests.RequestException, ValueError, TossThemeError) as e:
            last_error = e
            if attempt < TOSS_MAX_RETRY - 1:
                time_module.sleep(TOSS_RETRY_BACKOFF_SEC * (attempt + 1))

    raise TossThemeError(f"{path} 호출 실패 ({TOSS_MAX_RETRY}회 재시도): {last_error}")


def _get(path: str) -> dict:
    """토스 API GET 호출 (fallback 용)."""
    url = f"{TOSS_API_BASE}{path}"
    res = requests.get(url, headers=_HEADERS, timeout=TOSS_TIMEOUT_SEC)
    if res.status_code != 200:
        raise TossThemeError(f"HTTP {res.status_code}: {res.text[:200]}")
    body = res.json()
    if "result" not in body:
        raise TossThemeError(f"예상치 못한 응답 형식: {str(body)[:200]}")
    return body["result"]


# ──────────────────────────────────────────────────────────────
# 공개 API
# ──────────────────────────────────────────────────────────────

def fetch_theme_ranking(limit: int = 30) -> list[ThemeRank]:
    """'지금 뜨는 산업' 테마 랭킹을 등락률 내림차순으로 조회한다.

    Args:
        limit: 반환할 최대 테마 수

    Returns:
        ThemeRank 리스트 (등락률 내림차순)

    Raises:
        TossThemeError: 기본 엔드포인트와 fallback 모두 실패한 경우
    """
    payload = {
        "nation": TOSS_NATION,
        "duration": TOSS_DURATION,
        "sortBy": TOSS_SORT_BY,
    }

    try:
        result = _post(TOSS_RANKING_PATH, payload)
        return _parse_ranking_v2(result, limit)
    except TossThemeError as primary_error:
        print(f"[급등테마] 기본 랭킹 API 실패 → fallback 시도: {primary_error}")
        try:
            return _parse_ranking_fallback(_get(TOSS_FALLBACK_RANKING_PATH), limit)
        except TossThemeError as fallback_error:
            raise TossThemeError(
                f"테마 랭킹 조회 실패 (기본: {primary_error} / fallback: {fallback_error})"
            ) from fallback_error


def _parse_ranking_v2(result: dict, limit: int) -> list[ThemeRank]:
    """v2 ranking 응답 파싱."""
    themes: list[ThemeRank] = []

    for row in result.get("tics", [])[:limit]:
        tics_id = _to_int(row.get("ticsId"))
        name = (row.get("name") or "").strip()
        if not tics_id or not name:
            continue

        leading = row.get("leadingStock") or {}
        themes.append(
            ThemeRank(
                tics_id=tics_id,
                name=name,
                rank=_to_int(row.get("rank")),
                # 토스는 비율(0.0744)로 내려주므로 % 로 환산
                fluctuation_rate=_to_float(row.get("fluctuationRate")) * 100.0,
                trading_value=_to_int(row.get("tradingAmountKrw")),
                market_cap=_to_int(row.get("totalMarketCapKrw")),
                stock_count=_to_int(row.get("stockCount")),
                leading_stock_code=_normalize_code(leading.get("productCode")),
                leading_stock_name=(leading.get("name") or "").strip(),
            )
        )

    return themes


def _parse_ranking_fallback(result: dict, limit: int) -> list[ThemeRank]:
    """GET /api/v1/tics/rankings 응답 파싱.

    거래대금·시가총액을 제공하지 않으므로 0 으로 채운다.
    급등 판정에서 거래대금 조건은 자동으로 통과 처리된다(상위 계층에서 0 은 미지 값으로 취급).
    """
    themes: list[ThemeRank] = []

    for row in result.get("data", [])[:limit]:
        tics_id = _to_int(row.get("ticsId"))
        name = (row.get("title") or "").strip()
        if not tics_id or not name:
            continue

        # "+7.31%" 형태의 문자열
        raw = (row.get("preciseValue") or row.get("value") or "0").replace("%", "").strip()
        themes.append(
            ThemeRank(
                tics_id=tics_id,
                name=name,
                rank=_to_int(row.get("ranking")),
                fluctuation_rate=_to_float(raw),
                trading_value=0,
                market_cap=0,
                stock_count=_to_int(row.get("totalCount")),
                leading_stock_code="",
                leading_stock_name="",
            )
        )

    return themes


def fetch_theme_stocks(tics_id: int, max_pages: int = TOSS_MAX_PAGES) -> list[ThemeStock]:
    """특정 테마의 구성 종목을 전 페이지 조회한다.

    Args:
        tics_id:   토스 산업분류 ID
        max_pages: 조회할 최대 페이지 수 (페이지당 10종목)

    Returns:
        ThemeStock 리스트. 국내 6자리 코드로 정규화되지 않는 종목은 제외한다.
    """
    path = TOSS_STOCKS_PATH.format(tics_id=tics_id)
    stocks: list[ThemeStock] = []
    seen: set[str] = set()

    for page in range(1, max_pages + 1):
        result = _post(path, {"nation": TOSS_NATION, "page": page})
        rows = result.get("stocks", [])
        if not rows:
            break

        for row in rows:
            code = _normalize_code(row.get("code"))
            if not code or code in seen:
                continue
            seen.add(code)

            price_info = row.get("price") or {}
            stocks.append(
                ThemeStock(
                    code=code,
                    name=(row.get("name") or "").strip(),
                    price=_to_float(price_info.get("close")),
                    change_rate=_to_float(row.get("changeRate")) * 100.0,
                    trading_value=_to_int(row.get("tradingValueKrw")),
                    volume=_to_int(row.get("volume")),
                    market_cap=_to_int(row.get("marketCapKrw")),
                )
            )

        total_count = _to_int(result.get("totalCount"))
        if total_count and len(seen) >= total_count:
            break
        if len(rows) < TOSS_PAGE_SIZE:
            break

        time_module.sleep(TOSS_REQUEST_INTERVAL_SEC)

    return stocks
