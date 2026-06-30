"""
KIS API — 투자자별·외국인·프로그램·회원사 매매동향 조회

사용 엔드포인트:
  - inquire-investor        (FHKST01010900) : 주식현재가 투자자 (당일 투자자별 순매수)
  - foreign-institution-total (HHKST03900300) : 외국인/기관 매매종목 가집계
  - comp-program-trade-today  (FHPPG04650100) : 종목별 프로그램매매 추이(체결)
  - inquire-member           (FHKST01010600) : 주식현재가 회원사
"""

import warnings
import requests

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

from .auth import GetHeaders, get_url_base


def _base() -> str:
    return get_url_base("REAL")


# ─────────────────────────────────────────────────────────────
# 1. 당일 투자자별 순매수 현황 (개인/외국인/기관)
# ─────────────────────────────────────────────────────────────

def GetInvestorToday(stock_code: str) -> dict:
    """
    주식현재가 투자자 — 당일 투자자별(개인/외국인/기관) 순매수 수량·금액.

    Returns:
        {
          "output1": { stck_prpr, prdy_vrss, ... },   # 현재가 정보
          "output2": [                                 # 투자자별 매매 (복수)
            { seln_rsqn, shnu_rsqn, ntby_rsqn, ... }
          ]
        }
    """
    url = f"{_base()}/uapi/domestic-stock/v1/quotations/inquire-investor"
    headers = GetHeaders(tr_id="FHKST01010900", mode="REAL")
    params = {
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_INPUT_ISCD": stock_code,
    }

    res = requests.get(url, headers=headers, params=params, timeout=15, verify=False)
    data = res.json()

    if res.status_code != 200 or data.get("rt_cd") != "0":
        raise RuntimeError(
            f"[KIS investor] 호출 실패 ({res.status_code}): {data.get('msg1', res.text[:200])}"
        )

    return {
        "output1": data.get("output1", {}),
        "output2": data.get("output2", []),
    }


# ─────────────────────────────────────────────────────────────
# 2. 외국인/기관 매매종목 가집계 (당일 시간대별 누적)
# ─────────────────────────────────────────────────────────────

def GetForeignInstitutionTotal(stock_code: str) -> dict:
    """
    국내기관_외국인 매매종목 가집계 — 당일 실시간 외국인+기관 순매수.

    Returns:
        {
          "output1": { ... },    # 종목 기본정보
          "output2": [ ... ],    # 시간대별 외국인/기관 누적 매매 내역
        }
    """
    url = f"{_base()}/uapi/domestic-stock/v1/quotations/foreign-institution-total"
    headers = GetHeaders(tr_id="HHKST03900300", mode="REAL")
    headers["tr_cont"] = "F"   # 첫 번째 페이지 요청
    params = {
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_INPUT_ISCD": stock_code,
        "FID_DIV_CLS_CODE": "0",   # 0:전체, 1:외국인, 2:기관
        "user_id": "",              # KIS 이 엔드포인트 필수 필드
    }

    res = requests.get(url, headers=headers, params=params, timeout=15, verify=False)
    data = res.json()
    rt_cd = data.get("rt_cd", "")

    # rt_cd=0: 정상, rt_cd=1: 다음 페이지 있음 (데이터는 유효)
    if res.status_code != 200 or rt_cd not in ("0", "1"):
        raise RuntimeError(
            f"[KIS foreign-total] 호출 실패 ({res.status_code}): {data.get('msg1', res.text[:200])}"
        )

    return {
        "output1": data.get("output1", {}),
        "output2": data.get("output2", []),
    }


# ─────────────────────────────────────────────────────────────
# 3. 종목별 프로그램매매 추이 (당일 체결 기준)
# ─────────────────────────────────────────────────────────────

def GetProgramTradeToday(stock_code: str) -> dict:
    """
    종목별 프로그램매매 추이(체결) — 당일 프로그램 순매수 시간대별.

    Returns:
        {
          "output1": { ... },
          "output2": [ { stck_cntg_hour, ... } ],
        }
    """
    url = f"{_base()}/uapi/domestic-stock/v1/quotations/comp-program-trade-today"
    headers = GetHeaders(tr_id="FHPPG04650100", mode="REAL")
    params = {
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_INPUT_ISCD": stock_code,
    }

    res = requests.get(url, headers=headers, params=params, timeout=15, verify=False)
    data = res.json()

    if res.status_code != 200 or data.get("rt_cd") != "0":
        raise RuntimeError(
            f"[KIS program-trade] 호출 실패 ({res.status_code}): {data.get('msg1', res.text[:200])}"
        )

    return {
        "output1": data.get("output1", {}),
        "output2": data.get("output2", []),
    }


# ─────────────────────────────────────────────────────────────
# 4. 회원사별 매매동향 (전 증권사)
# ─────────────────────────────────────────────────────────────

def GetMemberFirmTrading(stock_code: str) -> dict:
    """
    주식현재가 회원사 — 전체 증권사(회원사)별 당일 매도·매수 수량.

    Returns:
        {
          "output1": { ... },
          "output2": [
            {
              "mbcr_name": "삼성증권",
              "seln_qty":  "12345",
              "shnu_qty":  "23456",
              "ntby_qty":  "11111",
              ...
            },
            ...
          ]
        }
    """
    url = f"{_base()}/uapi/domestic-stock/v1/quotations/inquire-member"
    headers = GetHeaders(tr_id="FHKST01010600", mode="REAL")
    params = {
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_INPUT_ISCD": stock_code,
    }

    res = requests.get(url, headers=headers, params=params, timeout=15, verify=False)
    data = res.json()

    if res.status_code != 200 or data.get("rt_cd") != "0":
        raise RuntimeError(
            f"[KIS member-firm] 호출 실패 ({res.status_code}): {data.get('msg1', res.text[:200])}"
        )

    return {
        "output1": data.get("output1", {}),
        "output2": data.get("output2", []),
    }
