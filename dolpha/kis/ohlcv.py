"""
KIS API — 국내 주식 OHLCV 조회 모듈

KIS inquire-daily-itemchartprice 엔드포인트 사용.
한 번 호출에 최대 100 거래일 반환 → 날짜 범위가 넓으면 자동 페이지네이션.

반환 DataFrame 스키마 (stockCommon.GetOhlcv* 계열과 동일):
  index : Date (YYYY-MM-DD)
  open, high, low, close, volume, value, change
"""

import time
import warnings
import requests
import pandas as pd
from datetime import datetime, timedelta

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

from .auth import GetHeaders, get_url_base

# KIS OHLCV API
_PATH = "uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
_TR_ID = "FHKST03010100"

# 한 페이지당 최대 반환 거래일 수
_PAGE_SIZE = 100


def _yyyymmdd(date_str: str) -> str:
    """'YYYY-MM-DD' → 'YYYYMMDD'"""
    return date_str.replace("-", "")


def _fetch_page(stock_code: str, start_yyyymmdd: str, end_yyyymmdd: str, adj_ok: str) -> list:
    """
    KIS API 단일 페이지 호출.
    Returns:
        output2 list (거래일 내림차순 정렬됨)
    """
    url = f"{get_url_base('REAL')}/{_PATH}"
    fid_org_adj = "0" if adj_ok == "1" else "1"   # 0=수정주가, 1=원주가

    headers = GetHeaders(tr_id=_TR_ID, mode="REAL")
    params = {
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_INPUT_ISCD": stock_code,
        "FID_INPUT_DATE_1": start_yyyymmdd,
        "FID_INPUT_DATE_2": end_yyyymmdd,
        "FID_PERIOD_DIV_CODE": "D",
        "FID_ORG_ADJ_PRC": fid_org_adj,
    }

    res = requests.get(url, headers=headers, params=params, timeout=15, verify=False)

    if res.status_code == 200 and res.json().get("rt_cd") == "0":
        return res.json().get("output2", [])
    else:
        print(f"[KIS OHLCV] 호출 실패: {res.status_code} | {res.text[:200]}")
        return []


def GetOhlcvKR(
    stock_code: str,
    start_date: str,
    end_date: str,
    adj_ok: str = "1",
) -> pd.DataFrame | None:
    """
    KIS API로 국내 주식 일봉 OHLCV 조회 (날짜 범위 기반).

    Args:
        stock_code : 종목코드 (6자리, 예: "005930")
        start_date : 시작일 "YYYY-MM-DD"
        end_date   : 종료일 "YYYY-MM-DD"
        adj_ok     : "1" 수정주가(기본), "0" 원주가

    Returns:
        DataFrame (index=Date, columns=open/high/low/close/volume/value/change)
        또는 빈 DataFrame (데이터 없음)
    """
    start_dt = pd.to_datetime(start_date)
    end_dt   = pd.to_datetime(end_date)

    all_rows: list[dict] = []
    seen_dates: set[str] = set()

    # KIS는 end→start 역방향으로 100건씩 반환하므로
    # 페이지 끝을 end_date에서 시작해 조금씩 앞으로 이동한다.
    page_end_dt   = end_dt
    fail_count    = 0

    while True:
        page_start_dt = max(start_dt, page_end_dt - timedelta(days=150))  # 150 달력일 ≈ 100 거래일

        page_start_str = page_start_dt.strftime("%Y%m%d")
        page_end_str   = page_end_dt.strftime("%Y%m%d")

        result = _fetch_page(stock_code, page_start_str, page_end_str, adj_ok)
        time.sleep(0.21)   # API 속도 제한 대응

        added = 0
        earliest_in_page = None

        for row in result:
            if not row or row.get("stck_oprc", "") == "":
                continue

            date_str = row.get("stck_bsop_date", "")
            if not date_str or date_str in seen_dates:
                continue

            row_dt = pd.to_datetime(date_str, format="%Y%m%d")
            if row_dt < start_dt or row_dt > end_dt:
                continue

            seen_dates.add(date_str)
            all_rows.append({
                "Date":   date_str,
                "open":   float(row["stck_oprc"]),
                "high":   float(row["stck_hgpr"]),
                "low":    float(row["stck_lwpr"]),
                "close":  float(row["stck_clpr"]),
                "volume": float(row["acml_vol"]),
                "value":  float(row["acml_tr_pbmn"]),
            })
            added += 1

            if earliest_in_page is None or row_dt < earliest_in_page:
                earliest_in_page = row_dt

        if added == 0:
            fail_count += 1
            if fail_count >= 2:
                break
        else:
            fail_count = 0

        # 다음 페이지 : 이번 페이지에서 가장 오래된 날짜 하루 전까지
        if earliest_in_page is None or earliest_in_page <= start_dt:
            break

        page_end_dt = earliest_in_page - timedelta(days=1)
        if page_end_dt < start_dt:
            break

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df = df.set_index("Date")
    df = df.sort_index()

    # YYYY-MM-DD 형식으로 정규화
    df.index = pd.to_datetime(df.index, format="%Y%m%d").strftime("%Y-%m-%d")
    df.index.name = "Date"

    df.insert(6, "change",
              (df["close"] - df["close"].shift(1)) / df["close"].shift(1))
    df[["open", "high", "low", "close", "volume", "change"]] = (
        df[["open", "high", "low", "close", "volume", "change"]].apply(pd.to_numeric)
    )

    return df
