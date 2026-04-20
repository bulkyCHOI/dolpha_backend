"""
KIS API — 국내 업종/지수 목록 및 OHLCV 조회 모듈

업종기간별시세(FHKUP03500100) 엔드포인트 사용.
한 번 호출에 최대 100 거래일 반환 → 날짜 범위가 넓으면 자동 페이지네이션.

반환 DataFrame 스키마 (IndexOHLCV 저장용):
  index : Date (YYYY-MM-DD)
  open, high, low, close, volume, change
"""

import time
import warnings
import requests
import pandas as pd
from datetime import datetime, timedelta

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

from .auth import GetHeaders, get_url_base

_PATH = "uapi/domestic-stock/v1/quotations/inquire-index-daily-price"
_TR_ID = "FHKUP03500100"
_PAGE_SIZE = 100

# KIS 업종코드 정적 목록 (code, name, market)
# KOSPI: 0001~0099, KOSDAQ: 1001~1099
_INDEX_LIST = [
    # KOSPI 업종
    ("0001", "종합(KOSPI)", "KOSPI"),
    ("0002", "대형주", "KOSPI"),
    ("0003", "중형주", "KOSPI"),
    ("0004", "소형주", "KOSPI"),
    ("0005", "음식료품", "KOSPI"),
    ("0006", "섬유의복", "KOSPI"),
    ("0007", "종이목재", "KOSPI"),
    ("0008", "화학", "KOSPI"),
    ("0009", "의약품", "KOSPI"),
    ("0010", "비금속광물", "KOSPI"),
    ("0011", "철강금속", "KOSPI"),
    ("0012", "기계", "KOSPI"),
    ("0013", "전기전자", "KOSPI"),
    ("0014", "의료정밀", "KOSPI"),
    ("0015", "운수장비", "KOSPI"),
    ("0016", "유통업", "KOSPI"),
    ("0017", "전기가스업", "KOSPI"),
    ("0018", "건설업", "KOSPI"),
    ("0019", "운수창고업", "KOSPI"),
    ("0020", "통신업", "KOSPI"),
    ("0021", "금융업", "KOSPI"),
    ("0022", "은행", "KOSPI"),
    ("0023", "증권", "KOSPI"),
    ("0024", "보험", "KOSPI"),
    ("0025", "서비스업", "KOSPI"),
    ("0026", "제조업", "KOSPI"),
    # KOSDAQ 업종
    ("1001", "종합(KOSDAQ)", "KOSDAQ"),
    ("1002", "대형주", "KOSDAQ"),
    ("1003", "중형주", "KOSDAQ"),
    ("1004", "소형주", "KOSDAQ"),
    ("1005", "음식료·담배", "KOSDAQ"),
    ("1006", "섬유·의류", "KOSDAQ"),
    ("1007", "종이·목재", "KOSDAQ"),
    ("1008", "출판·매체복제", "KOSDAQ"),
    ("1009", "화학", "KOSDAQ"),
    ("1010", "제약", "KOSDAQ"),
    ("1011", "비금속", "KOSDAQ"),
    ("1012", "금속", "KOSDAQ"),
    ("1013", "기계·장비", "KOSDAQ"),
    ("1014", "전기·전자부품", "KOSDAQ"),
    ("1015", "의료·정밀기기", "KOSDAQ"),
    ("1016", "운송장비·부품", "KOSDAQ"),
    ("1017", "기타 제조", "KOSDAQ"),
    ("1018", "건설", "KOSDAQ"),
    ("1019", "유통", "KOSDAQ"),
    ("1020", "운송", "KOSDAQ"),
    ("1021", "금융", "KOSDAQ"),
    ("1022", "오락·문화", "KOSDAQ"),
    ("1023", "통신방송서비스", "KOSDAQ"),
    ("1024", "IT S/W & SVC", "KOSDAQ"),
    ("1025", "IT H/W", "KOSDAQ"),
]


def GetIndexListKR() -> pd.DataFrame:
    """
    KIS 업종코드 정적 목록을 DataFrame으로 반환.

    Returns:
        DataFrame (columns: Code, Name, Market)
    """
    return pd.DataFrame(_INDEX_LIST, columns=["Code", "Name", "Market"])


def _yyyymmdd(date_str: str) -> str:
    return date_str.replace("-", "")


def _fetch_page(index_code: str, start_yyyymmdd: str, end_yyyymmdd: str) -> list:
    url = f"{get_url_base('REAL')}/{_PATH}"
    headers = GetHeaders(tr_id=_TR_ID, mode="REAL")
    params = {
        "FID_COND_MRKT_DIV_CODE": "U",
        "FID_INPUT_ISCD": index_code,
        "FID_INPUT_DATE_1": start_yyyymmdd,
        "FID_INPUT_DATE_2": end_yyyymmdd,
        "FID_PERIOD_DIV_CODE": "D",
        "FID_ORG_ADJ_PRC": "0",
    }

    res = requests.get(url, headers=headers, params=params, timeout=15, verify=False)

    if res.status_code == 200 and res.json().get("rt_cd") == "0":
        return res.json().get("output2", [])
    else:
        print(f"[KIS 인덱스 OHLCV] 호출 실패 ({index_code}): {res.status_code} | {res.text[:200]}")
        return []


def GetIndexOhlcvKR(
    index_code: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    KIS API로 국내 업종 일봉 OHLCV 조회 (날짜 범위 기반).

    Args:
        index_code : KIS 업종코드 (예: "0001" = KOSPI 종합)
        start_date : 시작일 "YYYY-MM-DD"
        end_date   : 종료일 "YYYY-MM-DD"

    Returns:
        DataFrame (index=Date YYYY-MM-DD, columns=open/high/low/close/volume/change)
        또는 빈 DataFrame
    """
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    all_rows: list[dict] = []
    seen_dates: set[str] = set()

    page_end_dt = end_dt
    fail_count = 0

    while True:
        page_start_dt = max(start_dt, page_end_dt - timedelta(days=150))

        result = _fetch_page(
            index_code,
            page_start_dt.strftime("%Y%m%d"),
            page_end_dt.strftime("%Y%m%d"),
        )
        time.sleep(0.21)

        added = 0
        earliest_in_page = None

        for row in result:
            date_str = row.get("stck_bsop_date", "")
            if not date_str or date_str in seen_dates:
                continue

            close_val = row.get("bstp_nmix_prpr", "")
            if not close_val:
                continue

            row_dt = pd.to_datetime(date_str, format="%Y%m%d")
            if row_dt < start_dt or row_dt > end_dt:
                continue

            seen_dates.add(date_str)
            all_rows.append({
                "Date":   date_str,
                "open":   float(row.get("bstp_nmix_oprc", 0) or 0),
                "high":   float(row.get("bstp_nmix_hgpr", 0) or 0),
                "low":    float(row.get("bstp_nmix_lwpr", 0) or 0),
                "close":  float(close_val),
                "volume": float(row.get("acml_vol", 0) or 0),
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
    df.index = pd.to_datetime(df.index, format="%Y%m%d").strftime("%Y-%m-%d")
    df.index.name = "Date"

    df["change"] = (df["close"] - df["close"].shift(1)) / df["close"].shift(1)
    df[["open", "high", "low", "close", "volume", "change"]] = (
        df[["open", "high", "low", "close", "volume", "change"]].apply(pd.to_numeric)
    )

    return df
