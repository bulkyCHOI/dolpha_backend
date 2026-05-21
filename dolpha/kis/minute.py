"""
KIS API — 국내 주식 분봉 OHLCV 조회 모듈

KIS inquire-time-itemchartprice 엔드포인트 사용.
당일(또는 입력 시점)의 1분봉 데이터를 반환.

반환 스키마:
  [{
    "time": "HH:MM",         # 한국시간 시:분
    "datetime": "YYYY-MM-DD HH:MM:00",
    "open": float,
    "high": float,
    "low": float,
    "close": float,
    "volume": float,
  }, ...]  # 시간 오름차순 정렬
"""

import time
import warnings
from datetime import datetime
from typing import Generator, Iterator, Optional

import requests

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

from .auth import GetHeaders, get_url_base

_PATH = "uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice"
_TR_ID = "FHKST03010200"


def GetMinuteOhlcvKR(stock_code: str, end_hhmmss: Optional[str] = None) -> list[dict]:
    """
    KIS API로 국내 주식 당일 분봉 OHLCV 조회.

    KIS는 한 번 호출에 종료 시각 기준 과거 30건만 반환.
    pre-market(08:00~09:00) + 정규장(09:00~15:30) + after-market(15:30~20:00) 전체를 받기 위해 페이지네이션.

    Args:
        stock_code  : 종목코드 (6자리)
        end_hhmmss  : 조회 종료 시각 "HHMMSS" (None이면 현재 시각)
    Returns:
        분봉 리스트 (시간 오름차순). 데이터 없으면 빈 리스트.
    """
    url = f"{get_url_base('REAL')}/{_PATH}"
    headers = GetHeaders(tr_id=_TR_ID, mode="REAL")

    if end_hhmmss is None:
        now_hhmmss = datetime.now().strftime("%H%M%S")
        # 장 마감(15:30) 이후에 호출 시 현재 시각을 그대로 쓰면
        # 역방향 페이징 범위가 오후/야간으로 편향되어 09:00 이전 구간이 잘린다.
        # after-market 최대치(20:00)를 상한으로 두어 항상 정규장 전체를 포함.
        end_hhmmss = min(now_hhmmss, "200000")

    all_rows: dict[str, dict] = {}
    current_end = end_hhmmss

    # 최대 40페이지 (1200분, pre/after market 포함) 안전장치
    for _ in range(40):
        params = {
            "FID_ETC_CLS_CODE": "",
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": stock_code,
            "FID_INPUT_HOUR_1": current_end,
            "FID_PW_DATA_INCU_YN": "N",  # N: 당일, Y: 전일포함
        }

        res = requests.get(url, headers=headers, params=params, timeout=15, verify=False)
        if res.status_code != 200:
            print(f"[KIS 분봉] HTTP {res.status_code}: {res.text[:200]}")
            break

        body = res.json()
        if body.get("rt_cd") != "0":
            print(f"[KIS 분봉] rt_cd != 0: {body.get('msg1', body.get('msg_cd'))}")
            break

        rows = body.get("output2", [])
        if not rows:
            break

        new_count = 0
        earliest_hhmmss = current_end
        for r in rows:
            hhmmss = r.get("stck_cntg_hour")  # "HHMMSS"
            ymd = r.get("stck_bsop_date")  # "YYYYMMDD"
            if not hhmmss or not ymd:
                continue
            key = f"{ymd}{hhmmss}"
            if key in all_rows:
                continue
            try:
                all_rows[key] = {
                    "datetime": f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]} {hhmmss[:2]}:{hhmmss[2:4]}:00",
                    "time": f"{hhmmss[:2]}:{hhmmss[2:4]}",
                    "open": float(r.get("stck_oprc", 0) or 0),
                    "high": float(r.get("stck_hgpr", 0) or 0),
                    "low": float(r.get("stck_lwpr", 0) or 0),
                    "close": float(r.get("stck_prpr", 0) or 0),
                    "volume": float(r.get("cntg_vol", 0) or 0),
                }
                new_count += 1
                if hhmmss < earliest_hhmmss:
                    earliest_hhmmss = hhmmss
            except (ValueError, TypeError):
                continue

        # 진전 없으면 종료 (장 시작 시간보다 일찍 돌아간 경우)
        if new_count == 0:
            break

        # 08:00 이전이면 종료 (pre-market 포함)
        if earliest_hhmmss <= "080000":
            break

        # KIS API 레이트 리밋 방지 (초당 ~5회 제한)
        time.sleep(0.22)

        # 다음 페이지는 가장 이른 시각 1초 전부터 조회
        try:
            h, m, s = int(earliest_hhmmss[:2]), int(earliest_hhmmss[2:4]), int(earliest_hhmmss[4:6])
            total_sec = h * 3600 + m * 60 + s - 1
            if total_sec < 8 * 3600:
                break
            current_end = f"{total_sec // 3600:02d}{(total_sec % 3600) // 60:02d}{total_sec % 60:02d}"
        except (ValueError, TypeError):
            break

    return sorted(all_rows.values(), key=lambda x: x["datetime"])


_REGULAR_START = "090000"
_REGULAR_END   = "153000"


def _iter_minute_pages(stock_code: str, end_hhmmss: Optional[str] = None) -> Iterator[list[dict]]:
    """KIS 분봉(정규장 09:00~15:30, 가장 최근 거래일)을 페이지 단위로 yield.

    장 시작 전(09:00 이전)이라도 전 거래일 데이터를 반환.
    FID_PW_DATA_INCU_YN=Y(전일 포함)로 조회 후 가장 최근 영업일 날짜만 필터.
    """
    url = f"{get_url_base('REAL')}/{_PATH}"
    headers = GetHeaders(tr_id=_TR_ID, mode="REAL")

    seen_keys: set[str] = set()
    current_end = _REGULAR_END  # 항상 15:30 기준으로 역방향 페이지네이션
    target_ymd: Optional[str] = None  # 첫 응답에서 확정되는 최근 거래일

    for _ in range(25):
        params = {
            "FID_ETC_CLS_CODE": "",
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": stock_code,
            "FID_INPUT_HOUR_1": current_end,
            "FID_PW_DATA_INCU_YN": "Y",  # 전일 포함 → 장 전에도 전 거래일 데이터 조회 가능
        }

        res = requests.get(url, headers=headers, params=params, timeout=15, verify=False)
        if res.status_code != 200:
            break

        body = res.json()
        if body.get("rt_cd") != "0":
            break

        rows = body.get("output2", [])
        if not rows:
            break

        # 첫 페이지에서 가장 최근 거래일을 확정
        if target_ymd is None:
            dates = [r.get("stck_bsop_date") for r in rows if r.get("stck_bsop_date")]
            if not dates:
                break
            target_ymd = max(dates)

        batch: list[dict] = []
        new_count = 0
        earliest_hhmmss = current_end

        for r in rows:
            hhmmss = r.get("stck_cntg_hour")
            ymd = r.get("stck_bsop_date")
            if not hhmmss or not ymd:
                continue
            # 확정된 거래일 + 정규장 시간 범위만 허용
            if ymd != target_ymd or hhmmss < _REGULAR_START or hhmmss > _REGULAR_END:
                continue
            key = f"{ymd}{hhmmss}"
            if key in seen_keys:
                continue
            seen_keys.add(key)
            try:
                volume = float(r.get("cntg_vol", 0) or 0)
                # volume=0인 봉은 거래 미발생 phantom 봉 → 제외
                if volume <= 0:
                    continue
                batch.append({
                    "datetime": f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]} {hhmmss[:2]}:{hhmmss[2:4]}:00",
                    "time": f"{hhmmss[:2]}:{hhmmss[2:4]}",
                    "open": float(r.get("stck_oprc", 0) or 0),
                    "high": float(r.get("stck_hgpr", 0) or 0),
                    "low": float(r.get("stck_lwpr", 0) or 0),
                    "close": float(r.get("stck_prpr", 0) or 0),
                    "volume": volume,
                })
                new_count += 1
                if hhmmss < earliest_hhmmss:
                    earliest_hhmmss = hhmmss
            except (ValueError, TypeError):
                continue

        if batch:
            yield batch

        if new_count == 0:
            break

        if earliest_hhmmss <= _REGULAR_START:
            break

        time.sleep(0.22)

        try:
            h, m, s = int(earliest_hhmmss[:2]), int(earliest_hhmmss[2:4]), int(earliest_hhmmss[4:6])
            total_sec = h * 3600 + m * 60 + s - 1
            if total_sec < 9 * 3600:
                break
            current_end = f"{total_sec // 3600:02d}{(total_sec % 3600) // 60:02d}{total_sec % 60:02d}"
        except (ValueError, TypeError):
            break
